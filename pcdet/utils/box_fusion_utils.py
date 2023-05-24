import torch
from scipy import stats
import numpy as np
from scipy.spatial import cKDTree
import pickle as pkl
from pathlib import Path
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils
from pcdet.utils.transform_utils import get_rotation_near_weighted_mean
from tqdm import tqdm 

def get_matching_boxes(boxes, discard=3, radius=1.0):
    """
    Get all centroids nearby and discard any detections 
    that do not match with any other source-detector. 
    
    discard=3 for now (cause KDE fails when it's 3 or less);
    might not be an issue if we use TTA
    """

    tree = cKDTree(boxes[:,:3])
    qbp_boxes = tree.query_ball_point(boxes[:,:3], r=radius)
    qbp_boxes_filt = [tuple(sets) for sets in qbp_boxes if len(sets) > discard]
    
    return list(set(qbp_boxes_filt))

def get_sample_inds_with_max_likelihood(samples, kde):
    if len(samples.shape) == 1:
        log_likelihood = kde(samples)    
    else:
        log_likelihood = kde(samples.T)    
    return np.where(log_likelihood == log_likelihood.max())[0][0]

def get_kde(data, bw_method=None, weights=None, return_max=False):
    """
    Returns the Kernel Density Estimate for a given data. 
    weights: a list of weights for each box
    return_max: returns the peak value of the KDE
    """    
    if len(data.shape) == 1:
        kde = stats.gaussian_kde(data, bw_method=bw_method, weights=weights)
    else:
        kde = stats.gaussian_kde(data.T, bw_method=bw_method, weights=weights) 
    
    if not return_max:
        return kde
    else:
        if len(data.shape) == 1:
            x = np.linspace(min(data), max(data),100)
            loglh = kde(x)
            if return_max:
                x_ind = np.where(loglh == loglh.max())[0][0] # If more than one max, then just choose 0th idx
                return kde, x[x_ind]
                
        elif len(data.shape) == 2:
            if data.shape[1] == 2:
                x,y = np.mgrid[min(data[:,0]):max(data[:,0]):20j, 
                               min(data[:,1]):max(data[:,1]):20j]
                xy = np.vstack([x.ravel(), y.ravel()])
                Z = np.reshape(kde(xy), x.shape)
                if return_max:
                    x_ind, y_ind = np.where(Z == Z.max())
                    x_ind, y_ind = x_ind.item(), y_ind.item()
                    return kde, np.array([x[x_ind][0], y[0][y_ind]])
            else:
                # KDE of 3D coordinates is best but slow
                # If 100j steps are too slow, we can drop to 20j (similar time to 2D kde)
                x,y,z = np.mgrid[min(data[:,0]):max(data[:,0]):20j, 
                                 min(data[:,1]):max(data[:,1]):20j,
                                 min(data[:,2]):max(data[:,2]):20j]
                xyz = np.vstack([x.ravel(), y.ravel(), z.ravel()])
                Z = np.reshape(kde(xyz), x.shape)
                x_ind, y_ind, z_ind = np.where(Z == Z.max())
                
                return kde, np.array([x[x_ind.item()][0][0], y[0][y_ind.item()][0], z[0][0][z_ind.item()]])
        else: 
            raise NotImplementedError



def find_nearest_gtbox(frame_gt_boxes, combined_box, return_iou=True):
    """
    Find closest GT to our given box
    """
    gt_tree = cKDTree(frame_gt_boxes[:,:3])
    nearest_gt = gt_tree.query_ball_point(combined_box[:3].reshape(1,-1), r=3.0)
    if len(nearest_gt[0]) == 0:        
        return None
    nearest_gt_box = frame_gt_boxes[nearest_gt[0][0]]
    if return_iou:
        gt_box = np.reshape(nearest_gt_box, (1, -1))
        gt_box_cuda = torch.from_numpy(gt_box.astype(np.float32)).cuda()
        combined_box_cuda = torch.from_numpy(combined_box.reshape(1,-1).astype(np.float32)).cuda()

        iou = iou3d_nms_utils.boxes_iou3d_gpu(gt_box_cuda, combined_box_cuda)
        return (nearest_gt_box, iou.item())
    else:
        return nearest_gt_box
    
def nms(boxes, score, thresh=0.05):
    boxs_gpu = torch.from_numpy(boxes.astype(np.float32) ).cuda()
    scores_gpu = torch.from_numpy(score.astype(np.float32) ).cuda()

    nms_inds = iou3d_nms_utils.nms_gpu(boxs_gpu, scores_gpu, thresh=thresh)
    nms_mask = np.zeros(boxs_gpu.shape[0], dtype=bool)
    nms_mask[nms_inds[0].cpu().numpy()] = 1
    return nms_mask            

def compute_iou(boxes_a, boxes_b):
    gt_box_a, _ = common_utils.check_numpy_to_torch(boxes_a)
    gt_box_b, _ = common_utils.check_numpy_to_torch(boxes_b)
    gt_box_a, gt_box_b = gt_box_a.cuda(), gt_box_b.cuda()
    # get ious
    iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(gt_box_a[:, :7], gt_box_b[:, :7]).cpu()

    ious, match_idx = torch.max(iou_matrix, dim=1)
    ious, match_idx = ious.numpy(), match_idx.numpy()
    gt_box_a, gt_box_b = gt_box_a.cpu().numpy(), gt_box_b.cpu().numpy()

    match_pairs_idx = np.concatenate((
        np.array(list(range(gt_box_a.shape[0]))).reshape(-1, 1),
        match_idx.reshape(-1, 1)), axis=1)
    return ious, match_pairs_idx

def load_src_paths_txt(src_paths_txt):
    with open(src_paths_txt, 'r') as f:
        pkl_pths = [line.split('\n')[0] for line in f.readlines()]

    det_annos = {}
    for idx, pkl_pth in enumerate(pkl_pths):
        with open(pkl_pth, 'rb') as f:
            if not Path(pkl_pth).is_absolute():
                pkl_pth = Path(pkl_pth).resolve()     
            
            label_str = str(pkl_pth).split('/')[3:5]
            label_str.append(str(pkl_pth).split('/')[-2])
            label = '.'.join(label_str) # source-dataset.detector.eval_tag
            det_annos[label] = pkl.load(f)
    return det_annos

def combine_box_pkls(det_annos, score_th=0.1):
    combined_dets = []
    len_data = len(det_annos[list(det_annos.keys())[0]])
    for idx in tqdm(range(len_data), total=len_data, desc='aggregating_proposals'):
        frame_dets = {}
        frame_dets['boxes_lidar'], frame_dets['score'], frame_dets['source'], frame_dets['source_id'], frame_dets['frame_id'], frame_dets['class_ids'], frame_dets['names'] = [],[],[],[],[],[],[]
        for src_id, key in enumerate(det_annos.keys()):    
            frame_dets['frame_id'] = det_annos[key][idx]['frame_id']
            score_mask = det_annos[key][idx]['score'] > score_th            
            frame_dets['boxes_lidar'].extend(det_annos[key][idx]['boxes_lidar'][score_mask])
            frame_dets['score'].extend(det_annos[key][idx]['score'][score_mask])
            frame_dets['source'].extend([key for i in range(len(det_annos[key][idx]['score'][score_mask]))])
            frame_dets['source_id'].extend([src_id for i in range(len(det_annos[key][idx]['score'][score_mask]))])
            frame_dets['class_ids'].extend(det_annos[key][idx]['pred_labels'][score_mask])
            frame_dets['names'].extend(det_annos[key][idx]['name'][score_mask])

        frame_dets['boxes_lidar'] = np.vstack(frame_dets['boxes_lidar']) if len(frame_dets['score']) != 0  else np.array([]).reshape(-1,7)
        frame_dets['score'] = np.hstack(frame_dets['score']) if len(frame_dets['score']) != 0 else np.array([])
        frame_dets['source'] = np.array(frame_dets['source']) if len(frame_dets['score']) != 0 else np.array([])
        frame_dets['source_id'] = np.array(frame_dets['source_id']) if len(frame_dets['score']) != 0 else np.array([])
        frame_dets['class_ids'] = np.array(frame_dets['class_ids'], dtype=np.int32) if len(frame_dets['score']) != 0 else  np.array([])
        frame_dets['names'] = np.array(frame_dets['names']) if len(frame_dets['score']) != 0 else  np.array([])
        combined_dets.append(frame_dets)
        assert frame_dets['class_ids'].shape == frame_dets['score'].shape
    return combined_dets

def kde_fusion(boxes, src_weights, bw_c=1.0, bw_dim=2.0, bw_ry=0.1, bw_cls=0.5, bw_score=2.0):
    """
    Combines the centroids, dims, ry and scores of multiple predicted boxes
    Args:
        boxes: (N,9) np array. Boxes for filtering
        src_weights : (list). List of weights for each source detector

    Returns:
        combined box: (9) np array. A final box with params [x,y,z,dx,dy,dz,ry,score,class_id]

    Centroids are selected rather than aggregated as aggregating can lead to odd centering
    Rotations are selected as it is quite sensitive to aggregation
    Dimensions/score are the peak value of the KDE since we want to factor in the sizing/cls conf of diff detectors                    
    """
    def get_kde_value(data, src_weights, bw, return_max=False, verbose=False):
        """
        If data points are too similar, KDE will fail. This is because
        the covariance matrix is not invertible because the values are too close 
        (zero covriance on some diagonal elements).
        
        In these cases, we just take the weighted average.
        """
        if return_max:
            try:
                _, new_val = get_kde(data, return_max=return_max, weights=src_weights, bw_method=bw)
                return new_val
            except Exception as e:
                if verbose:
                    print(f'data:{data}, error:{e}')
                return np.average(data, axis=0, weights=src_weights)
        else:
            try:
                kde = get_kde(data, return_max=return_max, weights=src_weights, bw_method=bw)
                new_inds = get_sample_inds_with_max_likelihood(data, kde)    
                return new_inds
            except Exception as e:
                if verbose:
                    print(f'data:{data}, error:{e}')
                return None
        
    centroids = boxes[:,:3]
    centroid_weight = src_weights['centroid'] if 'centroid' in src_weights.keys() else src_weights['conf_score']
    det = np.linalg.det(np.cov(centroids.T, rowvar=1,bias=False))
    if det < 1e-7:
        # Split into separate KDE for xy, and KDE for z                
        new_cxy_inds = get_kde_value(centroids[:,:2], centroid_weight, bw_c, return_max=False)        
        if new_cxy_inds is not None:
            new_cxy = centroids[:,:2][new_cxy_inds]  
        else:
            new_cxy = np.average(centroids[:,:2], axis=0, weights=centroid_weight)
              
        new_cz_inds = get_kde_value(centroids[:,2], centroid_weight, bw_c, return_max=False)        
        if new_cz_inds is not None:
            new_cz = centroids[:,2][new_cz_inds]
        else:
            new_cz = np.average(centroids[:,2], axis=0, weights=centroid_weight)
        
        new_cxyz = np.hstack([new_cxy, new_cz])
    else:
        # KDE for xyz
        new_cxyz_inds = get_kde_value(centroids, centroid_weight, bw_c, return_max=False)
        if new_cxyz_inds is not None:
            new_cxyz = centroids[new_cxyz_inds]
        else:
            new_cxyz = np.average(centroids, axis=0, weights=centroid_weight)
    
    new_dx = get_kde_value(boxes[:,3], None, bw_dim, return_max=True)
    new_dy = get_kde_value(boxes[:,4], None, bw_dim, return_max=True)
    new_dz = get_kde_value(boxes[:,5], None, bw_dim, return_max=True)
    
    heading_weight = src_weights['heading'] if 'heading' in src_weights.keys() else src_weights['conf_score']
    sin_rys = np.sin(boxes[:,6])
    ry_ind = get_kde_value(sin_rys, heading_weight, bw_ry, return_max=False)    
    if ry_ind is not None:
        new_ry = boxes[:,6][ry_ind]
    else:
        new_ry = get_rotation_near_weighted_mean(boxes[:,6])
    
    cls_weight = src_weights['class'] if 'class' in src_weights.keys() else src_weights['conf_score']
    cls_ind = get_kde_value(boxes[:,7], cls_weight, bw_cls, return_max=False)
    if cls_ind is not None:
        new_class = boxes[:,7][cls_ind]
    else:
        # Return majority class
        unique, counts = np.unique(boxes[:,7], return_counts=True)
        new_class = int(unique[np.argmax(counts)])
    
    score_weight = src_weights['score'] if 'score' in src_weights.keys() else src_weights['conf_score']
    new_score = get_kde_value(boxes[:,8], score_weight, bw_score, return_max=True)
    
    combined_box = np.hstack([new_cxyz[0], new_cxyz[1], new_cxyz[2], new_dx, new_dy, new_dz, new_ry, new_class, new_score])
    return combined_box


def wbf(matched_boxes, src_weights=None):
    """
    Weighted Boxes Fusion (WBF)
    This function uses the official WBF code from https://github.com/ZFTurbo/Weighted-Boxes-Fusion

    Solovyev, R., Wang, W., & Gabruseva, T. (2021). Weighted boxes fusion: 
    Ensembling boxes from different object detection models. Image and Vision 
    Computing, 107, 104117.

    WBF in 3D does not deal with headings and expects all box corners to be normalized to [0,1]
    If you wish to use this, copy over and import the functions from:
    https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf_3d.py
    """
    def normalize(c1, c2, dim, min_corner):
        if min_corner:
            normed = (c1[:,dim] - min(c1[:,dim]))/(max(c2[:,dim])-min(c1[:,dim]))
        else:
            normed = (c2[:,dim] - min(c1[:,dim]))/(max(c2[:,dim])-min(c1[:,dim]))
        return normed

    def unnormalize(normed, c1, c2, dim):
        total = max(c2[:,dim])-min(c1[:,dim])
        minval = min(c1[:,dim])
        return normed*total + minval
    
    wbf_score = np.average(matched_boxes[:,-1], axis=0)
    wbf_heading = get_rotation_near_weighted_mean(matched_boxes[:,6])
    wbf_centroid = np.average(matched_boxes[:,:3], axis=0)
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(matched_boxes)
    template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]

    centre_offset = matched_boxes[:,:3] - wbf_centroid
    corners_offset = corners3d + centre_offset[..., np.newaxis].reshape(-1,1,3)
    opp_corner1 = corners_offset[:,2]
    opp_corner2 = corners_offset[:,4]
    normed_x1 = normalize(opp_corner1, opp_corner2, 0, min_corner=True)
    normed_x2 = normalize(opp_corner1, opp_corner2, 0, min_corner=False)
    normed_y1 = normalize(opp_corner1, opp_corner2, 1, min_corner=True)
    normed_y2 = normalize(opp_corner1, opp_corner2, 1, min_corner=False)
    normed_z1 = normalize(opp_corner1, opp_corner2, 2, min_corner=True)
    normed_z2 = normalize(opp_corner1, opp_corner2, 2, min_corner=False)
    normed_opp = np.stack([normed_x1,normed_y1,normed_z1,normed_x2,normed_y2,normed_z2]).T
    boxes_list = normed_opp.reshape(-1,1,6).tolist()
    scores_list = matched_boxes[:,8].reshape(-1,1).tolist()
    labels_list = matched_boxes[:,7].reshape(-1,1).tolist()
    weights = np.ones(len(matched_boxes[:,7])).tolist()
    boxes, scores, labels = weighted_boxes_fusion_3d(boxes_list, scores_list, 
                                                     labels_list, weights=weights, iou_thr=0.2, skip_box_thr=0.0)
    boxes = boxes[0]
    wbf_box = np.array([unnormalize(boxes[0], opp_corner1, opp_corner2, 0),
                       unnormalize(boxes[1], opp_corner1, opp_corner2, 1),
                       unnormalize(boxes[2], opp_corner1, opp_corner2, 2),
                       unnormalize(boxes[3], opp_corner1, opp_corner2, 0),
                       unnormalize(boxes[4], opp_corner1, opp_corner2, 1),
                       unnormalize(boxes[5], opp_corner1, opp_corner2, 2)])
    
    l_offset = wbf_box[3]-(wbf_box[3]-wbf_box[0])/2
    w_offset = wbf_box[4]-(wbf_box[4]-wbf_box[1])/2
    h_offset = wbf_box[5]-(wbf_box[5]-wbf_box[2])/2

    box_centre = wbf_centroid - np.array([l_offset, w_offset, h_offset])
    dims = np.array([(wbf_box[3]-wbf_box[0]), (wbf_box[4]-wbf_box[1]), (wbf_box[5]-wbf_box[2])])
    wbf_box = np.hstack([box_centre, dims, wbf_heading, 1, wbf_score])
    return wbf_box

def wbf_myimplementation(matched_boxes, src_weights):
    """
    My 3D implementation of Weighted Boxes Fusion (WBF) which performs similarly to the official
    without needing to normalize box corners.

    The main idea of WBF is averaging the box corners with the confidence as weights.
    In 3D we can't just average box corners in the lidar frame as 3D boxes are oriented,
    unlike 2D boxes which are axis-aligned.
    """
    wbf_score = np.average(matched_boxes[:,-1], axis=0)
    wbf_heading = get_rotation_near_weighted_mean(matched_boxes[:,6])
    wbf_centroid = np.average(matched_boxes[:,:3], axis=0)

    # Get box in canonical frame
    boxes3d, _ = common_utils.check_numpy_to_torch(matched_boxes)
    template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    
    # Offset each box relative to the mean of all the box centroids
    centre_offset = matched_boxes[:,:3] - wbf_centroid
    corners_offset = corners3d + centre_offset[..., np.newaxis].reshape(-1,1,3)
    
    # Average the box corners
    wbf_box_corners = np.zeros((8,3))
    for i in range(wbf_box_corners.shape[0]):
        wbf_box_corners[i] = np.average(corners_offset[:,i], axis=0, weights=src_weights)
        
    # Recompute the centroid
    l_halfdist = (max(wbf_box_corners[:,0])-min(wbf_box_corners[:,0]))/2
    l_offset = max(wbf_box_corners[:,0])-l_halfdist

    w_halfdist = (max(wbf_box_corners[:,1])-min(wbf_box_corners[:,1]))/2
    w_offset = max(wbf_box_corners[:,1])-w_halfdist

    h_halfdist = (max(wbf_box_corners[:,2])-min(wbf_box_corners[:,2]))/2
    h_offset = max(wbf_box_corners[:,2]) - h_halfdist

    # Get final box params in OpenPCDet format (x,y,z,l,w,h,ry,class,score) - hardcoded class here
    box_centre = wbf_centroid - np.array([l_offset, w_offset, h_offset])
    dims = np.array([2*l_halfdist, 2*w_halfdist, 2*h_halfdist])
    wbf_box = np.hstack([box_centre, dims, wbf_heading, 1, wbf_score])
    return wbf_box

def wbf_p(matched_boxes, src_weights):
    """
    Weighted box fusion like WBF but just taking weighted average of box 
    parameters instead of the corners.
    """
    wbfp = np.average(matched_boxes[:,:6], axis=0, weights=src_weights)
    wbfp_score = np.average(matched_boxes[:,-1], axis=0)
    wbfp_heading = get_rotation_near_weighted_mean(matched_boxes[:,6])
    wbfp = np.hstack([wbfp, wbfp_heading, 1, wbfp_score])
    return wbfp

def label_fusion(boxes_lidar, fusion_name, discard, radius, nms_thresh=0.05, weights=None):
    """
    boxes_lidar (N,9): array of box proposals from src detectors
    
    return: 
        fused_boxes (N,9): fused box proposals
    """
    matched_inds_list = get_matching_boxes(boxes_lidar, discard=discard, radius=radius)
        
    # Aggregate these into one box
    combined_frame_boxes, num_matched_boxes = [], []
    for m_box_inds in matched_inds_list:

        matched_boxes = boxes_lidar[list(m_box_inds)]
        unique = np.unique(matched_boxes[:,:3].round(decimals=4), axis=0)
        if len(unique) < discard:
            continue
        fusion_func = globals()[fusion_name]
        m_box_weights = {}
        m_box_weights['conf_score'] = matched_boxes[:,8]
        if weights is not None:
            for k,v in weights.items():
                m_box_weights[k] = weights[k][list(m_box_inds)]*matched_boxes[:,8]
        
        combined_box = fusion_func(matched_boxes, src_weights=m_box_weights)        
        combined_frame_boxes.append(combined_box)
        num_matched_boxes.append(len(unique))
    
    if combined_frame_boxes:
        num_dets_per_box = np.array(num_matched_boxes)
        fused_boxes = np.array(combined_frame_boxes)
        nms_mask = nms(fused_boxes[:,:7], fused_boxes[:,8], thresh=nms_thresh)
        fused_boxes = fused_boxes[nms_mask]
        num_dets_per_box = num_dets_per_box[nms_mask]
    else:
        fused_boxes = np.empty((0,9))
        num_dets_per_box = np.empty(0)
        
    return fused_boxes.astype(np.float32), num_dets_per_box.astype(int)
