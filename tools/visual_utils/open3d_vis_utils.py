"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel(0)) # Suppress paint_uniform_color warning

box_colormap = [
    [1, 1, 1], # ignore
    [0, 0, 1], # car
    [1, 0.55, 0], # ped
    [0, 0.8, 0.8], # cyc
    [0.21568627, 0.49411765, 0.72156863],
    [0.89411765, 0.10196078, 0.10980392],
    [0.59607843, 0.30588235, 0.63921569],
    [1.        , 0.49803922, 0.        ],
    [1.        , 1.        , 0.2       ],
    [0.65098039, 0.3372549 , 0.15686275],
    [0.96862745, 0.50588235, 0.74901961],
    [0.6       , 0.6       , 0.6       ]
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba

def draw_scenes_msda(points, idx, gt_boxes, det_annos, draw_origin=False, min_score=0.2, use_linemesh=True):

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    

    # cmap = np.array(plt.get_cmap('Set1').colors)
    cmap = np.array([[49,131,106],[176,73,73],[25,97,120],[182,176,47]])/255
    for sid, key in enumerate(det_annos.keys()):
        points = points if sid == 0 else None
        mask = det_annos[key][idx]['score'] > min_score
        geom = get_geometries(points=points, 
                                ref_boxes=det_annos[key][idx]['boxes_lidar'][mask],                         
                                ref_scores=det_annos[key][idx]['score'][mask], 
                                ref_labels=[1 for i in range(len(det_annos[key][idx]['name'][mask]))],
                                ref_box_colors=cmap[sid % len(cmap)],
                                gt_boxes=gt_boxes, 
                                draw_origin=draw_origin, 
                                line_thickness=0.04,
                                use_linemesh=use_linemesh)
        for g in geom:                
            vis.add_geometry(g)

    ctr = vis.get_view_control()
    ctr.set_front([ 0.66741310889048566, -0.35675856751501511, 0.65366892735219662 ])
    ctr.set_lookat([ -18.284592676097365, 3.7960852036759234, -16.806735299460072 ])
    ctr.set_up([ -0.55585420737713021, 0.34547108891144618, 0.75609247243143607 ])
    ctr.set_zoom(0.21900000000000003)

    vis.get_render_option().point_size = 2.0
    vis.run()
    vis.destroy_window()

def draw_scenes(points=None, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, ref_box_colors=None, 
                point_colors=None, draw_origin=False, use_linemesh=True):

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    geom = get_geometries(points, gt_boxes=gt_boxes, 
                          ref_boxes=ref_boxes, ref_labels=ref_labels, 
                          ref_scores=ref_scores, ref_box_colors=ref_box_colors, 
                          point_colors=point_colors, draw_origin=draw_origin,
                          line_thickness=0.06, use_linemesh=use_linemesh)
    vis.clear_geometries()
    for g in geom:                
        vis.add_geometry(g)
    
    ctr = vis.get_view_control()  
    
    ctr.set_front([ -0.31094269624370807, -0.52800088868119233, 0.79027191599130253 ])
    ctr.set_lookat([ -3.9253878764499586, -4.1870200341400947, -16.570707875396788 ])
    ctr.set_up([ 0.41025289806528631, 0.67547432104665128, 0.61272098155326737 ])
    ctr.set_zoom(0.40)
    # ctr.set_front([ -0.85415171319858785, 0.0084795734346973951, 0.51995475541077896 ])
    # ctr.set_lookat([ 22.078260806001634, 1.0249602339143569, -2.8088354431826907 ])
    # ctr.set_up([ 0.51984622231746436, -0.012211597572028807, 0.85417257157263038 ])
    # ctr.set_zoom(0.219)

    # Original, zoom in, ego vehicle moving towards
    # ctr.set_front([ 0.59083558928204927, 0.44198102848405585, 0.6749563518464804 ])
    # ctr.set_lookat([ -22.160006279021506, -13.245452622148209, -17.37984247123935 ])
    # ctr.set_up([ -0.5805376677351477, -0.3480484359390435, 0.73609666660094375 ])
    # ctr.set_zoom(0.17900000000000005)

    # Wide, ego vehicle moving away
    # ctr.set_front([ 0.66741310889048566, -0.35675856751501511, 0.65366892735219662 ])
    # ctr.set_lookat([ -18.284592676097365, 3.7960852036759234, -16.806735299460072 ])
    # ctr.set_up([ -0.55585420737713021, 0.34547108891144618, 0.75609247243143607 ])
    # ctr.set_zoom(0.21900000000000003)

    vis.get_render_option().point_size = 2.0
    vis.run()
    vis.destroy_window()

def get_geometries(points, gt_boxes=None, ref_boxes=None, ref_labels=None, 
                   ref_scores=None, ref_box_colors=None, point_colors=None, 
                   draw_origin=False, line_thickness=0.06, use_linemesh=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    geometries = []

    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        geometries.append(axis_pcd)

    if points is not None:
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        # pts.paint_uniform_color(np.array([0.14, 0.34, 0.69]))
        pts.paint_uniform_color(np.array([0.72,0.72,0.72]))
        geometries.append(pts)

    if gt_boxes is not None:
        box = get_box(gt_boxes, (0, 0, 1.0), ref_labels, use_linemesh=use_linemesh)
        geometries.extend(box)

    if ref_boxes is not None:
        # color = ref_box_colors if ref_box_colors is not None else (0,0.6,0)
        # color = ref_box_colors if ref_box_colors is not None else (0.255,0.518,0.89)
        color = ref_box_colors if ref_box_colors is not None else (0.19215686, 0.59215686, 0.41568627)
        box = get_box(ref_boxes, color, ref_labels, ref_scores, line_thickness=line_thickness, use_linemesh=use_linemesh)
        geometries.extend(box)

    return geometries

def get_box(boxes, color=(0, 1, 0), ref_labels=None, score=None, line_thickness=0.06, use_linemesh=True): #0.02
    """
    Linemesh gives much thicker box lines but is extremely slow. Use only if you don't need to change viewpoint
    """
    ret_boxes = []
        
    # cmap = np.array([[49,131,106],[176,73,73],[160,155,30],[25,97,120],[0,0,0],[120,59,24],[120,24,110]])/255 # for track vis
    for i in range(boxes.shape[0]):
        # color = cmap[i % len(cmap)] # delete later
        line_set, box3d = translate_boxes_to_open3d_instance(boxes[i], use_linemesh=use_linemesh, line_thickness=line_thickness)
        if ref_labels is None: # Pred boxes
            if use_linemesh:
                for lines in line_set:
                    lines.paint_uniform_color(color)
                    ret_boxes.append(lines)
            else:
                line_set.paint_uniform_color(color)
                ret_boxes.append(line_set)

        else:  # GT boxes
            if use_linemesh:
                for lines in line_set:
                    lines.paint_uniform_color(box_colormap[ref_labels[i]])
                    # lines.paint_uniform_color(color)
                    ret_boxes.append(lines)      
            else:
                line_set.paint_uniform_color(box_colormap[ref_labels[i]])
                # line_set.paint_uniform_color(color)
                ret_boxes.append(line_set)
    return ret_boxes


def translate_boxes_to_open3d_instance(gt_boxes, use_linemesh=False, line_thickness=0.04): # 0.02 was original
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    if use_linemesh:
        # Use line_mesh for thicker box lines 
        line_mesh = LineMesh(line_set.points, line_set.lines, radius=line_thickness) 
        line_mesh_geoms = line_mesh.cylinder_segments

        return line_mesh_geoms, box3d
    else:
        return line_set, box3d


# ---------------------------------------------------
# LineMesh hotfix  to get thicker 3d bounding box lines
# Source: https://github.com/isl-org/Open3D/pull/738
# ---------------------------------------------------

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes
        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.
        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = open3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=open3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #     R=open3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=True)
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=open3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)