

# MS3D Parameter Explanation
MS3D configuration for each round of self-training on the given target dataset is located in `tools/cfgs/target_dataset/label_generation/roundN/cfgs/ps_config.yaml`. 

Here is an explanation the parameters.

```yaml
EXP_NAME: W_L_VMFI_TTA_PA_PC_VA_VC_64  # all files will be saved/searched with this prefix

DETS_TXT: /MS3D/tools/cfgs/target_nuscenes/label_generation/round1/cfgs/W_L_VMFI_TTA_PA_PC_VA_VC_64.txt  # pre-trained predictions
SAVE_DIR: /MS3D/tools/cfgs/target_nuscenes/label_generation/round1/ps_labels 
DATA_CONFIG: /MS3D/tools/cfgs/dataset_configs/nuscenes_dataset_da.yaml   # target dataset uda config

# PS_SCORE_TH and ENSEMBLE_KBF params are given as: [veh_th, ped_th, cyc_th]. Cyclist is not currently supported in MS3D.
PS_SCORE_TH: 
  POS_TH: [0.7,0.6,0.5] # labels with score above this are used as pseudo-labels
  NEG_TH: [0.2,0.1,0.3] # labels with score under this are removed

## MS3D Step 1: Ensemble pre-trained detectors from different sources
ENSEMBLE_KBF:
  DISCARD: [4, 4, 4] # if less than N predictions overlapping, we do not fuse the box
  RADIUS: [1.5, 0.3, 0.2] # find all centroids within a radius as fusion candidates
  NMS: [0.1, 0.3, 0.1]

## MS3D Step 2: Tracking with SimpleTrack
# giou as proposed by SimpleTrack is similar to iou if in range [0,1]
# RUNNING: use detection box as tracked box, and update kalman filter state
# REDUNDANCY: use kalman filter predicted box as tracked box
# running asso_th is (1-asso_th), redundancy asso_th is (asso_th) [see SimpleTrack github]
TRACKING:
  VEH_ALL:
    RUNNING:
        SCORE_TH: 0.5
        MAX_AGE_SINCE_UPDATE: 2
        MIN_HITS_TO_BIRTH: 2
        ASSO: giou
        ASSO_TH: 1.3
    REDUNDANCY:
        SCORE_TH: 0.1
        MAX_REDUNDANCY_AGE: 3
        ASSO_TH: -0.3
  VEH_STATIC:
    RUNNING:
        SCORE_TH: 0.3
        MAX_AGE_SINCE_UPDATE: 3
        MIN_HITS_TO_BIRTH: 2
        ASSO: iou_2d
        ASSO_TH: 0.7
    REDUNDANCY:
        SCORE_TH: 0.1
        MAX_REDUNDANCY_AGE: 2
        ASSO_TH: 0.5
  PEDESTRIAN:
    RUNNING:
        SCORE_TH: 0.2
        MAX_AGE_SINCE_UPDATE: 2
        MIN_HITS_TO_BIRTH: 2
        ASSO: giou
        ASSO_TH: 1.5
    REDUNDANCY:
        SCORE_TH: 0.1
        MAX_REDUNDANCY_AGE: 3
        ASSO_TH: -0.5

## MS3D Step 3: Refine all boxes temporally to get final pseudo-labels
TEMPORAL_REFINEMENT:

  # Retroactive Object Labeling
  TRACK_FILTERING: 
    # Number of confident detections required such that we consider the track as a pseudo-label
    MIN_DETS_ABOVE_POS_TH_FOR_TRACKS_VEH_ALL: 7
    MIN_DETS_ABOVE_POS_TH_FOR_TRACKS_VEH_STATIC: 3
    MIN_DETS_ABOVE_POS_TH_FOR_TRACKS_PED: 1
    
    # Use object tracks only if there's enough tracks
    MIN_NUM_STATIC_VEH_TRACKS: 7
    MIN_NUM_PED_TRACKS: 7

    # After a few self-training rounds, static peds may be good to use as well
    USE_STATIC_PED_TRACKS: false
  
  # Refine static vehicle position for each frame with ROLLING_KDE_WINDOW historical predictions
  # The score of the final refined box is max(MIN_STATIC_SCORE, fused_box_score)
  ROLLING_KBF:
    MIN_STATIC_SCORE: 0.8
    ROLLING_KDE_WINDOW: 10

  # Propagate the refined static box to the previous N_EXTRA_FRAMES and future N_EXTRA_FRAMES
  PROPAGATE_BOXES:
    MIN_STATIC_TRACKS: 10 # make sure there are at least MIN_STATIC_TRACKS boxes that are above POS_TH for each static vehicle 
    N_EXTRA_FRAMES: 40 # number of frames to propagate temporally
    DEGRADE_FACTOR: 0.98 # score of propagated boxes = DEGRADE_FACTOR*score
    MIN_SCORE_CLIP: 0.3 # unused at the moment
```        

Final pseudo-labels are given in the format: (x,y,z,dx,dy,dz,heading,class_id,score). Only positive class_ids are used for training.