

# MS3D Parameter Explanation
MS3D configuration for each round of self-training on the given target dataset is located in `tools/cfgs/target_dataset/label_generation/roundN/cfgs/ps_config.yaml`. 

Here is an explanation the parameters.

```yaml
# all files will be saved/searched with this prefix
EXP_NAME: W_L_VMFI_TTA_PA_PC_VA_VC_64  

DETS_TXT: /MS3D/tools/cfgs/target_nuscenes/label_generation/round1/cfgs/W_L_VMFI_TTA_PA_PC_VA_VC_64.txt  # pre-trained predictions
SAVE_DIR: /MS3D/tools/cfgs/target_nuscenes/label_generation/round1/ps_labels 
DATA_CONFIG: /MS3D/tools/cfgs/dataset_configs/nuscenes_dataset_da.yaml   # target dataset uda config

# PS_SCORE_TH and ENSEMBLE_KBF params are given as: [veh_th, ped_th, cyc_th]. Cyclist is not currently supported in MS3D.
PS_SCORE_TH: 
  POS_TH: [0.7,0.6,0.5] # box score above this -> use as pseudo-labels
  NEG_TH: [0.2,0.1,0.3] # box score below this -> remove in KBF step

## MS3D Step 1: Ensemble pre-trained detectors
ENSEMBLE_KBF:
  DISCARD: [4, 4, 4] # discard if less than N predictions overlapping
  RADIUS: [1.5, 0.3, 0.2] # select fusion candidates by radius
  NMS: [0.1, 0.3, 0.1]

## MS3D Step 2: Tracking with SimpleTrack (explained below)
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
    # Use object tracks only if there's enough tracks
    MIN_NUM_STATIC_VEH_TRACKS: 7
    MIN_NUM_PED_TRACKS: 7
    
    # Number of confident detections required for each track
    MIN_DETS_ABOVE_POS_TH_FOR_TRACKS_VEH_ALL: 7
    MIN_DETS_ABOVE_POS_TH_FOR_TRACKS_VEH_STATIC: 3
    MIN_DETS_ABOVE_POS_TH_FOR_TRACKS_PED: 1      

    # May be able to use static after a few self-training rounds
    USE_STATIC_PED_TRACKS: false
  
  # Refine vehicle position with ROLLING_KDE_WINDOW historical predictions
  # Final refined box score is max(MIN_STATIC_SCORE, fused_box_score)
  ROLLING_KBF:
    MIN_STATIC_SCORE: 0.8
    ROLLING_KDE_WINDOW: 10

  # Propagate refined static box N_EXTRA_FRAMES in the past/future
  PROPAGATE_BOXES:
    MIN_STATIC_TRACKS: 10 
    N_EXTRA_FRAMES: 40 # too high can lead to more false positives
    DEGRADE_FACTOR: 0.98 # propagated box score = DEGRADE_FACTOR*orig_score
    MIN_SCORE_CLIP: 0.3 # unused at the moment
```        

**Final pseudo-labels** are given in the format: (x,y,z,dx,dy,dz,heading,class_id,score). Only positive class_ids are used for training.

### Tracker details
We use SimpleTrack for the tracker with `giou` and `iou_2d` for association and Kalman Filter (KF) as the motion model.

Generalized IOU `giou` as proposed by SimpleTrack is similar to IOU if in range [0,1], but also allows association if the two boxes are not overlapping.

Below we provide a quick overview of the tracker configs:
```yaml
RUNNING: # use detection box as tracked box, update KF state
    SCORE_TH: 0.5 # boxes above 0.5 are counted as valid detection box
    MAX_AGE_SINCE_UPDATE: 2 # end track if no boxes > 0.5 for 2 time steps
    MIN_HITS_TO_BIRTH: 2 # start track only if 2 consecutive boxes > 0.5
    ASSO: iou_2d
    ASSO_TH: 0.7 
REDUNDANCY: # no detection box, so we use KF prediction as tracked box
    SCORE_TH: 0.1 # boxes above 0.1 prolongs the track lifespan
    MAX_REDUNDANCY_AGE: 3 # end if no running boxes after 3 redundancy boxes
    ASSO_TH: 0.3
```
With `ASSO_TH`, the running asso_th refers to (1-asso_th) and redundancy asso_th is (asso_th). For example:
- RUNNING `iou_2d=0.7` means the track is associated if the boxes IOU > 0.3 overlap. 
- REDUNDANCY `iou_2d=0.3` means boxes are associated if IOU > 0.3. 

It's an odd implementation quirk in SimpleTrack. Refer to this issue in their [github](https://github.com/tusen-ai/SimpleTrack/issues/29).