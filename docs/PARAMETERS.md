

# MS3D Parameter Explanation
MS3D is predominantly in the "SELF_TRAIN" section of the `config.yaml`. Here is an explanation the parameters.

```yaml
SELF_TRAIN:
    SCORE_THRESH: 0.6 # only boxes above 0.6 score are used as pseudo-labels
    NEG_THRESH: 0.4 # any labels under this are removed
    UPDATE_PSEUDO_LABEL: [-1] # update pseudo-labels at specified epochs
    UPDATE_PSEUDO_LABEL_INTERVAL: 100 # or set interval to update pseudo-labels
    MS_DETECTOR_PS:
        FUSION: kde_fusion
        # DISCARD: If less than N predictions, we do not fuse the box
        # RADIUS: Find all centroids within distance "RADIUS" as fusion candidates
        ACCUM1:
            PATH: '/OpenPCDet/tools/cfgs/target-lyft/det_1f_paths_5hz.txt' 
            DISCARD: 4 
            RADIUS: 2.0
        ACCUM16:
            PATH: '/OpenPCDet/tools/cfgs/target-lyft/det_16f_paths.txt'
            DISCARD: 3
            RADIUS: 1.0

        MIN_STATIC_SCORE: 0.7 # alpha (in the paper)
        MIN_DETS_FOR_TRACK_1F: 3 
        MIN_DETS_FOR_TRACK_16F: 3
        ROLLING_KDE_WINDOW: 16 # H (in the paper)
        PROPAGATE_STATIC_BOXES:         
            ENABLED: True
            MIN_DETS: 7 # if track has less than MIN_DETS we do not propagate
            N_EXTRA_FRAMES: 40
            DEGRADE_FACTOR: 0.95 # beta (in the paper)
            MIN_SCORE_CLIP: 0.5
        TRACKING:
            ACCUM1_CFG: /OpenPCDet/tracker/configs/msda_configs/msda_1frame_giou.yaml
            ACCUM16_CFG: /OpenPCDet/tracker/configs/msda_configs/msda_16frame_iou.yaml

    INIT_PS: None # you can set an existing ps_label_e0.pkl here

    # optional: can do a pseudo-label update with UPDATE_PSEUDO_LABEL: [22,26]
    MEMORY_ENSEMBLE:
        ENABLED: True
        NAME: simplified_cons_ensemble
        IOU_THRESH: 0.1
```        

MEMORY_ENSEMBLE `simplified_cons_ensemble` updates the pseudo-labels by replacing existing pseudo-labels with the detector's prediction. When we propagate the static boxes across all frames, it can be thought of as an object prior. In some cases refining these object priors help to increase number of labels and improve performance. We found that ST3D's consistency ensemble often generates a lot of false positives.