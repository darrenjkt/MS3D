from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
import nuscenes_utils
from pathlib import Path

data_path = Path('/OpenPCDet') / 'data' / 'nuscenes' / 'v1.0-trainval'
version = 'v1.0-trainval'
nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

available_scenes = nuscenes_utils.get_available_scenes(nusc)
available_scene_names = [s['name'] for s in available_scenes]
train_scenes = list(filter(lambda x: x in available_scene_names, splits.train))
# train_scenes_tokens = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])

stoken2name = dict([(available_scenes[available_scene_names.index(s)]['token'],s) for s in train_scenes])

# Count num vehicles in scene
cls_of_interest = ['vehicle.bus.rigid','vehicle.car','vehicle.truck']
scene_veh_dict = {}
prev_scene_token = ''
for index, sample in enumerate(nusc.sample):
    if sample['scene_token'] not in stoken2name.keys():
        continue
    cur_scene_token = sample['scene_token']
    if cur_scene_token != prev_scene_token:
        scene_veh_dict[cur_scene_token] = 0
        prev_scene_token = cur_scene_token

    anns = sample['anns']
    for ann in anns:
        ann_meta = nusc.get('sample_annotation', ann)
        if ann_meta['category_name'] in cls_of_interest:
            scene_veh_dict[cur_scene_token] += 1


scene_nframes_dict = dict([(scene['name'],scene['nbr_samples']) for scene in nusc.scene])
sorted_scenes = [(stoken2name[k],v) for k, v in sorted(scene_veh_dict.items(), key=lambda item: item[1], reverse=True)]

# Choose number of scenes
num_scenes = 190
selected_scenes = [k for k,v in sorted_scenes[:num_scenes]]
sel_num_frames = sum([scene_nframes_dict[sname] for sname in selected_scenes])

remaining_scenes = [k for k,v in sorted_scenes[num_scenes:]]
rem_num_frames = sum([scene_nframes_dict[sname] for sname in remaining_scenes])

save_pth = str(data_path / 'ImageSets' / f'custom_train_scenes_{num_scenes}.txt')
with open(save_pth,'w') as f:
    f.write('\n'.join(selected_scenes))
print(f'Custom_train_scenes saved at {save_pth}\n{num_scenes} scenes, {sel_num_frames} frames')

save_pth = str(data_path / 'ImageSets' / f'custom_train_scenes_{len(scene_veh_dict)-num_scenes}.txt')
with open(save_pth,'w') as f:
    f.write('\n'.join(selected_scenes))
print(f'Custom_train_scenes saved at {save_pth}\n{len(scene_veh_dict)-num_scenes} scenes, {rem_num_frames} frames')
