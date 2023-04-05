from pathlib import Path
import os
import numpy as np
import pickle

def check_sequence_name_with_all_version(sequence_file):
    if not sequence_file.exists():
        found_sequence_file = sequence_file
        for pre_text in ['training', 'validation', 'testing']:
            if not sequence_file.exists():
                temp_sequence_file = Path(str(sequence_file).replace('segment', pre_text + '_segment'))
                if temp_sequence_file.exists():
                    found_sequence_file = temp_sequence_file
                    break
        if not found_sequence_file.exists():
            found_sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))
        if found_sequence_file.exists():
            sequence_file = found_sequence_file
    return sequence_file

root_path = Path('/MS3D/data/waymo')
split_dir = root_path / 'ImageSets' / 'train.txt'
sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
waymo_infos = []
seq_name_to_infos = {}
num_skipped_infos = 0
for k in range(len(sample_sequence_list)):
    sequence_name = os.path.splitext(sample_sequence_list[k])[0]
    info_path = root_path / 'waymo_processed_data_v0_5_0' / sequence_name / ('%s.pkl' % sequence_name)
    info_path = check_sequence_name_with_all_version(info_path)
    if not info_path.exists():
        num_skipped_infos += 1
        continue
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
        waymo_infos.extend(infos)

    seq_name_to_infos[infos[0]['point_cloud']['lidar_sequence']] = infos

infos.extend(waymo_infos[:])
print(f'Loaded Waymo: {len(infos)} frames')

prev_seq = ''
seq_nvehicles = {}
seq_nframes = {}
for info in infos:
    cur_seq = info['point_cloud']['lidar_sequence']
    if cur_seq != prev_seq:
        seq_nvehicles[cur_seq] = 0
        seq_nframes[cur_seq] = 0
        prev_seq = cur_seq

    num_vehicles = len(np.argwhere(info['annos']['name'] == 'Vehicle'))
    seq_nvehicles[cur_seq] += num_vehicles
    seq_nframes[cur_seq] += 1

sorted_scenes = [(k,v) for k, v in sorted(seq_nvehicles.items(), key=lambda item: item[1], reverse=True)]

num_scenes = 190
selected_scenes = [k for k,v in sorted_scenes[:num_scenes]]
num_sel_frames = sum([seq_nframes[sname] for sname in selected_scenes])    
selected_scenes = [scenes + '.tfrecord' for scenes in selected_scenes]

remaining_scenes = [k for k,v in sorted_scenes[num_scenes:]]
num_rem_frames = sum([seq_nframes[sname] for sname in remaining_scenes])   
remaining_scenes = [scenes + '.tfrecord' for scenes in remaining_scenes]

save_pth = str(root_path / 'ImageSets' / 'custom_train_190.txt')
with open(save_pth,'w') as f:
    f.write('\n'.join(selected_scenes))

print(f'Custom_train_scenes saved at {save_pth}\n{num_scenes} scenes, {num_sel_frames} frames')

save_pth = str(root_path / 'ImageSets' / 'custom_train_660.txt')
with open(save_pth,'w') as f:
    f.write('\n'.join(remaining_scenes))

print(f'Custom_train_scenes saved at {save_pth}\n{len(seq_nframes)-num_scenes} scenes, {num_rem_frames} frames')    
