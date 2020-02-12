import os
import inspect
import numpy as np
from collections import defaultdict
from tqdm import tqdm

CUR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
DATA_DIR = os.path.join(CUR_DIR, 'data/Stanford3dDataset_aligned')
EXPORT_DIR = os.path.join(CUR_DIR, 'scans')
META_DIR = os.path.join(CUR_DIR, 'meta_data')

#train_area = [[1, 2, 3, 4, 6], [1, 3, 5, 6], [2, 4, 5]]
#val_area = [[5], [2, 4], [1, 3, 6]]

train_split = {1: [0, 1], 2: [0, 2], 3: [0, 1], 4: [0, 2], 5: [1, 2], 6: [0, 1]}
val_split = {1: 2, 2: 1, 3: 2, 4: 1, 5: 0, 6: 2}

ann_class_to_id = {'table': 0, 'chair': 1, 'sofa': 2, 'bookcase': 3, 'board': 4, 'column': 5, 'window': 6, 'door': 7}

if not os.path.exists(EXPORT_DIR):
    os.mkdir(EXPORT_DIR)

if not os.path.exists(META_DIR):
    os.mkdir(META_DIR)

train_scans = [[], [], []]
val_scans = [[], [], []]
room_count = defaultdict(int)

for x in range(1, 7):
    print('Parsing Area {} ...'.format(x))
    AREA_DIR = os.path.join(DATA_DIR, 'Area_{}'.format(x))
    for room in tqdm(os.listdir(AREA_DIR)):
        if room[0] != '.':
            try:
                room_class = room.split('_')[0]    
                save_name = room_class + '_%06d'%(room_count[room_class])
                if os.path.exists(os.path.join(EXPORT_DIR, save_name+'_pc.npy')):
                    for split in train_split[x]:
                        train_scans[split].append(save_name)
                    val_scans[val_split[x]].append(save_name)
                    room_count[room_class] += 1
                    continue

                # export data from annotation
                ANN_DIR = os.path.join(AREA_DIR, room, 'Annotations')
                pc = []
                bbox = []
                for ins in tqdm(os.listdir(ANN_DIR)):
                    if not ins.endswith('.txt'):
                        continue
                    ins_pc = np.loadtxt(os.path.join(ANN_DIR, ins))
                    pc.append(ins_pc)
                    ins_class = ins.split('_')[0]
                    if ins_class in ann_class_to_id:
                        xmin = np.min(ins_pc[:, 0])
                        ymin = np.min(ins_pc[:, 1])
                        zmin = np.min(ins_pc[:, 2])
                        xmax = np.max(ins_pc[:, 0])
                        ymax = np.max(ins_pc[:, 1])
                        zmax = np.max(ins_pc[:, 2])
                        bbox.append(np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2,
                                              xmax-xmin, ymax-ymin, zmax-zmin, ann_class_to_id[ins_class]]))
                if len(bbox) > 0:
                    pc = np.vstack(pc)
                    bbox = np.vstack(bbox)

                    np.save(os.path.join(EXPORT_DIR, save_name+'_pc.npy'), pc)
                    np.save(os.path.join(EXPORT_DIR, save_name+'_bbox.npy'), bbox)

                    for split in train_split[x]:
                        train_scans[split].append(save_name)
                    val_scans[val_split[x]].append(save_name)
                    room_count[room_class] += 1
            except:
                continue
    print()

for i in range(3):
    with open(os.path.join(META_DIR, 'train{}.txt'.format(i)), 'w') as f:
        f.write('\n'.join(train_scans[i]))
    with open(os.path.join(META_DIR, 'val{}.txt'.format(i)), 'w') as f:
        f.write('\n'.join(val_scans[i]))