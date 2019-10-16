# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Evaluation routine for 3D object detection with SUN RGB-D and ScanNet.
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from model_util_vote import VoteConfigDistance
from ap_helper import APCalculator, parse_predictions, parse_groundtruths

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use SUN RGB-D V2 box labels.')
parser.add_argument('--split', default='val', help='Dataset split. train or val. [default: val]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')

# network
parser.add_argument('--ckpt', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--num_vote_heading', type=int, default=4, help='num_vote_heading for spatial discrete')
parser.add_argument('--max_r', type=str, default='2.0,6.0', help='max_r')
parser.add_argument('--max_z', type=str, default='1.6', help='max_z')
parser.add_argument('--disable_top_n_votes', action='store_true', help='disable_top_n_votes.')
parser.add_argument('--top_n_votes', type=int, default=3, help='Top n votes')
parser.add_argument('--best_n_votes', type=int, default=1024, help='Best n votes')
parser.add_argument('--sorted_by_prob', action='store_true', help='sorted_by_prob.')
parser.add_argument('--cluster_sampling', default='sorted_fps', help='Sampling strategy for vote clusters. [default: sorted_fps]')
parser.add_argument('--num_target', type=int, default=256, help='Point Number [default: 256]')
parser.add_argument('--cluster_radius', type=float, default=0.3, help='cluster_radius')
parser.add_argument('--cluster_nsample', type=int, default=16, help='cluster_nsample')

# eval
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader. [default: 8]')
parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
parser.add_argument('--use_3d_nms', action='store_true', help='Use 3D NMS instead of 2D NMS.')
parser.add_argument('--use_cls_nms', action='store_true', help='Use per class NMS.')
parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
parser.add_argument('--per_class_proposal', action='store_true', help='Duplicate each proposal num_class times.')
parser.add_argument('--conf_thresh', type=float, default=0.05, help='Filter out predictions with obj prob less than it. [default: 0.05]')
parser.add_argument('--faster_eval', action='store_true', help='Faster evaluation by skippling empty bounding box removal.')
parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')
parser.add_argument('--ap_iou_thresholds', default='0.25,0.5', help='A list of AP IoU thresholds [default: 0.25,0.5]')

# loss
parser.add_argument('--vote_cls_loss_weight', type=float, default=0.8, help='vote_cls_loss_weight')
parser.add_argument('--vote_cls_loss_weight_decay', default=False, action='store_true', help='enable vote_cls_loss_weight_decay')
parser.add_argument('--vote_cls_loss_weight_decay_steps', default=None, type=str, help='vote_cls_loss_weight_decay_steps')
parser.add_argument('--vote_cls_loss_weight_decay_rates', default=None, type=str, help='vote_cls_loss_weight_decay_rates')

# output
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')

# statistics
parser.add_argument('--compute_false_stat', action='store_true', help='compute_false_stat.')
parser.add_argument('--obj_pos_prob', type=float, default=0.5, help='obj_pos_prob')
parser.add_argument('--obj_neg_prob', type=float, default=0.5, help='obj_neg_prob')
FLAGS = parser.parse_args()

if FLAGS.use_cls_nms:
    assert(FLAGS.use_3d_nms)

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DUMP_DIR = FLAGS.dump_dir
CHECKPOINT_PATH = FLAGS.ckpt
assert(CHECKPOINT_PATH is not None)
FLAGS.DUMP_DIR = DUMP_DIR
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]

# vote_cls_loss_weight
VOTE_CLS_LOSS_WEIGHT_BASE = FLAGS.vote_cls_loss_weight
if FLAGS.vote_cls_loss_weight_decay:
    assert FLAGS.vote_cls_loss_weight_decay_steps is not None
    assert FLAGS.vote_cls_loss_weight_decay_rates is not None
    VOTE_CLS_LOSS_WEIGHT_DECAY_STEPS = [int(x) for x in FLAGS.vote_cls_loss_weight_decay_steps.split(',')]
    VOTE_CLS_LOSS_WEIGHT_DECAY_RATES = [float(x) for x in FLAGS.vote_cls_loss_weight_decay_rates.split(',')]

def get_current_vote_cls_loss_weight(epoch):
    weight = VOTE_CLS_LOSS_WEIGHT_BASE
    if FLAGS.vote_cls_loss_weight_decay:
        for i, wt_decay_epoch in enumerate(VOTE_CLS_LOSS_WEIGHT_DECAY_STEPS):
            if epoch >= wt_decay_epoch:
                weight *= VOTE_CLS_LOSS_WEIGHT_DECAY_RATES[i]
    return weight

# Prepare DUMP_DIR
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
DUMP_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    DUMP_FOUT.write(out_str+'\n')
    DUMP_FOUT.flush()
    print(out_str)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Construct vote config
VC = VoteConfigDistance(num_vote_heading=FLAGS.num_vote_heading,
                        max_r=[float(x) for x in FLAGS.max_r.split(',')],
                        max_z=[float(x) for x in FLAGS.max_z.split(',')],
                        top_n_votes=FLAGS.top_n_votes,
                        best_n_votes=FLAGS.best_n_votes)

if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset_multi_distance import SunrgbdDetectionVotesDatasetMultiDistance, MAX_NUM_OBJ
    from model_util_sunrgbd import SunrgbdDatasetConfig
    DATASET_CONFIG = SunrgbdDatasetConfig()
    TEST_DATASET = SunrgbdDetectionVotesDatasetMultiDistance(FLAGS.split, num_points=NUM_POINT,
        augment=False, use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_v1=(not FLAGS.use_sunrgbd_v2),
        vote_config=VC)
elif FLAGS.dataset == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset_multi_distance import ScannetDetectionDatasetMultiDistance, MAX_NUM_OBJ
    from model_util_scannet import ScannetDatasetConfig
    DATASET_CONFIG = ScannetDatasetConfig()
    TEST_DATASET = ScannetDetectionDatasetMultiDistance(FLAGS.split, num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        vote_config=VC)
else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)
print(len(TEST_DATASET))
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
    shuffle=FLAGS.shuffle_dataset, num_workers=FLAGS.num_workers, worker_init_fn=my_worker_init_fn)

# Init the model and optimzier
MODEL = importlib.import_module('votenet_multi_distance') # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1

Detector = MODEL.VoteNetMultiDistance

net = Detector(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_config=VC,
               sampling=FLAGS.cluster_sampling,
               disable_top_n_votes=FLAGS.disable_top_n_votes,
               sorted_by_prob=FLAGS.sorted_by_prob,
               cluster_radius=FLAGS.cluster_radius,
               cluster_nsample=FLAGS.cluster_nsample)
net.to(device)
criterion = MODEL.get_loss

# Load checkpoint if there is any
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    best_mAP = checkpoint['mAP']
    log_string("Loaded checkpoint %s (epoch: %d, best eval mAP@0.5: %f)"%(CHECKPOINT_PATH, epoch, best_mAP))

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box': (not FLAGS.faster_eval), 'use_3d_nms': FLAGS.use_3d_nms, 'nms_iou': FLAGS.nms_iou,
    'use_old_type_nms': FLAGS.use_old_type_nms, 'cls_nms': FLAGS.use_cls_nms, 'per_class_proposal': FLAGS.per_class_proposal,
    'conf_thresh': FLAGS.conf_thresh, 'dataset_config':DATASET_CONFIG}
# ------------------------------------------------------------------------- GLOBAL CONFIG END

def evaluate_one_epoch():
    stat_dict = {}
    if FLAGS.compute_false_stat:
        stat_dict['objectness_total_gt_pos'] = 0.0
        stat_dict['objectness_total_gt_neg'] = 0.0
        stat_dict['objectness_total_det_pos'] = 0.0
        stat_dict['objectness_total_det_neg'] = 0.0
        stat_dict['objectness_total_tp'] = 0.0
        stat_dict['objectness_total_tn'] = 0.0
        stat_dict['objectness_total_fp'] = 0.0
        stat_dict['objectness_total_fn'] = 0.0
        stat_dict['objectness_pos_prec'] = 0.0
        stat_dict['objectness_pos_rec'] = 0.0
        stat_dict['objectness_neg_prec'] = 0.0
        stat_dict['objectness_neg_rec'] = 0.0
        stat_dict['objectness_pos_err_rate'] = 0.0
        stat_dict['objectness_fp_rate'] = 0.0
        stat_dict['objectness_neg_err_rate'] = 0.0
        stat_dict['objectness_fn_rate'] = 0.0
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
        for iou_thresh in AP_IOU_THRESHOLDS]
    vote_cls_loss_weight = get_current_vote_cls_loss_weight(epoch)
    net.eval() # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = net(inputs)

        # add vote_cls_loss_weight to end_points
        end_points['vote_cls_loss_weight'] = vote_cls_loss_weight

        # Compute loss
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)

        if FLAGS.compute_false_stat:
            batch_size = float(end_points['objectness_scores'].size(0))
            objectness_scores = end_points['objectness_scores']
            objectness_prob = F.softmax(objectness_scores, dim=2)[:, :, 1] # (B, num_proposal)
            objectness_det_pos_mask = (objectness_prob > FLAGS.obj_pos_prob).long()
            objectness_det_neg_mask = (objectness_prob < FLAGS.obj_neg_prob).long()
            objectness_gt_pos_mask = end_points['objectness_label']
            objectness_gt_neg_mask = end_points['objectness_mask'].long() - end_points['objectness_label']
            cur_total_gt_pos = objectness_gt_pos_mask.sum().float().item()
            cur_total_gt_neg = objectness_gt_neg_mask.sum().float().item()
            cur_total_det_pos = objectness_det_pos_mask.sum().float().item()
            cur_total_det_neg = objectness_det_neg_mask.sum().float().item()
            cur_tp_mask = objectness_det_pos_mask*objectness_gt_pos_mask
            cur_tn_mask = objectness_det_neg_mask*objectness_gt_neg_mask
            cur_fp_mask = objectness_det_pos_mask*objectness_gt_neg_mask
            cur_fn_mask = objectness_det_neg_mask*objectness_gt_pos_mask
            cur_total_tp = cur_tp_mask.sum().float().item()
            cur_total_tn = cur_tn_mask.sum().float().item()
            cur_total_fp = cur_fp_mask.sum().float().item()
            cur_total_fn = cur_fn_mask.sum().float().item()
            stat_dict['objectness_total_gt_pos'] += cur_total_gt_pos / batch_size
            stat_dict['objectness_total_gt_neg'] += cur_total_gt_neg / batch_size
            stat_dict['objectness_total_det_pos'] += cur_total_det_pos / batch_size
            stat_dict['objectness_total_det_neg'] += cur_total_det_neg / batch_size
            stat_dict['objectness_total_tp'] += cur_total_tp / batch_size
            stat_dict['objectness_total_tn'] += cur_total_tn / batch_size
            stat_dict['objectness_total_fp'] += cur_total_fp / batch_size
            stat_dict['objectness_total_fn'] += cur_total_fn / batch_size
            stat_dict['objectness_pos_prec'] += cur_total_tp / cur_total_det_pos
            stat_dict['objectness_pos_rec'] += cur_total_tp / cur_total_gt_pos
            stat_dict['objectness_neg_prec'] += cur_total_tn / cur_total_det_neg
            stat_dict['objectness_neg_rec'] += cur_total_tn / cur_total_gt_neg
            stat_dict['objectness_pos_err_rate'] += cur_total_fp / cur_total_det_pos
            stat_dict['objectness_fp_rate'] += cur_total_fp / cur_total_gt_neg
            stat_dict['objectness_neg_err_rate'] += cur_total_fn / cur_total_det_neg
            stat_dict['objectness_fn_rate'] += cur_total_fn / cur_total_gt_pos

        # Accumulate statistics and print out
        for key in end_points:
            if ('loss' in key or 'acc' in key or 'ratio' in key) and key != 'vote_cls_loss_weight':
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
    
        # Dump evaluation results for visualization
        if batch_idx == 0:
            MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

    # Log statistics
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    # Evaluate average precision
    for i, ap_calculator in enumerate(ap_calculator_list):
        print('-'*10, 'iou_thresh: %f'%(AP_IOU_THRESHOLDS[i]), '-'*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string('eval %s: %f'%(key, metrics_dict[key]))

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss


def eval():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    loss = evaluate_one_epoch()

if __name__=='__main__':
    eval()
