import os
import sys
import time
import numpy as np
from datetime import datetime
import argparse
import torch
from torch.utils.data import DataLoader
import xlwt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

from utils import setup_logger
from models import GroupFreeDetector, get_loss
from models import APCalculator, parse_predictions, parse_groundtruths

sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from eval_det import get_iou_main,get_iou_obb

def parse_option():
    parser = argparse.ArgumentParser()
    # Eval
    parser.add_argument('--checkpoint_path', default=None, required=True, help='Model checkpoint path [default: None]')
    parser.add_argument('--avg_times', default=5, type=int, help='Average times')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')
    parser.add_argument('--dump_dir', default='dump', help='Dump dir to save sample outputs [default: None]')
    parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
    parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
    parser.add_argument('--conf_thresh', type=float, default=0.0,
                        help='Filter out predictions with obj prob less than it. [default: 0.05]')
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5], nargs='+',
                        help='A list of AP IoU thresholds [default: 0.25,0.5]')
    parser.add_argument('--faster_eval', action='store_true',
                        help='Faster evaluation by skippling empty bounding box removal.')
    parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')

    # Model
    parser.add_argument('--width', default=1, type=int, help='backbone width')
    parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
    parser.add_argument('--sampling', default='kps', type=str, help='Query points sampling method (kps, fps)')

    # Transformer
    parser.add_argument('--nhead', default=8, type=int, help='multi-head number')
    parser.add_argument('--num_decoder_layers', default=6, type=int, help='number of decoder layers')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='dim_feedforward')
    parser.add_argument('--transformer_dropout', default=0.1, type=float, help='transformer_dropout')
    parser.add_argument('--transformer_activation', default='relu', type=str, help='transformer_activation')
    parser.add_argument('--self_position_embedding', default='loc_learned', type=str,
                        help='position_embedding in self attention (none, xyz_learned, loc_learned)')
    parser.add_argument('--cross_position_embedding', default='xyz_learned', type=str,
                        help='position embedding in cross attention (none, xyz_learned)')

    # Loss
    parser.add_argument('--query_points_generator_loss_coef', default=0.8, type=float)
    parser.add_argument('--obj_loss_coef', default=0.1, type=float, help='Loss weight for objectness loss')
    parser.add_argument('--box_loss_coef', default=1, type=float, help='Loss weight for box loss')
    parser.add_argument('--sem_cls_loss_coef', default=0.1, type=float, help='Loss weight for classification loss')
    parser.add_argument('--center_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--center_delta', default=1.0, type=float, help='delta for smoothl1 loss in center loss')
    parser.add_argument('--size_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--size_delta', default=1.0, type=float, help='delta for smoothl1 loss in size loss')
    parser.add_argument('--heading_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--heading_delta', default=1.0, type=float, help='delta for smoothl1 loss in heading loss')
    parser.add_argument('--query_points_obj_topk', default=4, type=int, help='query_points_obj_topk')
    parser.add_argument('--size_cls_agnostic', action='store_true', help='Use class-agnostic size prediction.')

    # Data
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
    parser.add_argument('--dataset', default='scannet', help='Dataset name. sunrgbd or scannet. [default: scannet]')
    parser.add_argument('--num_point', type=int, default=50000, help='Point Number [default: 50000]')
    parser.add_argument('--data_root', default='data', help='data root path')
    parser.add_argument('--use_height', action='store_true', help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use SUN RGB-D V2 box labels.')

    args, unparsed = parser.parse_known_args()

    return args


def get_loader(args):
    # Init datasets and dataloaders
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Create Dataset and Dataloader
    if args.dataset == 'sunrgbd':
        from sunrgbd.sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset
        from sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig

        DATASET_CONFIG = SunrgbdDatasetConfig()
        TEST_DATASET = SunrgbdDetectionVotesDataset('val', num_points=args.num_point,
                                                    augment=False,
                                                    use_color=True if args.use_color else False,
                                                    use_height=True if args.use_height else False,
                                                    use_v1=(not args.use_sunrgbd_v2),
                                                    data_root=args.data_root)
    elif args.dataset == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        from scannet.scannet_detection_dataset import ScannetDetectionDataset
        from scannet.model_util_scannet import ScannetDatasetConfig

        DATASET_CONFIG = ScannetDatasetConfig()
        TEST_DATASET = ScannetDetectionDataset('val', num_points=args.num_point,
                                               augment=False,
                                               use_color=True if args.use_color else False,
                                               use_height=True if args.use_height else False,
                                               data_root=args.data_root)
    else:
        raise NotImplementedError(f'Unknown dataset {args.dataset}. Exiting...')

    logger.info(str(len(TEST_DATASET)))

    TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=args.batch_size * torch.cuda.device_count(),
                                 shuffle=args.shuffle_dataset,
                                 num_workers=4,
                                 worker_init_fn=my_worker_init_fn)
    return TEST_DATALOADER, DATASET_CONFIG


def get_model(args, DATASET_CONFIG):
    if args.use_height:
        num_input_channel = int(args.use_color) * 3 + 1
    else:
        num_input_channel = int(args.use_color) * 3

    model = GroupFreeDetector(num_class=DATASET_CONFIG.num_class,
                              num_heading_bin=DATASET_CONFIG.num_heading_bin,
                              num_size_cluster=DATASET_CONFIG.num_size_cluster,
                              mean_size_arr=DATASET_CONFIG.mean_size_arr,
                              input_feature_dim=num_input_channel,
                              width=args.width,
                              num_proposal=args.num_target,
                              sampling=args.sampling,
                              dropout=args.transformer_dropout,
                              activation=args.transformer_activation,
                              nhead=args.nhead,
                              num_decoder_layers=args.num_decoder_layers,
                              dim_feedforward=args.dim_feedforward,
                              self_position_embedding=args.self_position_embedding,
                              cross_position_embedding=args.cross_position_embedding,
                              size_cls_agnostic=True if args.size_cls_agnostic else False)

    criterion = get_loss
    return model, criterion


def load_checkpoint(args, model):
    # Load checkpoint if there is any
    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model']
        save_path = checkpoint.get('save_path', 'none')
        for k in list(state_dict.keys()):
            state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        model.load_state_dict(state_dict)
        logger.info(f"{args.checkpoint_path} loaded successfully!!!")

        del checkpoint
        torch.cuda.empty_cache()
    else:
        raise FileNotFoundError
    return save_path


def evaluate_one_time(test_loader, DATASET_CONFIG, CONFIG_DICT, AP_IOU_THRESHOLDS, model, criterion, args, time=0):
    stat_dict = {}
    if args.num_decoder_layers > 0:
        if args.dataset == 'sunrgbd':
            _prefixes = ['last_', 'proposal_']
            _prefixes += [f'{i}head_' for i in range(args.num_decoder_layers - 1)]
            prefixes = _prefixes.copy() + ['all_layers_']
        elif args.dataset == 'scannet':
            _prefixes = ['last_', 'proposal_']
            _prefixes += [f'{i}head_' for i in range(args.num_decoder_layers - 1)]
            prefixes = _prefixes.copy() + ['last_three_'] + ['all_layers_']
    else:
        prefixes = ['proposal_']  # only proposal
        _prefixes = prefixes

    if args.num_decoder_layers >= 3:
        last_three_prefixes = ['last_', f'{args.num_decoder_layers - 2}head_', f'{args.num_decoder_layers - 3}head_']
    elif args.num_decoder_layers == 2:
        last_three_prefixes = ['last_', '0head_']
    elif args.num_decoder_layers == 1:
        last_three_prefixes = ['last_']
    else:
        last_three_prefixes = []
    points_sa1 = []
    points_sa2 = []
    points_sa3 = []
    points_sa4 = []
    query_points_inds = []
    sample_choice = []
    proposal_pre_sem_cls = []
    gt_sem_cls = []
    gt_instance = []
    head0_pre_sem_cls = []
    head1_pre_sem_cls = []
    head2_pre_sem_cls = []
    head3_pre_sem_cls = []
    head4_pre_sem_cls = []
    last_pre_sem_cls = []


    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
                          for iou_thresh in AP_IOU_THRESHOLDS]
    mAPs = [[iou_thresh, {k: 0 for k in prefixes}] for iou_thresh in AP_IOU_THRESHOLDS]

    model.eval()  # set model to eval mode (for bn and dp)

    batch_pred_map_cls_dict = {k: [] for k in prefixes}
    batch_gt_map_cls_dict = {k: [] for k in prefixes}

    for batch_idx, batch_data_label in enumerate(test_loader):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)
        print(batch_data_label['scan_idx'])
        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = model(inputs)
############################################################################
        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG,
                                     num_decoder_layers=args.num_decoder_layers,
                                     query_points_generator_loss_coef=args.query_points_generator_loss_coef,
                                     obj_loss_coef=args.obj_loss_coef,
                                     box_loss_coef=args.box_loss_coef,
                                     sem_cls_loss_coef=args.sem_cls_loss_coef,
                                     query_points_obj_topk=args.query_points_obj_topk,
                                     center_loss_type=args.center_loss_type,
                                     center_delta=args.center_delta,
                                     size_loss_type=args.size_loss_type,
                                     size_delta=args.size_delta,
                                     heading_loss_type=args.heading_loss_type,
                                     heading_delta=args.heading_delta,
                                     size_cls_agnostic=args.size_cls_agnostic)
#############################################################################################################
        point_sa1 = end_points['sa1_inds'].cpu()
        point_sa2 = point_sa1[:, 0:1024]
        query_points_sample_inds = end_points['query_points_sample_inds'].cpu()   #256
        point_sa3 = point_sa2[:, 0:512]
        point_sa4 = point_sa3[:, 0:256]
        sample_choice_idx = batch_data_label['sample_choice'].cpu()
        gt_sem_cls_idx = batch_data_label['point_semantic_label'].cpu()
        point_instance_label = end_points['point_instance_label'].cpu()  # B, num_points
        for prefix in prefixes:
            if prefix == 'last_three_':
                break
            pred_sem_cls = torch.argmax(end_points[f'{prefix}sem_cls_scores'], -1)  # B,num_proposal
            end_points[f'{prefix}pred_sem_cls'] = pred_sem_cls


        for i in range(point_sa1.shape[0]):
            points_sa1.append(point_sa1[i].numpy())
            points_sa2.append(point_sa2[i].numpy())
            points_sa3.append(point_sa3[i].numpy())
            points_sa4.append(point_sa4[i].numpy())
            sample_choice.append(sample_choice_idx[i].numpy())
            gt_instance.append(point_instance_label[i].numpy())
            gt_sem_cls.append(gt_sem_cls_idx[i].numpy())
            query_points_inds.append(query_points_sample_inds[i].numpy())
            proposal_pre_sem_cls.append(end_points['proposal_pred_sem_cls'].cpu()[i].numpy())
            head0_pre_sem_cls.append(end_points['0head_pred_sem_cls'].cpu()[i].numpy())
            head1_pre_sem_cls.append(end_points['1head_pred_sem_cls'].cpu()[i].numpy())
            head2_pre_sem_cls.append(end_points['2head_pred_sem_cls'].cpu()[i].numpy())
            head3_pre_sem_cls.append(end_points['3head_pred_sem_cls'].cpu()[i].numpy())
            head4_pre_sem_cls.append(end_points['4head_pred_sem_cls'].cpu()[i].numpy())
            last_pre_sem_cls.append(end_points['last_pred_sem_cls'].cpu()[i].numpy())


   #########################################################################################################################
        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                if isinstance(end_points[key], float):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()

        for prefix in prefixes:
            if prefix == 'last_three_':
                end_points[f'{prefix}center'] = torch.cat([end_points[f'{ppx}center']
                                                           for ppx in last_three_prefixes], 1)
                end_points[f'{prefix}heading_scores'] = torch.cat([end_points[f'{ppx}heading_scores']
                                                                   for ppx in last_three_prefixes], 1)
                end_points[f'{prefix}heading_residuals'] = torch.cat([end_points[f'{ppx}heading_residuals']
                                                                      for ppx in last_three_prefixes], 1)
                if args.size_cls_agnostic:
                    end_points[f'{prefix}pred_size'] = torch.cat([end_points[f'{ppx}pred_size']
                                                                  for ppx in last_three_prefixes], 1)
                else:
                    end_points[f'{prefix}size_scores'] = torch.cat([end_points[f'{ppx}size_scores']
                                                                    for ppx in last_three_prefixes], 1)
                    end_points[f'{prefix}size_residuals'] = torch.cat([end_points[f'{ppx}size_residuals']
                                                                       for ppx in last_three_prefixes], 1)
                end_points[f'{prefix}sem_cls_scores'] = torch.cat([end_points[f'{ppx}sem_cls_scores']
                                                                   for ppx in last_three_prefixes], 1)
                end_points[f'{prefix}objectness_scores'] = torch.cat([end_points[f'{ppx}objectness_scores']
                                                                      for ppx in last_three_prefixes], 1)

            elif prefix == 'all_layers_':
                end_points[f'{prefix}center'] = torch.cat([end_points[f'{ppx}center']
                                                           for ppx in _prefixes], 1)
                end_points[f'{prefix}heading_scores'] = torch.cat([end_points[f'{ppx}heading_scores']
                                                                   for ppx in _prefixes], 1)
                end_points[f'{prefix}heading_residuals'] = torch.cat([end_points[f'{ppx}heading_residuals']
                                                                      for ppx in _prefixes], 1)
                if args.size_cls_agnostic:
                    end_points[f'{prefix}pred_size'] = torch.cat([end_points[f'{ppx}pred_size']
                                                                  for ppx in _prefixes], 1)
                else:
                    end_points[f'{prefix}size_scores'] = torch.cat([end_points[f'{ppx}size_scores']
                                                                    for ppx in _prefixes], 1)
                    end_points[f'{prefix}size_residuals'] = torch.cat([end_points[f'{ppx}size_residuals']
                                                                       for ppx in _prefixes], 1)
                end_points[f'{prefix}sem_cls_scores'] = torch.cat([end_points[f'{ppx}sem_cls_scores']
                                                                   for ppx in _prefixes], 1)
                end_points[f'{prefix}objectness_scores'] = torch.cat([end_points[f'{ppx}objectness_scores']
                                                                      for ppx in _prefixes], 1)

            batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT, prefix,
                                                   size_cls_agnostic=args.size_cls_agnostic)                 #统计时修改该函数，将移除空白box、NMS等注掉
            batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT,
                                                  size_cls_agnostic=args.size_cls_agnostic)
            # batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            # batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)
            for i in range(len(batch_pred_map_cls)):
                batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls[i])
            for i in range(len(batch_gt_map_cls)):
                batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls[i])




        # if (batch_idx + 1) % 10 == 0:
        #     logger.info(f'T[{time}] Eval: [{batch_idx + 1}/{len(test_loader)}]  ' + ''.join(
        #         [f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
        #          for key in sorted(stat_dict.keys()) if 'loss' not in key]))
        #     logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
        #                          for key in sorted(stat_dict.keys()) if
        #                          'loss' in key and 'proposal_' not in key and 'last_' not in key and 'head_' not in key]))
        #     logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
        #                          for key in sorted(stat_dict.keys()) if 'last_' in key]))
        #     logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
        #                          for key in sorted(stat_dict.keys()) if 'proposal_' in key]))
        #     for ihead in range(args.num_decoder_layers - 2, -1, -1):
        #         logger.info(''.join([f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
        #                              for key in sorted(stat_dict.keys()) if f'{ihead}head_' in key]))
    #############################################################################################################################
######################################################################################################################################

    pl_num = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_data1 = pl_num.add_sheet('proposal', cell_overwrite_ok=True)
    sheet_data2 = pl_num.add_sheet('head0', cell_overwrite_ok=True)
    sheet_data3 = pl_num.add_sheet('head1', cell_overwrite_ok=True)
    sheet_data4 = pl_num.add_sheet('head2', cell_overwrite_ok=True)
    sheet_data5 = pl_num.add_sheet('head3', cell_overwrite_ok=True)
    sheet_data6 = pl_num.add_sheet('head4', cell_overwrite_ok=True)
    sheet_data7 = pl_num.add_sheet('last', cell_overwrite_ok=True)
#     sheet_sa1 = pl_num.add_sheet('采样点云数量_sa1', cell_overwrite_ok=True)
#     sheet_sa2 = pl_num.add_sheet('采样点云数量_sa2', cell_overwrite_ok=True)
#     sheet_sa3 = pl_num.add_sheet('采样点云数量_sa3', cell_overwrite_ok=True)
#     sheet_sa4 = pl_num.add_sheet('采样点云数量_sa4', cell_overwrite_ok=True)
#     sheet1_sa1 = pl_num.add_sheet('object_sa1', cell_overwrite_ok=True)
#     sheet1_sa2 = pl_num.add_sheet('object_sa2', cell_overwrite_ok=True)
#     sheet1_sa3 = pl_num.add_sheet('object_sa3', cell_overwrite_ok=True)
#     sheet1_sa4 = pl_num.add_sheet('object_sa4', cell_overwrite_ok=True)
    col = ('class', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'chair', 'cup', 'curtain', 'desk',
           'door', 'dresser', 'keyboard', 'lamp', 'laptop', 'monitor','night_stand', 'plant', 'sofa', 'stool', 'table',
           'toilet', 'wardrobe', 'background')
    for i in range(24):
        sheet_data1.write(0, i, col[i])
        sheet_data2.write(0, i, col[i])
        # sheet_data3.write(0, i, col[i])
        # sheet_data4.write(0, i, col[i])
        # sheet_data5.write(0, i, col[i])
        # sheet_data6.write(0, i, col[i])
        # sheet_data7.write(0, i, col[i])

        sheet_data1.write(i, 0, col[i])
        sheet_data2.write(i, 0, col[i])
        # sheet_data3.write(i, 0, col[i])
        # sheet_data4.write(i, 0, col[i])
        # sheet_data5.write(i, 0, col[i])
        # sheet_data6.write(i, 0, col[i])
        # sheet_data7.write(i, 0, col[i])
#     pc_class = np.array(
#         ['bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'chair', 'cup', 'curtain', 'desk', 'door', 'dresser',
#          'keyboard', 'lamp', 'laptop', 'monitor',
#          'night_stand', 'plant', 'sofa', 'stool', 'table', 'toilet', 'wardrobe'])
#     for i in range(0, 23):
#         sheet_sa1.write(0, i, col[i])
#         sheet_sa2.write(0, i, col[i])
#         sheet_sa3.write(0, i, col[i])
#         sheet_sa4.write(0, i, col[i])
#         sheet1_sa1.write(0, i, col[i])
#         sheet1_sa2.write(0, i, col[i])
#         sheet1_sa3.write(0, i, col[i])
#         sheet1_sa4.write(0, i, col[i])
    data_path = '/data2/szh/scannet/scannet_train_detection_data_22_a3'
    all_scan_names = list(set([os.path.basename(x)[0:12] \
                               for x in os.listdir(data_path) if x.startswith('scene')]))
    split_set = 'val'
    if split_set == 'all':
        scan_names = all_scan_names
        # self.scan_names = ('scene0000_00')
        # print(all_scan_names)
    elif split_set in ['train', 'val', 'test']:
        split_filenames = os.path.join('/home2/szh/paper/Group-Free-3D', 'scannet/meta_data',
                                       'scannetv2_{}.txt'.format(split_set))
        with open(split_filenames, 'r') as f:
            scan_names = f.read().splitlines()
            # remove unavailiable scans
        #########################resample########################
        # resplit_filenames = os.path.join(ROOT_DIR, 'scannet/meta_data',
        #                                  'resample_scan.txt')
        # with open(resplit_filenames, 'r') as f:
        #     rescan_names = f.read().splitlines()
        # scan_names = rescan_names + scan_names + rescan_names
        #################################################################
        num_scans = len(scan_names)
        scan_names = [sname for sname in scan_names \
                      if sname in all_scan_names]
        print('kept {} scans out of {}'.format(len(scan_names), num_scans))
#     data_num_sa1 = np.zeros((len(scan_names), 22))
#     data_num_sa2 = np.zeros((len(scan_names), 22))
#     data_num_sa3 = np.zeros((len(scan_names), 22))
#     data_num_sa4 = np.zeros((len(scan_names), 22))
#
#     object_num_sa1 = np.zeros((len(scan_names), 22))
#     object_num_sa2 = np.zeros((len(scan_names), 22))
#     object_num_sa3 = np.zeros((len(scan_names), 22))
#     object_num_sa4 = np.zeros((len(scan_names), 22))
#     object_num = np.zeros(22)
#
    data_proposal = np.zeros((23, 23))
    data_head0 = np.zeros((23, 23))
    data_head1 = np.zeros((23, 23))
    data_head2 = np.zeros((23, 23))
    data_head3 = np.zeros((23, 23))
    data_head4 = np.zeros((23, 23))
    data_last = np.zeros((23, 23))



    for idx in range(len(scan_names)):
        bottle_num, cup_num, keyboard_num, laptop_num = 0, 0, 0, 0
        scan_name = scan_names[idx]
        # mesh_vertices = np.load(os.path.join(data_path, scan_name) + '_vert.npy')
        np.load.__defaults__ = (None, True, True, 'ASCII')
        # instance_labels = np.load(os.path.join(data_path, scan_name) + '_ins_label.npy')
        semantic_labels = np.load(os.path.join(data_path, scan_name) + '_sem_label.npy')
        instance_bboxes = np.load(os.path.join(data_path, scan_name) + '_bbox.npy')
        np.load.__defaults__ = (None, False, True, 'ASCII')
        # print(semantic_labels.shape[0])
 ##############################propoal统计############################################
        # semantic_labels = semantic_labels[sample_choice[idx]]
        print(idx)
        sence_gt_sem_cls = gt_sem_cls[idx]

        gt_map = batch_gt_map_cls_dict['0head_'][idx]
        pre_map = batch_pred_map_cls_dict['0head_'][idx]
        query_points_id = points_sa2[idx][query_points_inds][idx]
        query_points_gt_sem_cls = sence_gt_sem_cls[query_points_id]
        sence_proposal_pre_sem_cls = proposal_pre_sem_cls[idx]
        sence_head0_pre_sem_cls = head0_pre_sem_cls[idx]
        point_instance = gt_instance[idx]
        # sence_head1_pre_sem_cls = head1_pre_sem_cls[idx]
        # sence_head2_pre_sem_cls = head2_pre_sem_cls[idx]
        # sence_head3_pre_sem_cls = head3_pre_sem_cls[idx]
        # sence_head4_pre_sem_cls = head4_pre_sem_cls[idx]
        # sence_last_pre_sem_cls = last_pre_sem_cls[idx]
        cls = np.arange(0,22,1)
        for i in range(query_points_gt_sem_cls.shape[0]):
            q = query_points_gt_sem_cls[i]

            # if sence_proposal_pre_sem_cls[i] in cls:
            #     k1 = sence_proposal_pre_sem_cls[i]
            # else:
            #     k1 = 22
            # data_proposal[q][k1] = data_proposal[q][k1]+1

            if sence_head0_pre_sem_cls[i] in cls:
                k2 = sence_head0_pre_sem_cls[i]
            else:
                k2 = 22
            data_proposal[q][k2] = data_proposal[q][k2] + 1
            ovmax = -np.inf
            for j in range(len(gt_map)):
                iou = get_iou_main(get_iou_obb,(pre_map[i][1],gt_map[j][1]))
                if iou > ovmax:
                    ovmax = iou
                    box_lable = gt_map[j][0]
            if ovmax > 0.25 and box_lable == k2:
                data_head0[q][k2] = data_head0[q][k2] + 1

            # if sence_head1_pre_sem_cls[i] in cls:
            #     k3 = sence_head1_pre_sem_cls[i]
            # else:
            #     k3 = 22
            # data_head1[q][k3] = data_head1[q][k3] + 1
            #
            # if sence_head2_pre_sem_cls[i] in cls:
            #     k4 = sence_head2_pre_sem_cls[i]
            # else:
            #     k4 = 22
            # data_head2[q][k4] = data_head2[q][k4] + 1
            #
            # if sence_head3_pre_sem_cls[i] in cls:
            #     k5 = sence_head3_pre_sem_cls[i]
            # else:
            #     k5 = 22
            # data_head3[q][k5] = data_head3[q][k5] + 1
            #
            # if sence_head4_pre_sem_cls[i] in cls:
            #     k6 = sence_head4_pre_sem_cls[i]
            # else:
            #     k6 = 22
            # data_head4[q][k6] = data_head4[q][k6] + 1
            #
            # if sence_last_pre_sem_cls[i] in cls:
            #     k7 = sence_last_pre_sem_cls[i]
            # else:
            #     k7 = 22
            # data_last[q][k7] = data_head4[q][k7] + 1



    # #####################点云类型统计##########################################################
#         semantic_labels = semantic_labels[sample_choice[idx]]
#         instance_labels = instance_labels[sample_choice[idx]]
#
#         semantic_labels_sa1 = semantic_labels[points_sa1[idx]]
#         semantic_labels_sa2 = semantic_labels[points_sa2[idx]]
#         semantic_labels_sa3 = semantic_labels[points_sa3[idx]]
#         semantic_labels_sa4 = semantic_labels[points_sa4[idx]]
#
#         object_choice_sa1 = []
#         object_choice_sa2 = []
#         object_choice_sa3 = []
#         object_choice_sa4 = []
#         for i in range(22):
#             object_choice_sa1.append([])
#             object_choice_sa2.append([])
#             object_choice_sa3.append([])
#             object_choice_sa4.append([])
#
#
#         for i in range(semantic_labels_sa1.shape[0]):
#             for j in range(22):
#                 if semantic_labels_sa1[i] == pc_class[j]:
#                     data_num_sa1[idx][j] = data_num_sa1[idx][j] + 1
#                     object_choice_sa1[j].append(i)
#
#         for i in range(semantic_labels_sa2.shape[0]):
#             for j in range(22):
#                 if semantic_labels_sa2[i] == pc_class[j]:
#                     data_num_sa2[idx][j] = data_num_sa2[idx][j] + 1
#                     object_choice_sa2[j].append(i)
#         for i in range(semantic_labels_sa3.shape[0]):
#             for j in range(22):
#                 if semantic_labels_sa3[i] == pc_class[j]:
#                     data_num_sa3[idx][j] = data_num_sa3[idx][j] + 1
#                     object_choice_sa3[j].append(i)
#         for i in range(semantic_labels_sa4.shape[0]):
#             for j in range(22):
#                 if semantic_labels_sa4[i] == pc_class[j]:
#                     data_num_sa4[idx][j] = data_num_sa4[idx][j] + 1
#                     object_choice_sa4[j].append(i)
#
#         for i in range(22):
#             num_instance1 = len(np.unique(instance_labels[object_choice_sa1[i]]))
#             # instance_labels_sa1 = instance_labels[object_choice_sa1[i]]
#             num_instance2 = len(np.unique(instance_labels[object_choice_sa2[i]]))
#             num_instance3 = len(np.unique(instance_labels[object_choice_sa3[i]]))
#             num_instance4 = len(np.unique(instance_labels[object_choice_sa4[i]]))
#             object_num_sa1[idx][i] = num_instance1
#             object_num_sa2[idx][i] = num_instance2
#             object_num_sa3[idx][i] = num_instance3
#             object_num_sa4[idx][i] = num_instance4
#
#
# ######################################################################################
#         # for i in range(instance_bboxes.shape[0]):
#         #     for j in range(22):
#         #         if instance_bboxes[i][6] == pc_class[j]:
#         #             object_num[j] = object_num[j] + 1
#         print(idx)
    for i in range(23):
        for j in range(23):
            sheet_data1.write(i + 1, j + 1, data_proposal[i][j])
            sheet_data2.write(i + 1, j + 1, data_head0[i][j])
            # sheet_data3.write(i + 1, j + 1, data_head1[i][j])
            # sheet_data4.write(i + 1, j + 1, data_head2[i][j])
            # sheet_data5.write(i + 1, j + 1, data_head3[i][j])
            # sheet_data6.write(i + 1, j + 1, data_head4[i][j])
            # sheet_data7.write(i + 1, j + 1, data_last[i][j])
    save_path = '/data2/szh/data_proposal_1025_a3_1.xls'
    pl_num.save(save_path)
    print('save success!')
    # for i in range(0,len(scan_names)):
    #     name = scan_names[i]
    #     sheet_sa1.write(i+1,0,name)
    #     sheet_sa2.write(i + 1, 0, name)
    #     sheet_sa3.write(i + 1, 0, name)
    #     sheet_sa4.write(i + 1, 0, name)
    #     sheet1_sa1.write(i + 1, 0, name)
    #     sheet1_sa2.write(i + 1, 0, name)
    #     sheet1_sa3.write(i + 1, 0, name)
    #     sheet1_sa4.write(i + 1, 0, name)
    #     for j in range(0,22):
    #         num_sa1 = data_num_sa1[i][j]
    #         num_sa2 = data_num_sa2[i][j]
    #         num_sa3 = data_num_sa3[i][j]
    #         num_sa4 = data_num_sa4[i][j]
    #         if num_sa2 > num_sa1 :
    #             print(scan_names[i])
    #         num1_sa1 = object_num_sa1[i][j]
    #         num1_sa2 = object_num_sa2[i][j]
    #         num1_sa3 = object_num_sa3[i][j]
    #         num1_sa4 = object_num_sa4[i][j]
    #         # print(num)
    #         sheet_sa1.write(i + 1, j + 1, num_sa1)
    #         sheet_sa2.write(i + 1, j + 1, num_sa2)
    #         sheet_sa3.write(i + 1, j + 1, num_sa3)
    #         sheet_sa4.write(i + 1, j + 1, num_sa4)
    #
    #         sheet1_sa1.write(i + 1, j + 1, num1_sa1)
    #         sheet1_sa2.write(i + 1, j + 1, num1_sa2)
    #         sheet1_sa3.write(i + 1, j + 1, num1_sa3)
    #         sheet1_sa4.write(i + 1, j + 1, num1_sa4)

    # save_path = '/mnt/sda/szh/data+object_num_train_r_sample.xls'
    # pl_num.save(save_path)
    # print('save success!')
##################################################################################      d           d
    # for prefix in prefixes:
    #     for (batch_pred_map_cls, batch_gt_map_cls) in zip(batch_pred_map_cls_dict[prefix],
    #                                                       batch_gt_map_cls_dict[prefix]):
    #         for ap_calculator in ap_calculator_list:
    #             ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
    #     # Evaluate average precision
    #     for i, ap_calculator in enumerate(ap_calculator_list):
    #         metrics_dict = ap_calculator.compute_metrics()
    #         logger.info(f'===================>T{time} {prefix} IOU THRESH: {AP_IOU_THRESHOLDS[i]}<==================')
    #         for key in metrics_dict:
    #             logger.info(f'{key} {metrics_dict[key]}')
    #
    #         mAPs[i][1][prefix] = metrics_dict['mAP']
    #         ap_calculator.reset()
    # for mAP in mAPs:
    #     logger.info(f'T[{time}] IoU[{mAP[0]}]: ' +
    #                 ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))

    return mAPs


def eval(args, avg_times=5):
    test_loader, DATASET_CONFIG = get_loader(args)
    n_data = len(test_loader.dataset)
    logger.info(f"length of testing dataset: {n_data}")
    model, criterion = get_model(args, DATASET_CONFIG)
    logger.info(str(model))
    save_path = load_checkpoint(args, model)
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        logger.info("Let's use %d GPUs!" % (torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # Used for AP calculation
    CONFIG_DICT = {'remove_empty_box': (not args.faster_eval), 'use_3d_nms': True, 'nms_iou': args.nms_iou,
                   'use_old_type_nms': args.use_old_type_nms, 'cls_nms': True,
                   'per_class_proposal': True,
                   'conf_thresh': args.conf_thresh, 'dataset_config': DATASET_CONFIG}

    logger.info(str(datetime.now()))
    mAPs_times = [None for i in range(avg_times)]
    for i in range(avg_times):
        np.random.seed(i + args.rng_seed)
        mAPs = evaluate_one_time(test_loader, DATASET_CONFIG, CONFIG_DICT, args.ap_iou_thresholds,
                                 model, criterion, args, i)
        mAPs_times[i] = mAPs
        logger.info(f"checkpoint path {save_path}")

    mAPs_avg = mAPs.copy()

    for i, mAP in enumerate(mAPs_avg):
        for key in mAP[1].keys():
            avg = 0
            for t in range(avg_times):
                cur = mAPs_times[t][i][1][key]
                avg += cur
            avg /= avg_times
            mAP[1][key] = avg

    for mAP in mAPs_avg:
        logger.info(f'AVG IoU[{mAP[0]}]: \n' +
                    ''.join([f'{key}: {mAP[1][key]:.4f} \n' for key in sorted(mAP[1].keys())]))

    for mAP in mAPs_avg:
        logger.info(f'AVG IoU[{mAP[0]}]: \t' +
                    ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))

    logger.info(f"checkpoint path {save_path}")


if __name__ == '__main__':
    opt = parse_option()

    opt.dump_dir = os.path.join(opt.dump_dir, f'eval_{opt.dataset}_{int(time.time())}_{np.random.randint(100000000)}')
    logger = setup_logger(output=opt.dump_dir, name="eval")

    eval(opt, opt.avg_times)
