# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import sys
import pickle
import numpy as np
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import random
from model_util_scannet import rotate_aligned_boxes

from model_util_scannet import ScannetDatasetConfig

DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 64
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])


class ScannetDetectionDataset(Dataset):

    def __init__(self, split_set='train', num_points=20000,
                 use_color=False, use_height=False, augment=False,
                 data_root=None):

        if data_root is None:
            # self.data_path = os.path.join(BASE_DIR, '/mnt/sda/szh/scannet/scannet_train_detection_data_22_0928')#small
            # self.data_path = os.path.join(BASE_DIR,'/mnt/sda/szh/scannet/scannet_train_detection_data_22')  #random
            if split_set == 'train':
                # self.data_path = os.path.join(BASE_DIR, '/mnt/sda/szh/scannet/scannet_train_detection_data_22_copy_1005')#copy
                self.data_path = os.path.join(BASE_DIR,
                                              '/data2/scannet/scannet_train_detection_data_22')
            else:
                self.data_path = os.path.join(BASE_DIR, '/data2/szh/scannet/scannet_train_detection_data_22')  # random
        else:
            # self.data_path = os.path.join(data_root, 'scannet_train_detection_data_22')
            # print(self.data_path)
            if split_set == 'train':
                self.data_path = os.path.join(data_root, 'scannet_train_detection_data_22_copy_2')#copy
                print(self.data_path)
            else:
                self.data_path = os.path.join(data_root, 'scannet_train_detection_data_22')  # random
                print(self.data_path)
        # logger.info(f"length of testing dataset: {self.data_path}")
            # self.data_path = '/mnt/sda/szh/scannet/scans'
        all_scan_names = list(set([os.path.basename(x)[0:12] \
                                       for x in os.listdir(self.data_path) if x.startswith('scene')]))
        if split_set == 'all':
            self.scan_names = all_scan_names
                # self.scan_names = ('scene0000_00')
                # print(all_scan_names)
        elif split_set in ['train', 'val', 'test']:
            split_filenames = os.path.join(ROOT_DIR, 'scannet/meta_data',
                                               'scannetv2_{}.txt'.format(split_set))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()
                    # remove unavailiable scans
            ##########resample###########################
            if split_set == 'train':
                resplit_filenames = os.path.join(ROOT_DIR, 'scannet/meta_data',
                                                 'resample_scan.txt')
                with open(resplit_filenames, 'r') as f:
                    rescan_names = f.read().splitlines()
                self.scan_names = rescan_names + self.scan_names + rescan_names
            ########################################################
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                                   if sname in all_scan_names]
            print('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
            num_scans = len(self.scan_names)
        else:
            print('illegal split name')
            return

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment
        # print(len(self.scan_names))
    def __len__(self):
        # return len(self.mesh_vertices_list)
        return len(self.scan_names)
    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_OBJ,)
            size_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            point_obj_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
            point_instance_label: (N,) with int values in -1,...,num_box, indicating which object the point belongs to, -1 means a backgound point.
            scan_idx: int scan index in scan_names list
            pcl_color: unused
        """
        scan_name = self.scan_names[idx]
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name) + '_vert.npy')
        np.load.__defaults__ = (None, True, True, 'ASCII')
        instance_labels = np.load(os.path.join(self.data_path, scan_name) + '_ins_label.npy')
        semantic_labels = np.load(os.path.join(self.data_path, scan_name) + '_sem_label.npy')
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name) + '_bbox.npy')
        np.load.__defaults__ = (None, False, True, 'ASCII')

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

            # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        size_gts = np.zeros((MAX_NUM_OBJ, 3))

        # point_cloud, choices = pc_util.random_sampling(point_cloud,
        #                                                self.num_points, return_choices=True)
        #
        # instance_labels = instance_labels[choices]
        # semantic_labels = semantic_labels[choices]
        #
        # pcl_color = pcl_color[choices]

        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        # cup_num, bottle_num, keyboard_num, laptop_num, wardrobe_num = 0, 0, 0, 0, 0
        # cup_choice, bottle_choice, keyboard_choice, laptop_choice, wardrobe_choice = [], [], [], [], []
        # for i in range(semantic_labels.shape[0]):
        #     if semantic_labels[i] == 'cup':
        #         cup_num = cup_num + 1
        #         cup_choice.append(i)
        #     elif semantic_labels[i] == 'bottle':
        #         bottle_num = bottle_num + 1
        #         bottle_choice.append(i)
        #     elif semantic_labels[i] == 'keyboard':
        #         keyboard_num = keyboard_num + 1
        #         keyboard_choice.append(i)
        #     elif semantic_labels[i] == 'laptop':
        #         laptop_num = laptop_num + 1
        #         laptop_choice.append(i)
        #     elif semantic_labels[i] == 'wardrobe':
        #         wardrobe_num = wardrobe_num + 1
        #         wardrobe_choice.append(i)
        # if wardrobe_num > 10000:
        #     num_points = 50000 - cup_num - bottle_num - keyboard_num - laptop_num - 10000
        # else: num_points = 50000 - cup_num - bottle_num - keyboard_num - laptop_num - wardrobe_num
        #
        # # print(point_cloud.shape[0])
        # if point_cloud.shape[0] < num_points :
        #     choices = np.random.choice(point_cloud.shape[0], num_points, replace=True)
        # elif point_cloud.shape[0] >= num_points :
        #     choices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        #
        # if bottle_num != 0:
        #     choices = np.append(choices, bottle_choice)
        # if cup_num != 0:
        #     choices = np.append(choices, cup_choice)
        # if keyboard_num != 0:
        #     choices = np.append(choices, keyboard_choice)
        # if laptop_num != 0:
        #     choices = np.append(choices, laptop_choice)
        # if wardrobe_num != 0:
        #     if wardrobe_num > 10000:
        #         wardrobe_choice = random.sample(wardrobe_choice, 10000)
        #     choices = np.append(choices, wardrobe_choice)
        # # print(choices.shape[0])
        # # print(num_points, cup_num, bottle_num, keyboard_num, laptop_num, wardrobe_num,choices.shape[0])
        # point_cloud =point_cloud[choices]

        #
        point_cloud, choices = pc_util.random_sampling(point_cloud,
                                                       self.num_points, return_choices=True)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]

        pcl_color = pcl_color[choices]



        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        # target_bboxes[0:instance_bboxes.shape[0], :] = instance_bboxes[0:MAX_NUM_OBJ, 0:6]
        target_bboxes[0:instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

        gt_centers = target_bboxes[:, 0:3]
        gt_centers[instance_bboxes.shape[0]:, :] += 1000.0  # padding centers with a large number
        # compute GT Centers *AFTER* augmentation
        # generate gt centers
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label.
        point_obj_mask = np.zeros(self.num_points)
        point_instance_label = np.zeros(self.num_points) - 1
        for i_instance in np.unique(instance_labels):
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label
            # if semantic_labels[ind[0]] in DC.nyu40ids:
            if semantic_labels[ind[0]] in DC.modelnet40s:
                x = point_cloud[ind, :3]
                center = 0.5 * (x.min(0) + x.max(0))
                ilabel = np.argmin(((center - gt_centers) ** 2).sum(-1))
                point_instance_label[ind] = ilabel
                point_obj_mask[ind] = 1.0
        #########################################
        point_semantic_label = np.zeros(self.num_points)
        for i_semantic in range(semantic_labels.shape[0]):
            if semantic_labels[i_semantic] in DC.modelnet40s:
                for j in range(DC.modelnet40s.shape[0]):
                    if semantic_labels[i_semantic] == DC.modelnet40s[j]:
                        point_semantic_label[i_semantic] = j
                        continue
            else:
                point_semantic_label[i_semantic] = 22
                continue
        ###############################################################
        # class_ind = [np.where(DC.nyu40ids == x)[0][0] for x in instance_bboxes[:, -1]]
        class_ind = [np.where(DC.modelnet40s == x)[0][0] for x in instance_bboxes[:, -1]]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = class_ind
        size_residuals[0:instance_bboxes.shape[0], :] = \
            target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[class_ind, :]
        size_gts[0:instance_bboxes.shape[0], :] = target_bboxes[0:instance_bboxes.shape[0], 3:6]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ######
        ret_dict['sample_choice'] = choices.astype(np.int64)
        ######
        ret_dict['center_label'] = gt_centers.astype(np.float32)
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        ret_dict['size_gts'] = size_gts.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        # target_bboxes_semcls[0:instance_bboxes.shape[0]] = \
        #     [DC.nyu40id2class[x] for x in instance_bboxes[:, -1][0:instance_bboxes.shape[0]]]
        target_bboxes_semcls[0:instance_bboxes.shape[0]] = \
            [DC.modelnet40class[x] for x in instance_bboxes[:, -1][0:instance_bboxes.shape[0]]]
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['point_obj_mask'] = point_obj_mask.astype(np.int64)
        ret_dict['point_instance_label'] = point_instance_label.astype(np.int64)
        ####################
        ret_dict['point_semantic_label'] = point_semantic_label.astype(np.int64)
        ####################
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['pcl_color'] = pcl_color
        # ret_dict['scan_name'] = np.array(self.scan_names)
        return ret_dict