# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations

Usage example: python ./batch_load_scannet_data.py
"""
import os
import sys
import datetime
import numpy as np
from load_scannet_data import export
import pdb

SCANNET_DIR = '/data2/szh/scannet/scans'
TRAIN_SCAN_NAMES = [line.rstrip() for line in open('meta_data/scannet_train.txt')]
LABEL_MAP_FILE = 'meta_data/scannetv2-labels.combined.tsv'
DONOTCARE_CLASS_IDS = np.array([])
# OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
OBJ_CLASS_IDS = np.array(
            ['bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'chair', 'cup', 'curtain', 'desk', 'door', 'dresser',
             'keyboard', 'lamp', 'laptop', 'monitor',
             'night_stand', 'plant', 'sofa', 'stool', 'table', 'toilet', 'wardrobe'])
MAX_NUM_POINT = 50000
OUTPUT_FOLDER = '/data2/szh/scannet/scannet_train_detection_data_22'

def export_one_scan(scan_name, output_filename_prefix):
    mesh_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.aggregation.json')
    seg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans.
    mesh_vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic = \
        export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE, None)

    mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
    mesh_vertices = mesh_vertices[mask,:]
    semantic_labels = semantic_labels[mask]
    instance_labels = instance_labels[mask]

    # num_cup_label = 0
    # cup_choide = []
    # for i in range(semantic_labels.shape[0]):
    #     if semantic_labels[i] == 'curtain':
    #         num_cup_label = num_cup_label + 1
    #         cup_choide.append(i)
    # cup = semantic_labels[cup_choide]

    num_instances = len(np.unique(instance_labels))
    print('Num of instances: ', num_instances)

    bbox_mask = np.in1d(instance_bboxes[:,-1], OBJ_CLASS_IDS)
    instance_bboxes = instance_bboxes[bbox_mask,:]
    print('Num of care instances: ', instance_bboxes.shape[0])

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
    # NUM_POINT = 50000 - cup_num - bottle_num - keyboard_num - laptop_num - wardrobe_num
    N = mesh_vertices.shape[0]
    if N > MAX_NUM_POINT:
        # choices = np.random.choice(N, MAX_NUM_POINT, replace=False)

        # choices = np.random.choice(N, NUM_POINT, replace=False)
        # if bottle_num != 0:
        #     # choices = np.concatenate((choices, np.array(bottle_choice)), axis=0)
        #     choices = np.append(choices, bottle_choice)
        # if cup_num != 0:
        #     # choices = np.concatenate((choices, np.array(cup_choice)), axis=0)
        #     choices = np.append(choices, cup_choice)
        # if keyboard_num != 0:
        #     choices = np.append(choices, keyboard_choice)
        # if laptop_num != 0:
        #     choices = np.append(choices, laptop_choice)
        # if wardrobe_num != 0:
        #     choices = np.append(choices, wardrobe_choice)
        choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
        mesh_vertices = mesh_vertices[choices, :]
        semantic_labels = semantic_labels[choices]
        instance_labels = instance_labels[choices]

    np.save(output_filename_prefix+'_vert.npy', mesh_vertices)
    np.save(output_filename_prefix+'_sem_label.npy', semantic_labels)
    np.save(output_filename_prefix+'_ins_label.npy', instance_labels)
    np.save(output_filename_prefix+'_bbox.npy', instance_bboxes)

def batch_export():
    if not os.path.exists(OUTPUT_FOLDER):
        print('Creating new data folder: {}'.format(OUTPUT_FOLDER))
        os.mkdir(OUTPUT_FOLDER)

    for scan_name in TRAIN_SCAN_NAMES:
        print('-'*20+'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name)
        if os.path.isfile(output_filename_prefix+'_vert.npy'):
            print('File already exists. skipping.')
            print('-'*20+'done')
            continue
        try:
            export_one_scan(scan_name, output_filename_prefix)
        except:
            print('Failed export scan: %s'%(scan_name))
        print('-'*20+'done')
    # scan_name = 'scene0000_00'
    # output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name)
    # if os.path.isfile(output_filename_prefix + '_vert.npy'):
    #     print('File already exists. skipping.')
    #     print('-' * 20 + 'done')
    # try:
    #     export_one_scan(scan_name, output_filename_prefix)
    #     print('suceess export scan')
    # except:
    #     print('Failed export scan: %s' % (scan_name))
    # print('-' * 20 + 'done')

if __name__=='__main__':
    batch_export()
