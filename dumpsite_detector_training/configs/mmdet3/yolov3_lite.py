# Detector configuration. Data paths and initialization are set by train.py.
auto_scale_lr = {'base_batch_size': 16, 'enable': False}

backend_args = None

data_preprocessor = {'bgr_to_rgb': True,
 'mean': [93.435, 102.83, 100.309],
 'pad_size_divisor': 32,
 'std': [32.134, 34.352, 40.614],
 'type': 'DetDataPreprocessor'}

data_root = 'data/VOCdevkit/'

dataset_type = 'VOCDataset'

default_hooks = {'checkpoint': {'interval': 3,
                'max_keep_ckpts': 2,
                'rule': 'greater',
                'save_best': 'pascal_voc/mAP',
                'save_last': True,
                'type': 'CheckpointHook'},
 'logger': {'interval': 50, 'type': 'LoggerHook'},
 'param_scheduler': {'type': 'ParamSchedulerHook'},
 'sampler_seed': {'type': 'DistSamplerSeedHook'},
 'timer': {'type': 'IterTimerHook'},
 'visualization': {'draw': False, 'type': 'DetVisualizationHook'}}

default_scope = 'mmdet'

env_cfg = {'cudnn_benchmark': False,
 'dist_cfg': {'backend': 'nccl'},
 'mp_cfg': {'mp_start_method': 'fork', 'opencv_num_threads': 0}}

find_unused_parameters = True

launcher = 'pytorch'

log_level = 'INFO'

log_processor = {'by_epoch': True, 'type': 'LogProcessor', 'window_size': 50}

model = {'backbone': {'act_cfg': {'negative_slope': 0.1, 'type': 'LeakyReLU'},
              'init_cfg': {'checkpoint': 'open-mmlab://mmdet/mobilenet_v2', 'type': 'Pretrained'},
              'out_indices': (2, 4, 6),
              'type': 'MobileNetV2'},
 'bbox_head': {'anchor_generator': {'base_sizes': [[(116, 90), (156, 198), (373, 326)],
                                                   [(30, 61), (62, 45), (59, 119)],
                                                   [(10, 13), (16, 30), (33, 23)]],
                                    'strides': [32, 16, 8],
                                    'type': 'YOLOAnchorGenerator'},
               'bbox_coder': {'type': 'YOLOBBoxCoder'},
               'featmap_strides': [32, 16, 8],
               'in_channels': [96, 96, 96],
               'loss_cls': {'loss_weight': 1.0,
                            'reduction': 'sum',
                            'type': 'CrossEntropyLoss',
                            'use_sigmoid': True},
               'loss_conf': {'loss_weight': 1.0,
                             'reduction': 'sum',
                             'type': 'CrossEntropyLoss',
                             'use_sigmoid': True},
               'loss_wh': {'loss_weight': 2.0, 'reduction': 'sum', 'type': 'MSELoss'},
               'loss_xy': {'loss_weight': 2.0,
                           'reduction': 'sum',
                           'type': 'CrossEntropyLoss',
                           'use_sigmoid': True},
               'num_classes': 3,
               'out_channels': [96, 96, 96],
               'type': 'YOLOV3Head'},
 'data_preprocessor': {'bgr_to_rgb': True,
                       'mean': [93.435, 102.83, 100.309],
                       'pad_size_divisor': 32,
                       'std': [32.134, 34.352, 40.614],
                       'type': 'DetDataPreprocessor'},
 'neck': {'in_channels': [320, 96, 32],
          'num_scales': 3,
          'out_channels': [96, 96, 96],
          'type': 'YOLOV3Neck'},
 'test_cfg': {'conf_thr': 0.1,
              'max_per_img': 5,
              'min_bbox_size': 0,
              'nms': {'iou_threshold': 0.3, 'type': 'nms'},
              'nms_pre': 1000},
 'train_cfg': {'assigner': {'min_pos_iou': 0,
                            'neg_iou_thr': 0.5,
                            'pos_iou_thr': 0.5,
                            'type': 'GridAssigner'}},
 'type': 'YOLOV3'}

optim_wrapper = {'clip_grad': {'max_norm': 35, 'norm_type': 2},
 'optimizer': {'lr': 0.01, 'momentum': 0.9, 'type': 'SGD', 'weight_decay': 0.0005},
 'type': 'OptimWrapper'}

param_scheduler = [{'by_epoch': True, 'gamma': 0.1, 'milestones': [40, 80], 'type': 'MultiStepLR'}]

resume = False

test_cfg = {'type': 'TestLoop'}

test_dataloader = {'batch_size': 1,
 'dataset': {'ann_file': 'VOC2007/ImageSets/Main/test.txt',
             'backend_args': None,
             'data_prefix': {'sub_data_root': 'VOC2007/'},
             'data_root': 'data/VOCdevkit/',
             'metainfo': {'classes': ('domestic garbage',
                                      'construction waste',
                                      'agriculture forestry')},
             'pipeline': [{'backend_args': None, 'type': 'LoadImageFromFile'},
                          {'keep_ratio': True, 'scale': (1024, 1024), 'type': 'Resize'},
                          {'type': 'LoadAnnotations', 'with_bbox': True},
                          {'meta_keys': ('img_id',
                                         'img_path',
                                         'ori_shape',
                                         'img_shape',
                                         'scale_factor'),
                           'type': 'PackDetInputs'}],
             'test_mode': True,
             'type': 'VOCDataset'},
 'drop_last': False,
 'num_workers': 2,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}

test_evaluator = {'metric': 'mAP', 'type': 'VOCMetric'}

test_pipeline = [{'backend_args': None, 'type': 'LoadImageFromFile'},
 {'keep_ratio': True, 'scale': (1024, 1024), 'type': 'Resize'},
 {'type': 'LoadAnnotations', 'with_bbox': True},
 {'meta_keys': ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
  'type': 'PackDetInputs'}]

train_cfg = {'max_epochs': 100, 'type': 'EpochBasedTrainLoop', 'val_interval': 3}

train_dataloader = {'batch_sampler': {'type': 'AspectRatioBatchSampler'},
 'batch_size': 16,
 'dataset': {'dataset': {'datasets': [{'ann_file': 'VOC2007/ImageSets/Main/trainval_RD.txt',
                                       'backend_args': None,
                                       'data_prefix': {'sub_data_root': 'VOC2007/'},
                                       'data_root': 'data/VOCdevkit/',
                                       'filter_cfg': {'bbox_min_size': 32,
                                                      'filter_empty_gt': True,
                                                      'min_size': 32},
                                       'metainfo': {'classes': ('domestic garbage',
                                                                'construction waste',
                                                                'agriculture forestry')},
                                       'pipeline': [{'backend_args': None,
                                                     'type': 'LoadImageFromFile'},
                                                    {'type': 'LoadAnnotations', 'with_bbox': True},
                                                    {'keep_ratio': True,
                                                     'scale': (1024, 1024),
                                                     'type': 'Resize'},
                                                    {'bbox_params': {'filter_lost_elements': True,
                                                                     'format': 'pascal_voc',
                                                                     'label_fields': ['labels'],
                                                                     'type': 'BboxParams'},
                                                     'keymap': {'gt_bboxes': 'bboxes',
                                                                'gt_bboxes_labels': 'labels',
                                                                'img': 'image'},
                                                     'skip_img_without_anno': True,
                                                     'transforms': [{'p': 0.5,
                                                                     'transforms': [{'hue_shift_limit': 5,
                                                                                     'p': 1.0,
                                                                                     'sat_shift_limit': 20,
                                                                                     'type': 'HueSaturationValue',
                                                                                     'val_shift_limit': 15},
                                                                                    {'brightness_limit': 0.1,
                                                                                     'contrast_limit': 0.2,
                                                                                     'p': 1.0,
                                                                                     'type': 'RandomBrightnessContrast'}],
                                                                     'type': 'OneOf'},
                                                                    {'p': 0.3,
                                                                     'transforms': [{'hue_shift_limit': 5,
                                                                                     'p': 0.5,
                                                                                     'sat_shift_limit': 20,
                                                                                     'type': 'HueSaturationValue',
                                                                                     'val_shift_limit': 15},
                                                                                    {'brightness_limit': 0.1,
                                                                                     'contrast_limit': 0.2,
                                                                                     'p': 0.5,
                                                                                     'type': 'RandomBrightnessContrast'}],
                                                                     'type': 'OneOf'}],
                                                     'type': 'Albu'},
                                                    {'prob': 0.5, 'type': 'RandomFlip'},
                                                    {'type': 'PackDetInputs'}],
                                       'type': 'VOCDataset'}],
                         'ignore_keys': ['dataset_type'],
                         'type': 'ConcatDataset'},
             'times': 3,
             'type': 'RepeatDataset'},
 'num_workers': 2,
 'persistent_workers': True,
 'sampler': {'shuffle': True, 'type': 'DefaultSampler'}}

train_pipeline = [{'backend_args': None, 'type': 'LoadImageFromFile'},
 {'type': 'LoadAnnotations', 'with_bbox': True},
 {'prob': 0.5, 'type': 'RandomFlip'},
 {'type': 'PhotoMetricDistortion'},
 {'type': 'PackDetInputs'}]

val_cfg = {'type': 'ValLoop'}

val_dataloader = {'batch_size': 1,
 'dataset': {'ann_file': 'VOC2007/ImageSets/Main/test.txt',
             'backend_args': None,
             'data_prefix': {'sub_data_root': 'VOC2007/'},
             'data_root': 'data/VOCdevkit/',
             'metainfo': {'classes': ('domestic garbage',
                                      'construction waste',
                                      'agriculture forestry')},
             'pipeline': [{'backend_args': None, 'type': 'LoadImageFromFile'},
                          {'keep_ratio': True, 'scale': (1024, 1024), 'type': 'Resize'},
                          {'type': 'LoadAnnotations', 'with_bbox': True},
                          {'meta_keys': ('img_id',
                                         'img_path',
                                         'ori_shape',
                                         'img_shape',
                                         'scale_factor'),
                           'type': 'PackDetInputs'}],
             'test_mode': True,
             'type': 'VOCDataset'},
 'drop_last': False,
 'num_workers': 2,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}

val_evaluator = {'metric': 'mAP', 'type': 'VOCMetric'}

vis_backends = [{'type': 'LocalVisBackend'}]

visualizer = {'name': 'visualizer', 'type': 'DetLocalVisualizer', 'vis_backends': [{'type': 'LocalVisBackend'}]}

work_dir = './rd_generate_with_cj/yolov3_dump/'
