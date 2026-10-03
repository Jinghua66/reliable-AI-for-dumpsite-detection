# Detector configuration. Data paths and initialization are set by train.py.
auto_scale_lr = {'base_batch_size': 8, 'enable': False}

backend_args = None

custom_hooks = [{'enable': True,
  'out_dir': 'tools/temp/feats_and_pkl/fcos_rpn_rcnn_x101_feats',
  'type': 'SetCurrentImageHook'}]

custom_imports = {'allow_failed_imports': False,
 'imports': ['mmdet_custom.heads.convfc_bbox_head_with_probs',
             'mmdet_custom.hooks.set_current_image_hook',
             'my_voc_dataset']}

data_root = 'data/VOCdevkit/'

dataset_type = 'VOCDataset'

default_hooks = {'checkpoint': {'interval': 1,
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

launcher = 'none'

load_from = None

log_level = 'INFO'

log_processor = {'by_epoch': True, 'type': 'LogProcessor', 'window_size': 50}

model = {'backbone': {'base_width': 4,
              'depth': 101,
              'frozen_stages': 1,
              'groups': 64,
              'init_cfg': {'checkpoint': 'open-mmlab://resnext101_64x4d', 'type': 'Pretrained'},
              'norm_cfg': {'requires_grad': True, 'type': 'BN'},
              'norm_eval': True,
              'num_stages': 4,
              'out_indices': (0, 1, 2, 3),
              'style': 'pytorch',
              'type': 'ResNeXt'},
 'data_preprocessor': {'bgr_to_rgb': True,
                       'mean': [93.435, 102.83, 100.309],
                       'pad_size_divisor': 32,
                       'std': [32.134, 34.352, 40.614],
                       'type': 'DetDataPreprocessor'},
 'neck': {'add_extra_convs': 'on_output',
          'in_channels': [256, 512, 1024, 2048],
          'num_outs': 5,
          'out_channels': 256,
          'relu_before_extra_convs': True,
          'start_level': 1,
          'type': 'FPN'},
 'roi_head': {'bbox_head': {'bbox_coder': {'target_means': [0.0, 0.0, 0.0, 0.0],
                                           'target_stds': [0.1, 0.1, 0.2, 0.2],
                                           'type': 'DeltaXYWHBBoxCoder'},
                            'fc_out_channels': 1024,
                            'in_channels': 256,
                            'loss_bbox': {'loss_weight': 1.0, 'type': 'L1Loss'},
                            'loss_cls': {'loss_weight': 1.0,
                                         'type': 'CrossEntropyLoss',
                                         'use_sigmoid': False},
                            'num_classes': 3,
                            'reg_class_agnostic': False,
                            'roi_feat_size': 7,
                            'type': 'ProbShared2FCBBoxHead'},
              'bbox_roi_extractor': {'featmap_strides': [8, 16, 32, 64, 128],
                                     'out_channels': 256,
                                     'roi_layer': {'output_size': 7,
                                                   'sampling_ratio': 0,
                                                   'type': 'RoIAlign'},
                                     'type': 'SingleRoIExtractor'},
              'type': 'StandardRoIHead'},
 'rpn_head': {'feat_channels': 256,
              'in_channels': 256,
              'loss_bbox': {'loss_weight': 1.0, 'type': 'IoULoss'},
              'loss_centerness': {'loss_weight': 1.0,
                                  'type': 'CrossEntropyLoss',
                                  'use_sigmoid': True},
              'loss_cls': {'alpha': 0.25,
                           'gamma': 2.0,
                           'loss_weight': 1.0,
                           'type': 'FocalLoss',
                           'use_sigmoid': True},
              'num_classes': 1,
              'stacked_convs': 4,
              'strides': [8, 16, 32, 64, 128],
              'type': 'FCOSHead'},
 'test_cfg': {'rcnn': {'max_per_img': 5,
                       'nms': {'iou_threshold': 0.3, 'type': 'nms'},
                       'score_thr': 0.1},
              'rpn': {'max_per_img': 1000,
                      'min_bbox_size': 0,
                      'nms': {'iou_threshold': 0.7, 'type': 'nms'},
                      'nms_pre': 1000}},
 'train_cfg': {'rcnn': {'assigner': {'ignore_iof_thr': -1,
                                     'match_low_quality': False,
                                     'min_pos_iou': 0.5,
                                     'neg_iou_thr': 0.5,
                                     'pos_iou_thr': 0.5,
                                     'type': 'MaxIoUAssigner'},
                        'debug': False,
                        'pos_weight': -1,
                        'sampler': {'add_gt_as_proposals': True,
                                    'neg_pos_ub': -1,
                                    'num': 512,
                                    'pos_fraction': 0.25,
                                    'type': 'RandomSampler'}},
               'rpn': {'allowed_border': -1,
                       'assigner': {'ignore_iof_thr': -1,
                                    'match_low_quality': True,
                                    'min_pos_iou': 0.3,
                                    'neg_iou_thr': 0.3,
                                    'pos_iou_thr': 0.7,
                                    'type': 'MaxIoUAssigner'},
                       'debug': False,
                       'pos_weight': -1,
                       'sampler': {'add_gt_as_proposals': False,
                                   'neg_pos_ub': -1,
                                   'num': 256,
                                   'pos_fraction': 0.5,
                                   'type': 'InstanceBalancedPosSampler'}},
               'rpn_proposal': {'max_per_img': 1000,
                                'min_bbox_size': 0,
                                'nms': {'iou_threshold': 0.7, 'type': 'nms'},
                                'nms_pre': 2000}},
 'type': 'FasterRCNN'}

optim_wrapper = {'optimizer': {'lr': 0.01, 'momentum': 0.9, 'type': 'SGD', 'weight_decay': 0.0001},
 'type': 'OptimWrapper'}

param_scheduler = [{'begin': 0, 'by_epoch': False, 'end': 1000, 'start_factor': 0.001, 'type': 'LinearLR'},
 {'begin': 0, 'by_epoch': True, 'end': 20, 'gamma': 0.1, 'milestones': [16], 'type': 'MultiStepLR'}]

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

test_evaluator = {'iou_thrs': 0.5, 'metric': 'mAP', 'type': 'VOCMetric'}

test_pipeline = [{'backend_args': None, 'type': 'LoadImageFromFile'},
 {'keep_ratio': True, 'scale': (1024, 1024), 'type': 'Resize'},
 {'type': 'LoadAnnotations', 'with_bbox': True},
 {'meta_keys': ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
  'type': 'PackDetInputs'}]

train_cfg = {'max_epochs': 30, 'type': 'EpochBasedTrainLoop', 'val_interval': 1}

train_dataloader = {'batch_sampler': {'type': 'AspectRatioBatchSampler'},
 'batch_size': 4,
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
             'times': 5,
             'type': 'RepeatDataset'},
 'num_workers': 2,
 'persistent_workers': True,
 'sampler': {'shuffle': True, 'type': 'DefaultSampler'}}

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

val_evaluator = {'iou_thrs': 0.5, 'metric': 'mAP', 'type': 'VOCMetric'}

vis_backends = [{'type': 'LocalVisBackend'}]

visualizer = {'name': 'visualizer', 'type': 'DetLocalVisualizer', 'vis_backends': [{'type': 'LocalVisBackend'}]}

work_dir = './rd_generate/fcos_rpn_rcnn_x101_dump/'
