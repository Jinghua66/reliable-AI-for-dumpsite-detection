# Detector configuration. Data paths and initialization are set by train.py.
auto_scale_lr = {'base_batch_size': 8, 'enable': False}

backend_args = None

data_root = 'data/VOCdevkit/'

dataset_type = 'VOCDataset'

default_hooks = {'checkpoint': {'interval': 2,
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

model = {'backbone': {'depth': 50,
              'frozen_stages': 1,
              'init_cfg': {'checkpoint': 'torchvision://resnet50', 'type': 'Pretrained'},
              'norm_cfg': {'requires_grad': False, 'type': 'BN'},
              'norm_eval': True,
              'num_stages': 4,
              'out_indices': (3,),
              'style': 'pytorch',
              'type': 'ResNet'},
 'bbox_head': {'embed_dims': 256,
               'loss_bbox': {'loss_weight': 5.0, 'type': 'L1Loss'},
               'loss_cls': {'alpha': 0.25,
                            'gamma': 2.0,
                            'loss_weight': 1.0,
                            'type': 'FocalLoss',
                            'use_sigmoid': True},
               'loss_iou': {'loss_weight': 2.0, 'type': 'GIoULoss'},
               'num_classes': 3,
               'test_cfg': {'max_per_img': 2},
               'type': 'DABDETRHead'},
 'data_preprocessor': {'bgr_to_rgb': True,
                       'mean': [93.435, 102.83, 100.309],
                       'pad_size_divisor': 1,
                       'std': [32.134, 34.352, 40.614],
                       'type': 'DetDataPreprocessor'},
 'decoder': {'layer_cfg': {'cross_attn_cfg': {'attn_drop': 0.0,
                                              'cross_attn': True,
                                              'embed_dims': 256,
                                              'num_heads': 8,
                                              'proj_drop': 0.0},
                           'ffn_cfg': {'act_cfg': {'type': 'PReLU'},
                                       'embed_dims': 256,
                                       'feedforward_channels': 2048,
                                       'ffn_drop': 0.0,
                                       'num_fcs': 2},
                           'self_attn_cfg': {'attn_drop': 0.0,
                                             'cross_attn': False,
                                             'embed_dims': 256,
                                             'num_heads': 8,
                                             'proj_drop': 0.0}},
             'num_layers': 6,
             'query_dim': 4,
             'query_scale_type': 'cond_elewise',
             'return_intermediate': True,
             'with_modulated_hw_attn': True},
 'encoder': {'layer_cfg': {'ffn_cfg': {'act_cfg': {'type': 'PReLU'},
                                       'embed_dims': 256,
                                       'feedforward_channels': 2048,
                                       'ffn_drop': 0.0,
                                       'num_fcs': 2},
                           'self_attn_cfg': {'batch_first': True,
                                             'dropout': 0.0,
                                             'embed_dims': 256,
                                             'num_heads': 8}},
             'num_layers': 6},
 'neck': {'act_cfg': None,
          'in_channels': [2048],
          'kernel_size': 1,
          'norm_cfg': None,
          'num_outs': 1,
          'out_channels': 256,
          'type': 'ChannelMapper'},
 'num_patterns': 0,
 'num_queries': 300,
 'positional_encoding': {'normalize': True, 'num_feats': 128, 'temperature': 20},
 'test_cfg': {'max_per_img': 2},
 'train_cfg': {'assigner': {'match_costs': [{'eps': 1e-08, 'type': 'FocalLossCost', 'weight': 2.0},
                                            {'box_format': 'xywh',
                                             'type': 'BBoxL1Cost',
                                             'weight': 5.0},
                                            {'iou_mode': 'giou', 'type': 'IoUCost', 'weight': 2.0}],
                            'type': 'HungarianAssigner'}},
 'type': 'DABDETR',
 'with_random_refpoints': False}

optim_wrapper = {'clip_grad': {'max_norm': 0.1, 'norm_type': 2},
 'optimizer': {'lr': 0.0001, 'type': 'AdamW', 'weight_decay': 0.0001},
 'paramwise_cfg': {'custom_keys': {'backbone': {'decay_mult': 1.0, 'lr_mult': 0.1}}},
 'type': 'OptimWrapper'}

param_scheduler = [{'begin': 0, 'by_epoch': True, 'end': 50, 'gamma': 0.1, 'milestones': [40], 'type': 'MultiStepLR'}]

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

train_cfg = {'max_epochs': 50, 'type': 'EpochBasedTrainLoop', 'val_interval': 2}

train_dataloader = {'batch_sampler': {'type': 'AspectRatioBatchSampler'},
 'batch_size': 8,
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

val_evaluator = {'iou_thrs': 0.5, 'metric': 'mAP', 'type': 'VOCMetric'}

vis_backends = [{'type': 'LocalVisBackend'}]

visualizer = {'name': 'visualizer', 'type': 'DetLocalVisualizer', 'vis_backends': [{'type': 'LocalVisBackend'}]}

work_dir = './rd_generate_with_cj/dab_detr_dump/'
