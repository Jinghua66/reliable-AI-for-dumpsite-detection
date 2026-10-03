"""Configure and train detectors with MMDetection or Ultralytics.

Use --dry-run to export an effective configuration without starting training.
Use --fresh for backbone initialization or --checkpoint for detector weights.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import pprint
import runpy
from src.voc import CLASSES, read_ids

def dataset_config(node, voc, split):
    if isinstance(node, dict):
        if node.get('type') == 'VOCDataset':
            node['data_root'] = str(voc.parent) + '/'
            node['ann_file'] = str(voc/'ImageSets/Main'/(split+'.txt'))
            node['data_prefix'] = dict(sub_data_root=str(voc)+'/')
            node['metainfo'] = dict(classes=CLASSES)
        for value in node.values(): dataset_config(value,voc,split)
    elif isinstance(node,list):
        for value in node:dataset_config(value,voc,split)

def old_dataset_config(node,voc,split):
    if isinstance(node,dict):
        if node.get('type')=='VOCDataset':
            node['ann_file']=str(voc/'ImageSets/Main'/(split+'.txt'))
            node['img_prefix']=str(voc)+'/'
            node['classes']=CLASSES
        for value in node.values():old_dataset_config(value,voc,split)
    elif isinstance(node,list):
        for value in node:old_dataset_config(value,voc,split)

def dump(config,path):
    Path(path).write_text('\n\n'.join(k+' = '+pprint.pformat(v,width=100,sort_dicts=False) for k,v in config.items())+'\n',encoding='utf-8')

def resolve(a):
    catalog=json.loads(Path('configs/models.json').read_text())
    record=next((x for x in catalog if x['id']==a.model),None)
    if record is None:raise ValueError('Unknown model. See configs/models.json')
    path=Path(record['config']);voc=Path(a.voc).resolve();work=Path(a.work_dir).resolve()
    # Validation must be independent. No implicit use of test.txt for checkpoint selection.
    train_ids=set(read_ids(voc/'ImageSets/Main/train.txt'))
    val_ids=set(read_ids(voc/'ImageSets/Main/val.txt'))
    if not train_ids or not val_ids or train_ids & val_ids:raise ValueError('Empty or overlapping train/val splits')
    test_path=voc/'ImageSets/Main/test.txt'
    if test_path.exists() and set(read_ids(test_path)) & (train_ids|val_ids):raise ValueError('Test split overlaps train/val')
    if path.suffix=='.json':
        cfg=json.loads(path.read_text());cfg['data']=str(voc/'data.yaml')
        cfg.update(project=str(work.parent),name=work.name,seed=a.seed,device=a.device,workers=a.workers)
        if a.epochs:cfg['epochs']=a.epochs
        if a.batch:cfg['batch']=a.batch
        if a.checkpoint:cfg['model']=a.checkpoint
    else:
        cfg={k:copy.deepcopy(v) for k,v in runpy.run_path(str(path)).items() if not k.startswith('__')}
        cfg['work_dir']=str(work)
        if a.fresh:
            cfg['load_from']=None
            if record['backend']=='mmdet2':cfg['resume_from']=None
            else:cfg['resume']=False
        if a.checkpoint:cfg['load_from']=a.checkpoint
        if record['backend']=='mmdet3':
            for split,key in [('train','train_dataloader'),('val','val_dataloader'),('test','test_dataloader')]:
                if key in cfg:
                    dataset_config(cfg[key],voc,split)
                    cfg[key]['num_workers']=a.workers
                    cfg[key]['persistent_workers']=a.workers>0
            if a.epochs:cfg['train_cfg']['max_epochs']=a.epochs
            if a.batch:cfg['train_dataloader']['batch_size']=a.batch
            cfg['randomness']=dict(seed=a.seed,deterministic=True)
            cfg['launcher']='pytorch' if 'LOCAL_RANK' in os.environ else 'none'
            if 'auto_scale_lr' in cfg:
                cfg['auto_scale_lr'].pop('enalbe',None);cfg['auto_scale_lr']['enable']=False
        else:
            for split in ['train','val','test']:old_dataset_config(cfg['data'][split],voc,split)
            cfg['data']['workers_per_gpu']=a.workers
            if a.batch:cfg['data']['samples_per_gpu']=a.batch
            if a.epochs:cfg['runner']['max_epochs']=a.epochs
            cfg['seed']=a.seed;cfg['gpu_ids']=[int(a.device)]
            cfg['custom_imports']=dict(imports=['custom.bca_backbone'],allow_failed_imports=False)
    work.mkdir(parents=True,exist_ok=True)
    file=work/('resolved.json' if record['backend']=='ultralytics' else 'resolved.py')
    if file.suffix=='.json':file.write_text(json.dumps(cfg,indent=2))
    else:dump(cfg,file)
    (work/'run_manifest.json').write_text(json.dumps(dict(model=a.model,group=a.group,
        backend=record['backend'],source_config=record['config'],fresh=a.fresh,
        checkpoint=a.checkpoint,seed=a.seed,data_root=str(voc),
        mode='configuration_only' if a.dry_run else 'training'),indent=2))
    return record,cfg

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--voc',required=True)
    p.add_argument('--group',choices=['baseline','balanced','color_jitter','sd_lora'],required=True)
    p.add_argument('--work-dir',required=True);p.add_argument('--device',default='0')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int);p.add_argument('--batch',type=int)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--dry-run',action='store_true')
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--fresh',action='store_true');g.add_argument('--checkpoint')
    a=p.parse_args();record,cfg=resolve(a)
    print(json.dumps(dict(model=a.model,backend=record['backend'],work_dir=a.work_dir,dry_run=a.dry_run)))
    if a.dry_run:return
    if record['backend']=='mmdet3':
        from mmengine.config import Config
        from mmengine.runner import Runner
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        Runner.from_cfg(Config(cfg)).train()
    elif record['backend']=='mmdet2':
        import custom.bca_backbone
        from mmcv import Config
        from mmdet.apis import train_detector,set_random_seed
        from mmdet.datasets import build_dataset
        from mmdet.models import build_detector
        c=Config(cfg);set_random_seed(a.seed,deterministic=True)
        data=build_dataset(c.data.train)
        model=build_detector(c.model,train_cfg=c.get('train_cfg'),test_cfg=c.get('test_cfg'))
        model.init_weights();model.CLASSES=CLASSES
        train_detector(model,[data],c,distributed=False,validate=True,meta=dict(seed=a.seed,CLASSES=CLASSES))
    else:
        from ultralytics import YOLO
        weights=cfg.pop('model');model=YOLO(weights)
        model.train(**cfg)

if __name__=='__main__':main()
