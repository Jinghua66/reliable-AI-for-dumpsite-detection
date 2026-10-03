"""Create a training-only validation holdout with source-grouped augmentation.

All augmentations descending from validation images are excluded from training.
Train the generator using the resulting training partition before generating patches.
"""
import argparse
import csv
import json
import random
from pathlib import Path
from .voc import read_ids
from .augment import load_manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',default='manifests/original_train.txt')
    p.add_argument('--test',default='manifests/original_test.txt')
    p.add_argument('--manifest',default='manifests/augmentation_plan.csv')
    p.add_argument('--val-fraction',type=float,default=.1);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if not 0<a.val_fraction<1:p.error('val-fraction must be between zero and one')
    original=read_ids(a.train);test=read_ids(a.test)
    if set(original)&set(test):raise ValueError('Original training/test overlap')
    shuffled=list(original);random.Random(a.seed).shuffle(shuffled)
    n=max(1,round(len(shuffled)*a.val_fraction));validation=set(shuffled[:n])
    train=[i for i in original if i not in validation];val=[i for i in original if i in validation]
    if not train:raise ValueError('Holdout removed all training images')
    rows=load_manifest(a.manifest)
    if any(r['source_id'] not in set(original) for r in rows):raise ValueError('Unknown augmentation source')
    kept=[r for r in rows if r['source_id'] not in validation]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for key,values in [('train',train),('val',val),('test',test)]:
        (out/(key+'.txt')).write_text('\n'.join(values)+'\n')
    with (out/'augmentation_plan.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(kept)
    report=dict(protocol='source-grouped training-validation split',seed=a.seed,
        lora_must_be_refit_on_new_train_split=True,
        real_train=len(train),real_val=len(val),test=len(test),kept_augmentations=len(kept),excluded_augmentations=len(rows)-len(kept))
    (out/'split_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
