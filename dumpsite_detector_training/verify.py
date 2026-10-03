"""CPU-only verification of the public examples and all 18 configuration records."""
import argparse
import csv
import json
import runpy
from collections import Counter
from pathlib import Path
from PIL import Image
from src.voc import CLASSES, objects, parse, read_ids, sha256, validate
from src.augment import load_manifest

def audit(voc, split):
    voc=Path(voc); ids=read_ids(split); counts=Counter(); duplicates=[]
    for image_id in ids:
        root=parse(voc/'Annotations'/(image_id+'.xml')); validate(root)
        obs=objects(root); counts.update(c for c,_ in obs)
        if len(obs)>1 and len(set(b for _,b in obs))==1: duplicates.append(image_id)
        with Image.open(voc/'JPEGImages'/(image_id+'.jpg')) as im:
            if im.size != (int(root.findtext('size/width')),int(root.findtext('size/height'))):
                raise AssertionError(f'Dimensions differ: {image_id}')
    return dict(images=len(ids),unique_images=len(set(ids)),instances=sum(counts.values()),classes=dict(counts),collapsed_boxes=duplicates)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--voc',default='examples/VOC2007');p.add_argument('--split',default='examples/VOC2007/ImageSets/Main/train.txt')
    p.add_argument('--output',default='outputs/verification.json');p.add_argument('--examples',action='store_true')
    a=p.parse_args(); report=audit(a.voc,a.split)
    if report['collapsed_boxes']:raise AssertionError(report['collapsed_boxes'])
    if a.examples:
        splits={k:set(read_ids(Path(a.voc)/'ImageSets/Main'/(k+'.txt'))) for k in ('train','val','test')}
        assert not splits['train'] & splits['val'] and not splits['train'] & splits['test'] and not splits['val'] & splits['test']
        for k in ('val','test'): audit(a.voc,Path(a.voc)/'ImageSets/Main'/(k+'.txt'))
        for row in load_manifest('examples/manifest.csv'):
            assert row['source_id'] in splits['train']
            source=objects(parse(Path(a.voc)/'Annotations'/(row['source_id']+'.xml')))
            expected=objects(parse(Path('examples/expected/Annotations')/(row['image_id']+'.xml')))
            assert source == expected, 'Same-position synthesis must preserve ALL objects'
        models=json.loads(Path('configs/models.json').read_text())
        assert len(models)==18 and len({m['id'] for m in models})==18
        for model in models:
            config=Path(model['config']); assert config.is_file()
            if config.suffix=='.py':
                data=runpy.run_path(str(config)); assert '_base_' not in data
                assert data.get('model')
            else: assert json.loads(config.read_text())['model']
        hashes=json.loads(Path('examples/checksums.json').read_text())
        for name,digest in hashes.items():assert sha256(name)==digest,name
        report.update(configurations=18,example_splits={k:len(v) for k,v in splits.items()},checksums=len(hashes),annotation_preservation='passed')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
