"""Convert one prepared VOC dataset into a separate YOLO dataset; preserve source files."""
import argparse
import json
import shutil
from pathlib import Path
from .voc import CLASSES, objects, parse, read_ids, validate

def convert(source,destination):
    source,destination=Path(source),Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    # Retain VOC as audit/evaluation input and for train.py split validation.
    for folder in ('Annotations','ImageSets'):
        shutil.copytree(source/folder,destination/folder)
    for split in ('train','val','test'):
        ids=read_ids(source/'ImageSets/Main'/(split+'.txt'))
        image_dir=destination/'images'/split;label_dir=destination/'labels'/split
        image_dir.mkdir(parents=True);label_dir.mkdir(parents=True)
        for image_id in ids:
            root=parse(source/'Annotations'/(image_id+'.xml'));validate(root)
            w,h=int(root.findtext('size/width')),int(root.findtext('size/height'))
            rows=[]
            for name,(x1,y1,x2,y2) in objects(root):
                rows.append(f'{CLASSES.index(name)} {(x1+x2)/(2*w):.9f} {(y1+y2)/(2*h):.9f} {(x2-x1)/w:.9f} {(y2-y1)/h:.9f}')
            (label_dir/(image_id+'.txt')).write_text('\n'.join(rows)+'\n')
            shutil.copy2(source/'JPEGImages'/(image_id+'.jpg'),image_dir/(image_id+'.jpg'))
    # JSON is a valid YAML subset, accepted by yaml.safe_load.
    (destination/'data.yaml').write_text(json.dumps(dict(path=str(destination.resolve()),
        train='images/train',val='images/val',test='images/test',names=list(CLASSES)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--voc',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();convert(a.voc,a.output)
