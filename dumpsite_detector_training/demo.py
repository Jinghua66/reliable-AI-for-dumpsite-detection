"""Build all four small datasets, export 18 resolved training configs, and audit outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from src.augment import build
from src.convert_yolo import convert
from verify import audit

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='outputs/demo')
    a=p.parse_args();out=Path(a.output)
    if out.exists():p.error('Choose a NEW output directory; existing results will not be overwritten')
    out.mkdir(parents=True)
    reports={}
    for group in ['baseline','balanced','color_jitter','sd_lora']:
        dest=out/group/'VOC2007'
        build('examples/VOC2007','examples/VOC2007/ImageSets/Main/train.txt','examples/manifest.csv',
              'examples/patches',dest,group,[(k,'examples/VOC2007/ImageSets/Main/'+k+'.txt') for k in ('val','test')])
        reports[group]=audit(dest,dest/'ImageSets/Main/train.txt')
    for row in __import__('csv').DictReader(open('examples/manifest.csv',encoding='utf-8')):
        from src.voc import objects,parse
        source=objects(parse(Path('examples/VOC2007/Annotations')/(row['source_id']+'.xml')))
        for group in ['balanced','color_jitter','sd_lora']:
            assert objects(parse(out/group/'VOC2007/Annotations'/(row['image_id']+'.xml')))==source
    convert(out/'sd_lora/VOC2007',out/'yolo')
    catalog=json.loads(Path('configs/models.json').read_text())
    for m in catalog:
        voc=out/('yolo' if m['backend']=='ultralytics' else 'sd_lora/VOC2007')
        subprocess.run([sys.executable,'train.py','--model',m['id'],'--group','sd_lora','--voc',str(voc),
            '--work-dir',str(out/'resolved'/m['id']),'--fresh','--dry-run','--epochs','1','--batch','1','--workers','0'],check=True)
    reports['training_validation']='18 configurations resolved; GPU training not executed'
    (out/'report.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,ensure_ascii=False))

if __name__=='__main__':main()
