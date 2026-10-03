import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pytest
from src.augment import compose, build
from src.voc import objects, parse
from src.evaluate import score

def root():
    return ET.fromstring('''<annotation><size><width>20</width><height>20</height></size>
    <object><name>agriculture forestry</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>5</xmax><ymax>6</ymax></bndbox></object>
    <object><name>construction waste</name><bndbox><xmin>10</xmin><ymin>11</ymin><xmax>15</xmax><ymax>16</ymax></bndbox></object></annotation>''')

def test_replacement_preserves_other_objects_and_pixels():
    r=root();image=np.zeros((20,20,3),np.uint8);patch=np.full((3,3,3),255,np.uint8)
    result,updated=compose(image,r,0,patch)
    assert objects(updated)==objects(r)
    assert np.all(result[2:6,1:5]==255)
    result[2:6,1:5]=0
    assert np.array_equal(result,image)

def test_invalid_object_does_not_silently_overwrite():
    with pytest.raises(IndexError):compose(np.zeros((20,20,3),np.uint8),root(),9,np.zeros((3,3,3),np.uint8))



def test_one_to_one_matching_counts_duplicate_predictions(tmp_path):
    (tmp_path/'Annotations').mkdir();ET.ElementTree(root()).write(tmp_path/'Annotations/a.xml')
    split=tmp_path/'test.txt';split.write_text('a\n')
    row=dict(image_id='a',**{'class':'agriculture forestry'},score=.9,xmin=1,ymin=2,xmax=5,ymax=6)
    metrics=score(tmp_path,split,[row,dict(row,score=.8)])
    assert metrics['classes']['agriculture forestry']['TP']==1
    assert metrics['classes']['agriculture forestry']['FP']==1
    assert metrics['classes']['construction waste']['FN']==1

def test_missing_generated_patch_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        build('examples/VOC2007','examples/VOC2007/ImageSets/Main/train.txt',
              'examples/manifest.csv',tmp_path/'missing',tmp_path/'out','sd_lora')

def test_real_multiobject_example_matches_expected(tmp_path):
    import cv2
    import csv
    build('examples/VOC2007','examples/VOC2007/ImageSets/Main/train.txt',
          'examples/manifest.csv','examples/patches',tmp_path/'out','sd_lora')
    for row in csv.DictReader(open('examples/manifest.csv',encoding='utf-8')):
        name=row['image_id']
        actual=cv2.imread(str(tmp_path/'out/JPEGImages'/(name+'.jpg')))
        expected=cv2.imread(str(Path('examples/expected/JPEGImages')/(name+'.jpg')))
        # JPEG implementations can differ slightly across OS builds; geometry must be exact.
        assert actual.shape == expected.shape
        assert np.abs(actual.astype(np.int16)-expected.astype(np.int16)).mean() < 1.0
        assert objects(parse(tmp_path/'out/Annotations'/(name+'.xml')))==objects(parse(Path('examples/expected/Annotations')/(name+'.xml')))

def test_test_image_cannot_be_used_for_augmentation(tmp_path):
    import csv
    from src.voc import read_ids
    row=dict(image_id='illegal',source_id=read_ids('examples/VOC2007/ImageSets/Main/test.txt')[0],
             object_index=0,**{'class':'domestic garbage'},patch_file='none.jpg',generation_seed=0,transform_seed=0)
    f=tmp_path/'manifest.csv'
    with f.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    with pytest.raises(ValueError,match='outside training'):
        build('examples/VOC2007','examples/VOC2007/ImageSets/Main/train.txt',f,'examples/patches',tmp_path/'out','sd_lora')
