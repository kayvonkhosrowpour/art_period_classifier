"""
Filename: get.py
Author(s): Kayvon Khosrowpour

Description:
Copies images from a source location that are of interest
to the art classifier. We are only interested in the styles
defined in `keep_styles` in this script.
"""

from file_handling import get_imgs_from_dir
from shutil import copyfile
import pandas as pd
import os

def get_valid(truth, names, paths, train):
    # get all valid paths
    truth = truth[truth['in_train']==train]
    valid_names = truth.index.tolist()
    valid_paths = []
    for vn in valid_names:
        try:
            i = names.index(vn)
            valid_paths.append(paths[i])
        except ValueError:
            pass
    assert len(valid_paths) == len(valid_names)
    return valid_paths, valid_names

keep_styles = [
    'Medieval',
    'Renaissance',
    'Baroque',
    'Realism',
    'Impressionism',
    'Post-Impressionism',
    'Art Nouveau (Modern)',
    'Expressionism',
    'Abstract Expressionism'
]

med_rename = [
    'Byzantine',
    'Romanesque',
    'Mosan art',
    'Gothic',
    'International Gothic',
    'Early Byzantine',
    'Middle Byzantine'
]

ren_rename = [
    'Mannerism (Late Renaissance)',
    'High Renaissance',
    'Early Renaissance',
    'Renaissance',
    'Northern Renaissance'
]

# source
data_src = '/Users/kayvon/code/divp/proj/data/data_table/data_table_who.csv'
train_src = '/Volumes/Hey/train/train'
test_src = '/Volumes/Hey/test/test'

# destination
train_dst = '/Users/kayvon/Downloads/train'
test_dst = '/Users/kayvon/Downloads/test'

# load imgs from source
train_paths, train_names = get_imgs_from_dir(train_src)
test_paths, test_names = get_imgs_from_dir(test_src)

# load data
truth = pd.read_csv(data_src)

# set index to filename
truth.set_index('filename', inplace=True)

# get valid train imgs
valid_train_paths, valid_train_names = \
    get_valid(truth, train_names, train_paths, True)

# get valid test imgs
valid_test_paths, valid_test_names = \
    get_valid(truth, test_names, test_paths, False)

# drop unwanted imgs
truth = truth[~(truth.index.isin(valid_train_names) | truth.index.isin(valid_train_names))]

# show number in each class
print('Total number of images:', truth.size)
for i in range(0, len(keep_styles)):
    print(truth[truth['style']==keep_styles[i]].size, keep_styles[i])

# confirm
from pprint import pprint
pprint(valid_train_paths[:10])
pprint(valid_train_names[:10])
print()
pprint(valid_test_paths[:10])
pprint(valid_test_names[:10])
print()
resp = None
while (resp not in ['Y', 'N']):
    resp = input('Good? (Y/N) ')
if resp == 'Y':
    for v_tp, v_tn in zip(valid_train_paths, valid_train_names):
        print('Copying\nsrc=%s\ndst=%s\n' % (v_tp, os.path.join(train_dst, v_tn)))
        copyfile(v_tp, os.path.join(train_dst, v_tn))
    for v_tp, v_tn in zip(valid_test_paths, valid_test_names):
        print('Copying\nsrc=%s\ndst=%s\n' % (v_tp, os.path.join(test_dst, v_tn)))
        copyfile(v_tp, os.path.join(test_dst, v_tn))

