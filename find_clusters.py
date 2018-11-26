"""
File: find_clusters.py
Author(s): Kayvon Khosrowpour
Date created: 11/24/18

Description:
Finds the most dominant colors in a particular set of images.
"""

import argparse
import numpy as np
import cv2
import pandas as pd
import random
import os
from attr.constants import TruthColumns
from attr.kmeans import kmeans
from utilities.file_handling import recursive_get_imgs_from_dir
import matplotlib.pyplot as plt

def display_img(name, img, show=True, cvt=True):
    figure = plt.figure()
    axes = plt.axes()
    axes.set_title(name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes.imshow(img)
    if show:
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, help='supply valid style')
    args = vars(parser.parse_args())
    style = args['t']
    if style is None:
        err = 'Must provide one of the following styles:\n' + '\n'.join(TruthColumns.keep_styles)
        parser.error(err)
    return style

def get_imgs(style, truth_csv, location, shuffle=True):
    paths, names = recursive_get_imgs_from_dir(location)
    # read labels and get non-null labels
    truth = pd.read_csv(truth_csv, index_col=TruthColumns.info[0])
    truth.drop(['artist', 'title', 'genre', 'date'], axis='columns', inplace=True)
    truth.dropna(subset=['style'], inplace=True)
    truth = truth[truth['style'].isin([style])] # images only in style
    truth = truth[truth.index.isin(names)] # images only in provided path
    # get all valid paths
    valid_names = truth.index.tolist()
    valid_paths = []
    for vn in valid_names:
        try:
            i = names.index(vn)
            valid_paths.append(paths[i])
        except ValueError:
            pass
    assert len(valid_paths) == len(valid_names)

    if shuffle:
        random.shuffle(valid_paths), random.shuffle(valid_names)

    return valid_paths, valid_names

def get_resized(path, resize_len, cvt=False):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    resized = cv2.resize(img, (resize_len, resize_len))
    if cvt:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return resized

def make_composite(valid_paths, resize_len=300):
    sides = int(np.sqrt(len(valid_paths[:-10])))
    print('Creating composite of size %dx%d' % (sides, sides))
    l = w = sides * resize_len
    composite = np.zeros((w, l, 3), dtype=np.uint8)
    
    for s_i in range(0, sides):
        for s_j in range(0, sides):
            i = s_i * sides + s_j
            print('Composing', i, 'of', sides*sides)
            img = get_resized(valid_paths[i], resize_len)
            j = -1
            while (img is None):
                print('Found invalid image!')
                img = get_resized(valid_paths[j], resize_len)
                j -= 1
            x_start, y_start = s_i * resize_len, s_j * resize_len
            x_end, y_end = x_start + resize_len, y_start + resize_len
            composite[x_start:x_end,y_start:y_end,:] = img

    return composite

def save(clusters, kmeaned_img, directory):
    cv2.imwrite(os.path.join(directory, 'kmeaned_img.jpg'), kmeaned_img)
    file = open(os.path.join(directory, 'clusters.txt'), 'w') 
    for (px, color) in clusters:
        num_sp = 15-len(str(px))
        file.write(str(px) + ' '*num_sp + str(color) + '\n')
    file.close()

def main():
    style = parse_args()
    location = '/Volumes/Hey/train/train'
    directory = '/Users/kayvon/code/divp/proj/clustering/impressionism'
    if len(os.listdir(directory)) != 0:
        print('provided directory not empty!'), exit(-1)
    truth_csv = '/Users/kayvon/code/divp/proj/train/train_info.csv'
    valid_paths, valid_names = get_imgs(style, truth_csv, location)
    if len(valid_paths) == 0:
        print('no valid paths!'), exit(-1)
    comp = make_composite(valid_paths)
    print('Running kmeans...')
    clusters, kmeaned_img = kmeans(comp, K=25)
    save(clusters, kmeaned_img, directory)    

if __name__ == '__main__':
    main()