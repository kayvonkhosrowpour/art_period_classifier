"""
Filename: attr_extr.py
Author(s): Kayvon Khosrowpour
Date Created: 10/25/18

Description:
Contains an object AttributeExtractor for collecting image attributes.
"""

import pandas as pd
import os
import cv2
from attr.basic_attr import median_gray, median_hsv
from attr.entropy import get_entropy
from attr.constants import FrameColumns, TruthColumns
from utilities.file_handling import recursive_get_imgs_from_dir

class AttributeExtractor:
    def __init__(self, config):
        # save configurations
        self.data_dir = config.data_dir
        self.truth_csv = config.truth_csv
        self.results_dir = config.results_dir
        self.csv_title = config.csv_title
        self.colsdict = config.colsdict
        self.process = config.process 
        # build dataframe
        self.frame = self.init_frame(self.colsdict, self.truth_csv, self.process)
        
    def init_frame(self, colsdict, truth_csv, process):
        paths, filenames = recursive_get_imgs_from_dir(self.data_dir)
        # build columns
        frame_cols = [] + FrameColumns.info
        for (entry, value) in colsdict.items():
            if value:
                entry_cols = getattr(FrameColumns, entry)
                frame_cols += entry_cols
        frame = pd.DataFrame(columns=frame_cols)
        frame[FrameColumns.info[0]] = filenames
        frame[FrameColumns.info[1]] = paths
        
        # build truth dataframe
        print('Getting image truth data...')
        truth = pd.read_csv(truth_csv, index_col=TruthColumns.info[0])
        truth.drop(TruthColumns.drop_columns, axis='columns', inplace=True)
        
        # save correct labels
        frame = frame.merge(truth, on=[FrameColumns.info[0]], how='left')
        frame.dropna(subset=TruthColumns.dropna_columns, inplace=True)
        frame.sort_values(by=[FrameColumns.info[0]], kind='mergesort', inplace=True)

        # TODO: REMOVE IMAGES FROM STYLES WE'RE NOT CONSIDERING

        frame.set_index(FrameColumns.info[0], inplace=True)
        if process > 0:
            frame = frame.head(process)

        if len(filenames) != frame.index.size and process == -1:
            print('WARNING: %d images will not be analyzed because of '
                'missing style label.' % (len(filenames) - frame.index.size))

        return frame

    def extract(self):
        print('Extracting image attributes...')
        index = self.frame.index.tolist()
        paths = self.frame[FrameColumns.info[1]].tolist()
        for imgindex, imgpath in zip(index, paths):
            self.process_img(imgindex, imgpath)

    def process_img(self, imgindex, imgpath):
        # load image
        img = cv2.imread(imgpath)

        # extract desired attributes
        if self.colsdict['median_gray']:
            self.frame.loc[imgindex, FrameColumns.median_gray[0]] = median_gray(img)
        if self.colsdict['median_hsv']:
            hsv = median_hsv(img)
            for i, value in enumerate(FrameColumns.median_hsv):
                self.frame.loc[imgindex, value] = hsv[i]
        if self.colsdict['entropy']:
            self.frame.loc[imgindex, FrameColumns.entropy[0]] = get_entropy(img)
        if self.colsdict['kmeans']:
            # km = XXX
            # for value, i in enumerate(FrameColumns.kmeans):
            #     self.frame.loc[imgindex, value] = hsv[i]
            pass
        # TODO: add more extraction methods

    def save(self):
        print('Saving results...')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        csv_path = os.path.join(self.results_dir, self.csv_title + '.csv')
        i = 1
        while os.path.isfile(csv_path):
            csv_path = os.path.join(self.results_dir, self.csv_title + '(%d).csv' % (i))
            i += 1
        self.frame.to_csv(csv_path, sep=',', encoding='utf-8')
        print('Results saved to', csv_path, '\n')