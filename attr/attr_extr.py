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
from attr.kmeans import kmeans, kmeans_stats
from attr.constants import FrameColumns, TruthColumns, DISTANCE_COLORS
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
        # save parameter configs
        self.k = config.k
        self.dist_colors = DISTANCE_COLORS
        # build dataframe
        self.frame = self.init_frame(self.colsdict, self.truth_csv, self.process)
        
    def init_frame(self, colsdict, truth_csv, process):
        paths, filenames = recursive_get_imgs_from_dir(self.data_dir)
      
        # build basic columns
        frame_cols = [] + FrameColumns.info
        for (entry, value) in colsdict.items():
            if value:
                entry_cols = getattr(FrameColumns, entry)
                frame_cols += entry_cols

        # build parameter-specific columns for kmeans
        FrameColumns.num_px = ['num_px(cluster_%d)' % k for k in range(0, self.k)]
        FrameColumns.color = ['color(cluster_%d)' % k for k in range(0, self.k)]
        frame_cols += FrameColumns.num_px
        frame_cols += FrameColumns.color

        dist_cluster_colors = []
        for k in range (0, self.k):
            dist_cluster_colors += ['dist(cluster%d_dcolor%d)' % (k, d) for d in range(0, self.dist_colors.shape[0])]
        FrameColumns.dist_cluster_colors = dist_cluster_colors
        frame_cols += dist_cluster_colors

        # create frame and update with file data
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
            self.store_kmeans_entry(img, imgindex)

        # TODO: add more extraction methods

    def store_kmeans_entry(self, img, imgindex):
        clusters,_ = kmeans(img, self.k)
        num_px,colors,dist_top_2_vectors,ratio_top2_clusters,ratio_last2_clusters,\
            color_distances = kmeans_stats(clusters, dist_colors=self.dist_colors)
        dist_ratios = [dist_top_2_vectors,ratio_top2_clusters,ratio_last2_clusters]
        color_distances = color_distances.flatten()
        
        # store the kmeans ratios and differences
        for i, value in enumerate(FrameColumns.kmeans):
            self.frame.loc[imgindex, value] = dist_ratios[i]
        # store the number of pixels per cluster
        for i, value in enumerate(FrameColumns.num_px):
            self.frame.loc[imgindex, value] = num_px[i]
        # store the color for each cluster
        for i, value in enumerate(FrameColumns.color):
            self.frame.loc[imgindex, value] = colors[i]
        # store the color distances
        for i, value in enumerate(FrameColumns.dist_cluster_colors):
            self.frame.loc[imgindex, value] = color_distances[i]

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