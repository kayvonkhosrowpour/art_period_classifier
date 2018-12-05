"""
Filename: attr_extr.py
Author(s): Kayvon Khosrowpour
Date Created: 10/25/18

Description:
Contains an object AttributeExtractor for collecting image attributes.
To use the object, pass a main.Config object. Then call extract() to
extract image features. To output the image features to a csv file,
call save().
"""

import pandas as pd
import numpy as np
import os
import cv2
from attr.basic_attr import median_gray, median_hsv
from attr.entropy import get_entropy
from attr.kmeans import kmeans, kmeans_stats
from attr.constants import FrameColumns, TruthColumns, DISTANCE_COLORS
from attr.more_attributes import auto_canny, average_edges, x_y_gradient, low_pass_filter, high_pass_filter, band_pass_filter, stat_features
from utilities.file_handling import recursive_get_imgs_from_dir

class AttributeExtractor:
    """
    Given a configuration, acts as a storage object for retreiving truth data
    and extracted image features.
    """

    def __init__(self, config):
        """
        Initialize the object with the provided configuration. Also requires the setting
        of the attr.constants.DISTANCE_COLORS numpy array.

        Arguments:
            config: the main.Config object used to control the script.
        """

        # save configurations
        self.data_dir = config.data_dir
        self.truth_csv = config.truth_csv
        self.results_dir = config.results_dir
        self.csv_title = config.csv_title
        self.colsdict = config.colsdict
        self.process = config.process
        # save parameter configs
        self.k = config.k
        self.canny_1 = config.canny_1
        self.canny_2 = config.canny_2
        self.canny_3 = config.canny_3
        self.canny_4 = config.canny_4
        self.lpf= config.lpf
        self.bpf_1_1 = config.bpf_1_1
        self.bpf_1_2 = config.bpf_1_2
        self.bpf_2_1 = config.bpf_2_1
        self.bpf_2_2 = config.bpf_2_2
        self.bpf_3_1 = config.bpf_3_1
        self.bpf_3_2 = config.bpf_3_2
        self.hpf= config.hpf






        self.dist_colors = DISTANCE_COLORS
        # save misc
        self.keep_clusters = config.keep_clusters
        # build dataframe
        self.frame = self.init_frame(self.colsdict, self.truth_csv, self.process)

    def init_frame(self, colsdict, truth_csv, process):
        """
        Initializes the dataframe with the given configuration.

        Arguments:
            colsdict: the column dictionary from the config object.
            truth_csv: the csv containing truth data for each image in the dataset.
            process: the number of images to process
        Returns:
            frame: the initializes pandas frame containing the columns of data that
                is desired. The columns are specified by constants.FrameColumns and
                constants.TruthColumns. See constants.py for more info.
        """
        paths, filenames = recursive_get_imgs_from_dir(self.data_dir)

        # build basic columns
        frame_cols = [] + FrameColumns.info
        for (entry, value) in colsdict.items():
            if value:
                entry_cols = getattr(FrameColumns, entry)
                frame_cols += entry_cols

        # build parameter-specific columns for kmeans
        FrameColumns.num_px = ['log_num_px(cluster_%d)' % k for k in range(0, self.k)]
        frame_cols += FrameColumns.num_px
        if self.keep_clusters:
            FrameColumns.color = ['color(cluster_%d)' % k for k in range(0, self.k)]
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

        # save correct labels
        frame = frame.merge(truth, on=[FrameColumns.info[0]], how='left')
        frame.dropna(subset=TruthColumns.dropna_columns, inplace=True)

        total = len(filenames)
        if total != frame.index.size and process == -1:
            print('WARNING: %d images will not be analyzed because of '
                'missing style label.' % (total - frame.index.size))
            total = frame.index.size

        # remove images from styles we're not considering
        style = TruthColumns.dropna_columns[0]
        frame = frame[frame[style].isin(TruthColumns.keep_styles)]

        if total != frame.index.size and process == -1:
            print('NOTICE: ignoring %d of %d images not in TruthColumns.keep_styles' \
                % (total - frame.index.size, total))

        # sort by file name
        frame.sort_values(by=[FrameColumns.info[0]], kind='mergesort', inplace=True)

        frame.set_index(FrameColumns.info[0], inplace=True)
        if process > 0:
            frame = frame.head(process)

        if total != len(filenames):
            print() # style, hoe

        return frame

    def extract(self):
        """
        Extracts image features from all the images within the initialized pandas frame.
        """

        print('Extracting image attributes...')
        index = self.frame.index.tolist()
        paths = self.frame[FrameColumns.info[1]].tolist()
        for imgindex, imgpath in zip(index, paths):
            print('Processing', imgindex)
            self.process_img(imgindex, imgpath)

    def process_img(self, imgindex, imgpath):
        """
        For an image with `imgindex` into the pandas frame, extracts all necessary
        features for the image and updates the table.

        Arguments:
            imgindex: the index into the pandas frame for the given image.
            imgpath: the path to the image.
        """

        # load image
        img = cv2.imread(imgpath)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("processing another image...")


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
        if self.colsdict['average_edges']:
            self.frame.loc[imgindex, FrameColumns.average_edges[0]] = average_edges(img_gray, self.canny_1)
            self.frame.loc[imgindex, FrameColumns.average_edges[1]] = average_edges(img_gray, self.canny_2)
            self.frame.loc[imgindex, FrameColumns.average_edges[2]] = average_edges(img_gray, self.canny_3)
            self.frame.loc[imgindex, FrameColumns.average_edges[3]] = average_edges(img_gray, self.canny_4)
        if self.colsdict['x_y_gradient']:
            x, y = x_y_gradient(img_gray)
            self.frame.loc[imgindex, FrameColumns.x_y_gradient[0]] = x
            self.frame.loc[imgindex, FrameColumns.x_y_gradient[1]] = y
        if self.colsdict['freq_bands']:
            energy, median, deviation = low_pass_filter(img_gray, self.lpf)
            self.frame.loc[imgindex, FrameColumns.freq_bands[0]] = energy
            self.frame.loc[imgindex, FrameColumns.freq_bands[1]] = median
            self.frame.loc[imgindex, FrameColumns.freq_bands[2]] = deviation
            energy, median, deviation = band_pass_filter(img_gray, self.bpf_1_2, self.bpf_1_1)
            self.frame.loc[imgindex, FrameColumns.freq_bands[3]] = energy
            self.frame.loc[imgindex, FrameColumns.freq_bands[4]] = median
            self.frame.loc[imgindex, FrameColumns.freq_bands[5]] = deviation
            energy, median, deviation = band_pass_filter(img_gray, self.bpf_2_2, self.bpf_2_1)
            self.frame.loc[imgindex, FrameColumns.freq_bands[6]] = energy
            self.frame.loc[imgindex, FrameColumns.freq_bands[7]] = median
            self.frame.loc[imgindex, FrameColumns.freq_bands[8]] = deviation
            energy, median, deviation = band_pass_filter(img_gray, self.bpf_3_2, self.bpf_3_1)
            self.frame.loc[imgindex, FrameColumns.freq_bands[9]] = energy
            self.frame.loc[imgindex, FrameColumns.freq_bands[10]] = median
            self.frame.loc[imgindex, FrameColumns.freq_bands[11]] = deviation
            energy, median, deviation = high_pass_filter(img_gray, self.hpf)
            self.frame.loc[imgindex, FrameColumns.freq_bands[12]] = energy
            self.frame.loc[imgindex, FrameColumns.freq_bands[13]] = median
            self.frame.loc[imgindex, FrameColumns.freq_bands[14]] = deviation
        if self.colsdict['stats']:
            dev, kurtosis, skew = stat_features(img_gray)
            self.frame.loc[imgindex, FrameColumns.stats[0]] = dev
        if self.colsdict['freq_stats']:
            img_fft = abs(np.fft.fftshift(np.fft.fft2(img_gray)))
            dev, kurtosis, skew = stat_features(img_fft)
            self.frame.loc[imgindex, FrameColumns.freq_stats[0]] = dev







        # TODO: add more extraction methods

    def store_kmeans_entry(self, img, imgindex):
        """
        For a single image instance, stores only the kmeans data for the image into
        the pandas table. This was moved from self.process_img because it's kinda long.

        Arguments:
            img: the bgr image from self.process_img.
            imgindex: the imgindex from self.process_img.
        """

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
        if self.keep_clusters:
            for i, value in enumerate(FrameColumns.color):
                self.frame.loc[imgindex, value] = colors[i]
        # store the color distances
        for i, value in enumerate(FrameColumns.dist_cluster_colors):
            self.frame.loc[imgindex, value] = color_distances[i]

    def save(self):
        """
        Saves the results after evaluating all images. Note: calling save() before calling
        extract() will output an empty dataframe. This is useful for seeing the columns, but
        not much else.

        Outputs:
            a csv in self.results_dir with a title self.csv_title.
        """
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
