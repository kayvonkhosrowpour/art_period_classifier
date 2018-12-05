"""
File: main.py
Author(s): Kayvon Khosrowpour
Date created: 10/23/18

Description:
Main driver of the attribute extractor script. Loads a config from provided arg
and according to the config, extracts attributes and stores them for each img
in a csv.

Example usage:
    python3 main.py -c configs/basic_test.ini
"""

import argparse
import configparser
import os
from ast import literal_eval
from attr.attr_extr import AttributeExtractor

class Config:
    def __init__(self):
        # load args from cmd line
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', type=str, help='supply path/to/CONFIG.ini')
        args = vars(parser.parse_args())
        config_file = args['c']
        if config_file is None:
            parser.error('No config file specified.')

        # load config file in a parser
        cparser = configparser.ConfigParser()
        cparser.read(config_file)

        # build columns according to config
        self.colsdict = self.init_columns_dict(cparser)

        # how many images to process
        self.process = cparser.getint('RUN', 'process')

        # who for?
        self.who = cparser.get('DIR', 'who')

        # load and save dir configs to self
        self.data_dir = os.path.normpath(cparser['DIR']['data_dir'])
        self.truth_csv = os.path.normpath(cparser['DIR']['truth_csv'])
        self.results_dir = os.path.normpath(cparser['DIR']['results_dir'])
        self.csv_title = cparser['DIR']['csv_title']

        # load and save algorithm parameters
        self.k = cparser.getint('PARAMS', 'k_kmeans')
        self.canny_1 = cparser.getfloat('PARAMS', 'canny_1')
        self.canny_2 = cparser.getfloat('PARAMS', 'canny_2')
        self.canny_3 = cparser.getfloat('PARAMS', 'canny_3')
        self.canny_4 = cparser.getfloat('PARAMS', 'canny_4')
        self.lpf = cparser.getint('PARAMS', 'lpf')
        self.bpf_1_1 = cparser.getint('PARAMS', 'bpf_1_1')
        self.bpf_1_2 = cparser.getint('PARAMS', 'bpf_1_2')
        self.bpf_1_2 = cparser.getint('PARAMS', 'bpf_1_2')
        self.bpf_2_1 = cparser.getint('PARAMS', 'bpf_2_1')
        self.bpf_2_2 = cparser.getint('PARAMS', 'bpf_2_2')
        self.bpf_3_1 = cparser.getint('PARAMS', 'bpf_3_1')
        self.bpf_3_2 = cparser.getint('PARAMS', 'bpf_3_2')
        self.hpf = cparser.getint('PARAMS', 'hpf')

        # misc
        self.keep_clusters = cparser.getboolean('MISC', 'keep_clusters')

        if (not (os.path.exists(self.data_dir) and
                 os.path.isfile(self.truth_csv) and
                 os.path.exists(self.results_dir))):
            print(self.data_dir)
            print(self.truth_csv)
            print(self.results_dir)
            raise configparser.ParsingError('One or more DIR configs do not exist.')

    def init_columns_dict(self, cparser):
        # which data points to include
        colsdict = cparser._sections['COLUMNS']
        true, false = ('true', '1', 'yes'), ('false', '0', 'no')
        for k, v in colsdict.items():
            if v.lower() in true:
                colsdict[k] = True
            elif v.lower() in false:
                colsdict[k] = False
            else:
                raise configparser.ParsingError('Invalid boolean in config')
        return colsdict

def main():
    # load configuration
    config = Config()

    # create extractor, extract, and save data from imgs
    extr = AttributeExtractor(config)
    extr.extract()
    extr.save()

if __name__ == '__main__':
    main()
