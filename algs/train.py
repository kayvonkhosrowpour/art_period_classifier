"""
File: train.py
Author(s): Kayvon Khosrowpour
Date created: 11/26/18

Description:
Given a desired model and CSV, will perform training on the given
data using that model. Includes support for a random forest
classifier, an AdaBoost classifier, and XGBoost classifier.

For RandomForestClassifier:
    python3 train.py -c configs/rfc_config.ini
For AdaBoostClassifier:
    python3 train.py -c configs/adab_defaults.ini
For XGBoostClassifier:
    python3 train.py -c configs/xgb_defaults.ini
"""

import argparse
import configparser
import pandas as pd
import os

class Config:
    def __init__(self, cparser):
        """
        Initialize base Config. Save the config.ini passed in from cmd line.
        Arguments:
            cparser: a configparser.ConfigParser object.
        """
        self.csv = cparser.get('DIR', 'data_csv')
        self.model_title = cparser.get('DIR', 'model_title')
        self.save_dir = os.path.join(os.path.normpath(cparser.get('DIR', 'save_dir')),
            self.model_title)
        self.frame, self.mapping = self.get_data()
        self.setup_files()

    def get_data(self):
        """
        If `self.csv` is initialized, loads the dataframe from the csv, converts
        class names 'style' to integers.
        Returns:
            frame: the dataframe with classnames converted to integers.
            mapping: the classname to integer mappings in a dict.
        """
        frame = pd.read_csv(self.csv)
        frame.drop(['filename', 'path'], axis='columns', inplace=True)
        frame['style'] = pd.Categorical(frame['style'])
        mapping = dict(enumerate(frame['style'].cat.categories))
        frame['style'] = frame['style'].cat.codes
        return frame, mapping

    def setup_files(self):
        """
        Prepares the file/dir structure for the model training. Creates
        a directory according to `self.save_dir` and saves the classname
        to integer mapping as a simple textfile.
        """
        os.mkdir(self.save_dir)
        map_file = open(os.path.join(self.save_dir, 'mapping.txt'), 'w')
        map_file.write(str(self.mapping))
        map_file.close()

class RFC_Config(Config):
    def __init__(self, cparser):
        super().__init__(cparser)


class ADA_Config(Config):
    def __init__(self, cparser):
        super().__init__(cparser)

class XGB_Config(Config):
    def __init__(self, cparser):
        super().__init__(cparser)

def parse():
    """
    Parse the cmd line arguments to find csv location and determine what analysis to run.
    Returns:
        config: the config object encapsulating the configuration params.
    """
    # parse cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, help='supply path/to/config.ini', required=True)
    args = parser.parse_args()
    if args.c is None:
        parser.error('No config file provided.')

    # parse config file
    cparser = configparser.ConfigParser()
    cparser.read(args.c)
    model = cparser.get('TYPE', 'model')
    if model == 'RandomForestClassifier':
        config = RFC_Config(cparser)
    elif model == 'AdaBoostClassifier':
        config = ADA_Config(cparser)
    elif model == 'XGBoostClassifier':
        config = XGB_Config(cparser)
    else:
        cparser.ParsingError('Invalid model')

    return config

def main():
    config = parse()

if __name__ == '__main__':
    main()