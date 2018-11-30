"""
File: configs.py
Author(s): Kayvon Khosrowpour
Date created: 11/28/18

Description:
Implementations of configuration objects for various classifiers
that are implemented in this training module. Also includes
initialization functions to convert a config path to an object.

Includes:
RandomForestClassifier, AdaBoostClassifier, XGBoostClassifier
"""

import argparse
import configparser
import pandas as pd
import os
import importlib.util
import pprint

class Config:
    def __init__(self, cparser):
        """
        Initialize base Config. Save the config.ini passed in from cmd line.
        Arguments:
            cparser: a configparser.ConfigParser object.
        """
        self.model = cparser.get('TYPE', 'model')
        self.train_data_csv = cparser.get('DIR', 'train_data_csv')
        self.data_table = cparser.get('DIR', 'data_table')
        self.model_title = cparser.get('DIR', 'model_title')
        self.save_dir = os.path.join(os.path.normpath(cparser.get('DIR', 'save_dir')),
            self.model_title)
        self.frame, self.mapping = self.get_data()

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    def __repr__(self):
        return str(self)

    def get_data(self):
        """
        If `self.train_data_csv` is initialized, loads the dataframe from the csv, converts
        class names 'style' to integers.
        Returns:
            frame: the dataframe with classnames converted to integers.
            mapping: the classname to integer mappings in a dict.
        """
        frame = pd.read_csv(self.train_data_csv)
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
        self.cv_type = none_get(cparser, 'TUNING', 'cv_type')
        if self.cv_type == 'GridSearchCV':
            self.init_gridsearchCV(cparser)
        elif self.cv_type is None:
            self.init_hyperparams(cparser)
        else:
            raise configparser.ParsingError('Invalid cv_type %s' % str(self.cv_type))

    def init_gridsearchCV(self, cparser):
        """
        If the `self.cv_type` is GridSearchCV, then set up the parameters.
        Arguments:
            cparser: the ConfigParser
        """
        path = os.path.normpath(none_get(cparser, 'TUNING', 'param_grid_module'))
        spec = importlib.util.spec_from_file_location('param_grid', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.param_grid = module.param_grid
        self.scoring = none_get(cparser,'GRIDSEARCHCVPARAMS', 'scoring')
        self.n_jobs = none_get(cparser,'GRIDSEARCHCVPARAMS', 'n_jobs', )
        self.pre_dispatch = none_get(cparser,'GRIDSEARCHCVPARAMS', 'pre_dispatch')
        self.cv = none_get(cparser, 'GRIDSEARCHCVPARAMS', 'cv', type=int)

    def init_hyperparams(self, cparser):
        """
        If the `self.cvtype` is not GridSearchCV, then setup the individual
        hyperparameters.
        Arguments:
            cparser: the ConfigParser
        """
        self.n_estimators = none_get(cparser, 'HYPERPARAMS', 'n_estimators', type=int)
        self.criterion = none_get(cparser, 'HYPERPARAMS', 'criterion')
        self.max_depth = none_get(cparser, 'HYPERPARAMS', 'max_depth')
        self.min_samples_split = none_get(cparser, 'HYPERPARAMS', 'min_samples_split', 
            type=int)
        self.min_samples_leaf = none_get(cparser, 'HYPERPARAMS', 'min_samples_leaf', 
            type=int)
        self.min_weight_fraction_leaf = none_get(cparser, 'HYPERPARAMS', 
            'min_weight_fraction_leaf', type=float)
        self.max_features = none_get(cparser, 'HYPERPARAMS', 'max_features')
        self.max_leaf_nodes = none_get(cparser, 'HYPERPARAMS', 'max_leaf_nodes')
        self.n_jobs = none_get(cparser, 'HYPERPARAMS', 'n_jobs')

class ADA_Config(Config):
    def __init__(self, cparser):
        super().__init__(cparser)

class XGB_Config(Config):
    def __init__(self, cparser):
        super().__init__(cparser)

def none_get(cparser, x, y, type=str):
    if type is str:
        z = cparser.get(x, y)
    elif type is int:
        z = cparser.getint(x, y)
    elif type is float:
        z = cparser.getfloat(x, y)
    return None if z == 'None' else z

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

    print('========== Confirm model config ===========')
    print(config)
    print('===========================================')
    resp = None
    while not (resp == 'Y' or resp == 'N' or resp == 'DEBUG'):
        resp = input('Confirm? (Y/N/DEBUG) ')
    print()
    if resp == 'Y':
        config.setup_files()
    elif resp == 'N':
        config = None

    return config, resp == 'Y'