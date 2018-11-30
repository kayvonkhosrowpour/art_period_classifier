"""
File: train.py
Author(s): Kayvon Khosrowpour
Date created: 11/29/18

Description:
Implementations of classifier objects for classifiers. Abstracts
train() and save() by providing a single config.

Includes:
RandomForestClassifier, AdaBoostClassifier, XGBoostClassifier
"""

import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from time import sleep

class Model:
    def __init__(self, config):
        self.config = config
        self.clf = None
    def save(self):
        """
        Saves the model `self.clf` to the specified directory from `self.config`.
        """
        path = os.path.join(self.config.save_dir, self.config.model_title+'.pkl')
        print('\nSaving clf model at', path, '...')
        joblib.dump(self.clf, path)
        print('Save successful!\n')

class RFC_Model(Model):
    def __init__(self, config):
        """
        Initialize the RFC_Model (RandomForestClassifier wrapper) with
        the appropriate parameters according to the config.
        """
        super().__init__(config)
        if config.cv_type == 'GridSearchCV':
            self.rfc = RandomForestClassifier()
            self.clf = GridSearchCV(
                self.rfc,
                config.param_grid,
                scoring=config.scoring,
                n_jobs=config.n_jobs,
                pre_dispatch=config.pre_dispatch,
                cv=config.cv
            )
        elif config.cv_type is None:
            self.rfc = RandomForestClassifier(
                n_estimators=config.n_estimators,
                criterion=config.criterion,
                max_depth=config.max_depth,
                min_samples_split=config.min_samples_split,
                min_samples_leaf=config.min_samples_leaf,
                min_weight_fraction_leaf=config.min_weight_fraction_leaf,
                max_features=config.max_features,
                max_leaf_nodes=config.max_leaf_nodes,
                n_jobs=config.n_jobs,
                verbose=config.verbose
            )
            self.clf = self.rfc

    def train(self):
        """
        Train on the config data according to the initialized classifier
        `self.clf`. Assumes that all features should be included and that
        the class column is 'style'.
        """
        print('Beginning training...'); sleep(1)
        frame = self.config.frame
        X, y = frame.drop('style', axis=1), frame['style']
        self.clf.fit(X, y)

class ADA_Model(Model):
    def __init__(self, config):
        super().__init__(config)

    def train(self):
        pass

class XGB_Model(Model):
    def __init__(self, config):
        super().__init__(config)

    def train(self):
        pass

def build_model(config):
    """
    Builds the appropriate model from the config
    """
    if config.model == 'RandomForestClassifier':
        model = RFC_Model(config)
    elif config.model == 'AdaBoostClassifier':
        model = ADA_Model(config)
    elif config.model == 'XGBoostClassifier':
        model = XGB_Model(config)
    else:
        raise ValueError('Invalid config.model initialization.')

    return model
