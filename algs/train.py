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
from sklearn.metrics import (f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix)
from time import sleep
from pprint import pprint
import pandas as pd
import numpy as np

class Model:
    def __init__(self, config):
        self.config = config
        self.clf = None

    def test(self, save):
        """
        Tests the supplied data.
        """
        frame = self.config.frame
        frame = frame[~frame['in_train']] # only consider training images
        X_test, y_test = frame.drop(['style', 'in_train'], axis=1), np.array(frame['style'].tolist())
        
        y_pred = self.est.predict(X_test)

        # list of strings to output to metrics file
        metrics = []        

        # calc metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')

        # replace numerical classes with string classes
        #frame['style'].replace(self.config.mapping, inplace=True)

        # create confusion matrix
        cm = confusion_matrix(y_test.tolist(), y_pred)
        print(cm, '\n'); metrics.append(str(cm))

        acc_out    = 'accuracy       : %s' % accuracy; metrics.append(acc_out)
        f1_out     = 'f1 score       : %s' % f1; metrics.append(f1_out)
        recall_out = 'recall score   : %s' % recall; metrics.append(recall_out)
        prec_out   = 'precision score: %s' % precision; metrics.append(prec_out)
        print(acc_out)
        print(f1_out)
        print(recall_out)
        print(prec_out)

        # save metrics
        self.metrics = metrics
        if save:
            print('Saving performance metrics...')
            path = os.path.join(self.config.save_dir, 'metrics.txt')
            f = open(path, 'w')
            f.write('\n'.join(metrics))
            f.close()

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
            self.rfc = RandomForestClassifier(verbose=1)
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
                verbose=1
            )
            self.clf = self.rfc

    def train(self):
        """
        Train on the config data according to the initialized classifier
        `self.clf`. Assumes that all features should be included and that
        the class column is 'style'.
        """
        frame = self.config.frame
        frame = frame[frame['in_train']==True] # only consider training images
        X, y = frame.drop(['style', 'in_train'], axis=1), frame['style']
       
        print('Training on', len(X.index), 'images...'); sleep(1)
        self.clf.fit(X, y)

        print()
        print('Training complete!')
        if type(self.clf) is GridSearchCV:
            print('\nBest parameters from GridSearchCV:')
            print(self.clf.best_params_)
            self.est = self.clf.best_estimator_
        else:
            self.est = self.rfc

        print('Best estimator:')
        print(self.est, '\n')

        print('Features importances:')
        imp = list(zip(X.columns, self.est.feature_importances_))
        imp = sorted(imp, reverse=True, key=lambda x: x[1])
        pprint(imp)
        print()

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
