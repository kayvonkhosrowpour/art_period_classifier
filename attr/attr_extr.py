"""
Filename: attr_extr.py
Author(s): Kayvon Khosrowpour
Date Created: 10/25/18

Description:
Contains an object AttributeExtractor for collecting image attributes.
"""

import pandas as pd

class AttributeExtractor:
    def __init__(self, data_dir, results_dir):
        self.data_dir = data_dir
        self.results_dir = results_dir
        # TODO: make pandas dataframe

    def extract(self):
        print('Extracting imgs...')
        pass

    def save(self):
        print('Saving results...')
        pass
