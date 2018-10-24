"""
File: main.py
Author(s): Kayvon Khosrowpour
Date created: 10/23/18

Description:
Main driver of the attribute extractor script. Loads a config from provided arg
and according to the config, extracts attributes and stores them for each img
in a csv.
"""

import argparse
import configparser
from attr.attr_extr import AttributeExtractor

class Config:
    def __init__(self):
        # load args from cmd line
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', type=str, help='supply path/to/CONFIG.ini')

        self.args = vars(parser.parse_args())
        config_file = self.args['c']
        if config_file is None:
            parser.error('No config file specified.')

        # save configs to self
        parser = configparser.ConfigParser()
        parser.read(config_file)
        self.data_dir = parser['DIR']['data_dir']
        self.results_dir = parser['DIR']['results_dir']
        # TODO: add more here

def main():
    # load configuration
    config = Config()

    # create extractor and extract data from imgs
    extr = AttributeExtractor(config.data_dir, config.results_dir)
    extr.extract()

    # save data
    extr.save()

    # done
    print('Results saved to', config.results_dir, '\n')

if __name__ == '__main__':
    main()