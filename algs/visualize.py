"""
File: visualize.py
Author(s): Kayvon Khosrowpour
Date created: 11/26/18

Description:
Analyzes a CSV of data for training purposes. This script can
generate a correlation matrix and scatter matrix of the data.
This is extremely useful in determining if the features that
are provided by the CSV are useful and sufficient.

Example usage for both correlation matrix and scatter matrix:
    python3 visualize.py -csv /Users/kayvon/code/divp/proj/data/results/basic_test.csv -n basic --cm --sm
Example usage for correlation matrix only:
    python3 visualize.py -csv /Users/kayvon/code/divp/proj/data/results/basic_test.csv -n basic --cm
Example usage for scatter matrix only:
    python3 visualize.py -csv /Users/kayvon/code/divp/proj/data/results/basic_test.csv -n basic --sm
"""

import os, argparse
import pandas as pd
import seaborn as sns; sns.set(style='ticks', color_codes=True)
import matplotlib.pyplot as plt

class_title = 'style'
interested_cols = ['median_gray', 'median_hue', class_title]

def parse():
    """
    Parse the cmd line arguments to find csv location and determine what analysis to run.
    Returns:
        csv: path to the csv
        cm: True if correlation matrix should be run
        sm: True if the scatter matrix should be run
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv', type=str, help='supply path/to/data.csv', required=True)
    parser.add_argument('-n', type=str, help='name of the run', required=True)
    parser.add_argument('--cm', action='store_true', help='to run correlation matrix')
    parser.add_argument('--sm', action='store_true', help='to run scatter matrix')
    args = parser.parse_args()
    if args.cm is None and args.sm is None:
        parser.error('No analysis flag --cm or --sm')
    csv_path = os.path.normpath(args.csv)
    assert os.path.isfile(csv_path)

    return csv_path, args.cm, args.sm, args.n

def calc_corr_matrix(frame, name):
    print('Calculating correlation matrix...')
    frame = frame[interested_cols]
    corr = frame.drop(class_title, axis='columns').corr().round(2)
    sns.heatmap(corr, center=0, annot=True, annot_kws={'size': 3}, cmap='seismic')
    plt.savefig('visuals/heatmap_%s.png' % name, bbox_inches='tight',
        dpi=500)

def calc_scat_matrix(frame, name):
    print('Calculating scatter matrix...')
    frame = frame[interested_cols]
    sns.pairplot(frame, diag_kind='hist', hue=class_title,
        plot_kws=dict(alpha=0.25))
    plt.savefig('visuals/pairplot_%s.png' % name, bbox_inches='tight',
        dpi=500)

def get_frame(csv_path):
    frame = pd.read_csv(csv_path); assert (frame is not None)
    frame.drop(['filename', 'path'], axis='columns', inplace=True)
    return frame

def main():
    csv_path, cm, sm, n = parse()
    frame = get_frame(csv_path)
    if cm: calc_corr_matrix(frame, n)
    if sm: calc_scat_matrix(frame, n)

if __name__ == '__main__':
    main()