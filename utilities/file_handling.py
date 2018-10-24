"""
File: file_handling.py
Author(s): Kayvon Khosrowpour

Description:
Provides functions for handling files, like reading csvs.
"""

import os, csv

def get_csv_files_from_dir(csv_directory):
    """
    This method gets all of the csv files in csv_directory, and
    returns their filepaths and names as a sorted list.

    Args:
      csv_directory: the filepath to a directory of csv files

    Returns:
      csv_paths: a list of filepaths of CSVs
      csv_names: a list of the names of the CSVs
    """
    csv_paths = []
    csv_names = []
    for item in os.scandir(csv_directory):
        if item.name.endswith('.csv') or item.name.endswith('.CSV'):
            csv_paths.append(item.path)
            csv_names.append(item.name)

    csv_paths = sorted(csv_paths)
    csv_names = sorted(csv_names)
    return csv_paths, csv_names

def get_images_from_dir(image_dir):
    """
    This method searches through a directory, finds all jpg and png images
    located inside it, and returns their filepaths as a sorted list.

    Args:
      image_dir: the filepath to a directory of images

    Returns:
      path_list: a list of filepaths of valid images located inside the
        image_dir
      names_list: a list of filenames of valid images located inside the
        image_dir (not full filepaths)
    """
    path_list = []
    name_list = []
    for item in os.scandir(image_dir):
        if item.name.endswith('.JPEG') or item.name.endswith('.png') \
                or item.name.endswith('.jpg') or item.name.endswith('.jpeg'):
            path_list.append(item.path)
            name_list.append(item.name)

    path_list = sorted(path_list)
    name_list = sorted(name_list)
    return path_list, name_list

