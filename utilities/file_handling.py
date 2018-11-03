"""
File: file_handling.py
Author(s): Kayvon Khosrowpour

Description:
Provides functions for handling files, like reading csvs.
"""

import os, csv

def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

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

def recursive_get_imgs_from_dir(root, sort=True):
    """
    This method searches through a root directory, finds all jpg and png images
    located inside it, and its subdirectories, and returns their filepaths
    as a sorted list.

    Args:
      image_dir: the filepath to a directory of images
      sort (optional): True if lists should be sorted, false otherwise

    Returns:
      path_list: a list of filepaths of valid images located inside the
        image_dir
      names_list: a list of filenames of valid images located inside the
        image_dir (not full filepaths)
    """
    path_list, name_list = get_imgs_from_dir(root)
    for subdir in get_immediate_subdirectories(root):
        sub_pl, sub_nl = get_imgs_from_dir(subdir)
        path_list += sub_pl
        name_list += sub_nl
    if sort:
        path_list = sorted(path_list)
        name_list = sorted(name_list)
    return path_list, name_list

def get_imgs_from_dir(image_dir, sort=True):
    """
    This method searches through a directory, finds all jpg and png images
    located inside it, and returns their filepaths as a sorted list.

    Args:
      image_dir: the filepath to a directory of images
      sort (optional): True if lists should be sorted, false otherwise

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
    if sort:
        path_list = sorted(path_list)
        name_list = sorted(name_list)
    return path_list, name_list

