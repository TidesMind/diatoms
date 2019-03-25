""" Derek Thompson
    University of Washington
    Diatom project
    This module does things an undergrad does like count and label data :)
"""

import os
import fnmatch
import csv
import cv2

""" This method counts the number of images per each class
    in the path provided. Make sure that the base path is the
    path leads to a directory of folders where each folder
    is a class folder with images inside or behavior is undefined
    
    Arguments:
        base_path: a string representing the path to the base folder
        class_counts: a dictionary of already counted classes or initialized to empty dictionary
                      if none is provided
        
    Returns a Dictionary with class_name -> count
    
    Raises Exception if base_path is not a directory
"""


def get_class_counts(base_path, class_counts={}):
    if not os.path.isdir(base_path):
        raise Exception('base_path provided was not a directory')
    for path, dirs, files in os.walk(base_path):
        for directory in dirs:
            count = len(fnmatch.filter(os.listdir(path + '/' + directory), "*.png"))
            if directory in class_counts:
                class_counts[directory] += count
            else:
                class_counts[directory] = count

    return class_counts


""" This is a method for the diatom data set.
    This method counts the number of images per each class over
    all years. 
    
    Arguments:
        base_path: a string to the director with folders labeled by year
    
    Returns a Dictionary with class_name -> count
    
    Raises Exception if base_path is not a directory
"""


def get_class_counts_all_years(base_path):
    if not os.path.isdir(base_path):
        raise Exception('base_path provided was not a directory')
    class_counts = {}
    for path, dirs, files in os.walk(base_path):
        for directory in dirs:
            class_counts = get_class_counts(path + '/' + directory, class_counts)

    return class_counts


def get_class_image_paths(base_path, class_image_paths={}):
    if not os.path.isdir(base_path):
        raise Exception('base_path provided was not a directory')
    for path, dirs, files in os.walk(base_path):
        for directory in dirs:
            for img in fnmatch.filter(os.listdir(path + '/' + directory), "*.png"):
                if directory in class_image_paths:
                    class_image_paths[directory].append(path + '/' + directory + '/' + img)
                else:
                    class_image_paths[directory] = []
                    class_image_paths[directory].append(path + '/' + directory + '/' + img)

    return class_image_paths


def get_image_paths_all_years(base_path):
    if not os.path.isdir(base_path):
        raise Exception('base_path provided was not a directory')
    class_image_paths = {}
    for path, dirs, files in os.walk(base_path):
        for directory in dirs:
            class_image_paths = get_class_image_paths(path + '/' + directory, class_image_paths)
    
    return class_image_paths


""" This method is for making the csv files for the dataset class to read in training and testing models.
    This method will use a 5 / 1 ratio for training / testing data.
    Currently client is responsible for deleting test and training csv files created.
    The csv files created will be in the following format:
        image_path  label_integer
    
    Arguments:
        train_csv_name: name of training data csv file to be created without .csv suffix
        test_csv_name: name of testing data scv file to be created without .csv suffix
"""


def make_csv_files(train_csv_path, test_csv_path, label_csv_path, diatom_dataset_path, ignored_classes=[]):
    if os.path.isdir(train_csv_path) or os.path.isfile(train_csv_path):
        raise FileExistsError('training csv file name can not already exist as a file or directory')
    if os.path.isdir(test_csv_path) or os.path.isfile(test_csv_path):
        raise FileExistsError('testing csv file name can not already exist as a file or directory')
    if os.path.isdir(label_csv_path) or os.path.isfile(label_csv_path):
        raise FileExistsError('label csv file name can not already exist as a file or directory')

    class_image_paths = get_image_paths_all_years(diatom_dataset_path)

    class_labels = {}
    with open(label_csv_path + '.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        i = 0
        for class_name in class_image_paths:
            if class_name not in ignored_classes:
                class_labels[class_name] = i
                i += 1
                filewriter.writerow([i, class_name])

    with open(train_csv_path + '.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        for class_name in class_image_paths:
            if class_name not in ignored_classes:
                for img in class_image_paths[class_name][:int(len(class_image_paths[class_name]) * .8)]:
                    filewriter.writerow([img, class_labels[class_name]])

    with open(test_csv_path + '.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        for class_name in class_image_paths:
            if class_name not in ignored_classes:
                for img in class_image_paths[class_name][int(len(class_image_paths[class_name]) * .8):]:
                    filewriter.writerow([img, class_labels[class_name]])


def square_images(imgs_dir, new_imgs_dir, desired_size, img_file_type):
    if not os.path.isdir(imgs_dir):
        raise FileNotFoundError("imgs_dir must be an existing directory")
    if not os.path.isdir(new_imgs_dir):
        raise FileNotFoundError("new_img_dir must be an existing directory")
    if desired_size < 1:
        raise Exception("size must be greater than 0")

    for img_path in fnmatch.filter(os.listdir(imgs_dir), img_file_type):
        im = cv2.imread(imgs_dir + '/' + img_path)
        old_size = im.shape[:2]
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]

        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        cv2.imwrite(new_imgs_dir + '/' + img_path, new_im)



# ignore = ['mix_elongated', 'bad', 'mix', 'other_interaction', 'bead']
# make_csv_files('/media/tides/SSD1/diatom_proj/diatoms/data/csv_files/t_train',
#                '/media/tides/SSD1/diatom_proj/diatoms/data/csv_files/t_test',
#                '/media/tides/SSD1/diatom_proj/diatoms/data/csv_files/t_labels',
#                '/media/tides/SSD1/diatom_proj/diatom_data', ignore)
