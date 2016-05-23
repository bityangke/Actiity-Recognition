"""

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                              Common Module                               |
//|                               Version 1.0                                |
//|                                                                          |
//|              Copyright 2015-2020, Marcos Vinicius Teixeira               |
//|                          All Rights Reserved.                            |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
OVERVIEW: feature_extraction.py
//  ================================
//  This module implement methods that are common to another modules, by exam-
//  the feature_extraction.py.
//
"""

from os.path import exists, isdir, basename, isfile, join, splitext
import sift
from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like, fromstring, asarray, array
import numpy as np
import scipy.cluster.vq as vq
import os
from os.path import splitext, exists

EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
PRE_ALLOCATION_BUFFER = 1000  # for sift

# dict for UCF classes
targets_ucf = {}

class Common:

    @staticmethod
    def process_ucf_dataset(datasetpath):
        all_classes = []
        all_videos = []

        maxvideos_infolder = 5

        # Getting each class name
        for cl in glob(datasetpath + "/*"):
            all_classes.extend([join(datasetpath, basename(cl))])

        count_vid = 0
        # Getting each video name
        for i in range(len(all_classes)):
            for vid in glob(all_classes[i] + "/*"):

                if count_vid >= maxvideos_infolder:
                    break

                all_videos.extend([join(all_classes[i], basename(vid))])
                count_vid += 1
            count_vid = 0

        # Getting the frames for each video, using the script 'process_video.py'
        for vid in all_videos:
            cl = vid.split("/")[-2]
            # Taking the frames ..
            cmnd = "python process_video.py " + cl + ' ' + vid
            os.system(cmnd)

    @staticmethod
    def generate_ucf_50_dataset(datasetpath):
        histograms_path = "histograms"
        all_classes = []

        # Getting each class name
        for cl in glob(datasetpath + "/*"):
            all_classes.extend([join(datasetpath, basename(cl))])

        # Extract the histograms (if necessary ..)
        if not exists(histograms_path):
            path = "frames"

            all_videos = []

            # Getting each video name
            for i in xrange(len(all_classes)):
                for vid in glob(all_classes[i] + "/*"):
                    all_videos.extend([join(all_classes[i], basename(vid))])

            # Now, extracting the features for each video
            for idx, vid in enumerate(all_videos):
                print("Analysing video ", vid)
                cmnd = "python FeatureExtractor.py dataset " + vid
                os.system(cmnd)

        X = []
        Y = []

        # fitting target vector
        for i in range(len(all_classes)):
            targets_ucf[basename(all_classes[i])] = i

        trainset_path = "histograms"

        # getting the histograms names
        hist_files = []
        for tname in glob(trainset_path + "/*"):
            hist_files.extend([join(trainset_path, basename(tname))])

        # window of frames for each video
        video_frame_windows = []

        # Now we'll open each histogram and fill the X and Y vector. The line will be
        # the X value and the Y the value in the name position at targets_ucf vector.
        for hist_f in hist_files:
            histofile = open(hist_f)
            lines = histofile.readlines()

            # getting the video category
            label = targets_ucf[basename(hist_f.split("0")[0])]

            video_frame_windows.append(len(lines))
            for i in range(len(lines)):
                X.append(lines[i])
                Y.append(label)

            histofile.close()

        # Passing the vector X to numpy array
        for i in range(len(X)):
            splited = X[i].split(" ")
            splited_float = []
            for index in range(len(splited) - 1):
                splited_float.append(splited[index])
            X[i] = np.asarray(splited_float, dtype='float')

        return X, Y, video_frame_windows

    @staticmethod
    def generate_ucf_dataset(datasetpath):
        histograms_path = "histograms"
        all_classes = []

        # Getting each class name
        for cl in glob(datasetpath + "/*"):
            all_classes.extend([join(datasetpath, basename(cl))])

        # Extract the histograms (if necessary ..)
        if not exists(histograms_path):
            path = "frames"

            all_videos = []

            # Getting each video name
            # for i in range(len(all_classes)):
            for i in xrange(10):
                for vid in glob(all_classes[i] + "/*"):
                    all_videos.extend([join(all_classes[i], basename(vid))])

                    # Now, extracting the features for each video
            for idx, vid in enumerate(all_videos):
                print("Analysing video ", idx)
                cmnd = "python FeatureExtractor.py dataset " + vid
                os.system(cmnd)

        X = []
        Y = []

        # fitting target vector
        for i in range(len(all_classes)):
            targets_ucf[basename(all_classes[i])] = i

        trainset_path = "histograms"

        # getting the histograms names
        hist_files = []
        for tname in glob(trainset_path + "/*"):
            hist_files.extend([join(trainset_path, basename(tname))])

        # Now we'll open each histogram and fill the X and Y vector. The line will be
        # the X value and the Y the value in the name position at targets_ucf vector.
        for hist_f in hist_files:
            histofile = open(hist_f)
            lines = histofile.readlines()

            label = targets_ucf[basename(hist_f.split("_")[1])]

            for i in range(len(lines)):
                X.append(lines[i])
                Y.append(label)

            histofile.close()

        # Passing the vector X to numpy array
        for i in range(len(X)):
            splited = X[i].split(" ")
            splited_float = []
            for index in range(len(splited) - 1):
                splited_float.append(splited[index])
            X[i] = np.asarray(splited_float, dtype='float')

        return X, Y


    # extracting the class names given a folder name (dataset)
    @staticmethod
    def get_classes(datasetpath):
        cat_paths = [files
                     for files in glob(datasetpath + "/*")
                     if isdir(files)]

        cat_paths.sort()
        cats = [basename(cat_path) for cat_path in cat_paths]

        return cats


    # getting the array of files(images) inside a given folder
    @staticmethod
    def get_imgfiles(path):
        all_files = []

        all_files.extend([join(path, basename(fname))
                          for fname in glob(path + "/*")
                          if splitext(fname)[-1].lower() in EXTENSIONS])
        return all_files


    # calculate the sift descriptor for each image of a input array. The output
    # is saved with the same name of image file plus '.sift'
    @staticmethod
    def extractSift(input_files):
        print "extracting Sift features"
        all_features_dict = {}

        for i, fname in enumerate(input_files):
            features_fname = fname + '.sift'

            if exists(features_fname) == False:
                # print "calculating sift features for", fname
                sift.process_image(fname, features_fname)

            # print "gathering sift features for", fname,
            locs, descriptors = sift.read_features_from_file(features_fname)

            # check if there is description for the image
            if len(descriptors) > 0:
                print descriptors.shape
                all_features_dict[fname] = descriptors

        return all_features_dict


    # Transforming a dict in a numpy array
    # ...
    @staticmethod
    def dict2numpy(dict):
        nkeys = len(dict)
        array = zeros((nkeys * PRE_ALLOCATION_BUFFER, 128))
        pivot = 0

        for key in dict.keys():
            value = dict[key]
            nelements = value.shape[0]
            while pivot + nelements > array.shape[0]:
                padding = zeros_like(array)
                array = vstack((array, padding))
            array[pivot:pivot + nelements] = value
            pivot += nelements
        array = resize(array, (pivot, 128))
        return array


    # calculating histograms given a codebook that represents the vocabulary and
    # the array of descriptors, generated by each image
    @staticmethod
    def computeHistograms(codebook, descriptors):
        code, dist = vq.vq(descriptors, codebook)
        histogram_of_words, bin_edges = histogram(code,
                                                  bins=range(codebook.shape[0] + 1),
                                                  normed=True)
        return histogram_of_words


    # writing the histograms into the file
    @staticmethod
    def writeHistogramsToFile(nwords, fnames, all_word_histgrams, features_fname):
        data_rows = zeros(nwords + 1)  # +1 for the category label

        for fname in fnames:
            histogram = all_word_histgrams[fname]

            if (histogram.shape[0] != nwords):  # scipy deletes empty clusters
                nwords = histogram.shape[0]
                data_rows = zeros(nwords + 1)

            data_row = hstack((0, histogram))
            data_rows = vstack((data_rows, data_row))

        data_rows = data_rows[1:]

        savetxt(features_fname, data_rows)

    @staticmethod
    def writeHashMatrixToFile(filename, hashMatrix):
        savetxt(filename, hashMatrix)


    # passing the codebook of string to numpy array
    @staticmethod
    def stringToNumpy(codebook_file):
        codebook = []

        lines = codebook_file.readlines()

        for line in lines:
            line_array = fromstring(line, dtype=float, sep=' ')
            codebook.append(line_array)

        return asarray(codebook)