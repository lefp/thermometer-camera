#!/bin/env python3

from argparse import ArgumentParser
from urllib.request import urlopen
import cv2 as cv
import numpy as np
import skimage as sk
from skimage.color import rgb2gray
from skimage.feature import (match_descriptors, corner_harris, corner_peaks, ORB, plot_matches)
import matplotlib.pyplot as plt
import pytesseract as tess

MIN_FRAME_INTERVAL_S = 1

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('url')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    im_to_match = sk.io.imread('clock.jpg')
    im_to_match = sk.filters.gaussian(im_to_match, sigma=0.5, channel_axis=2)
    im_to_match = rgb2gray(im_to_match)
    im_to_match = cv.Canny((255*im_to_match).astype(np.uint8), 150, 200)

    while True:
        # fetch and decode image
        # im_input = bytearray(urlopen(args.url).read())
        # im_input = cv.imdecode(np.asarray(im_input), cv.IMREAD_COLOR)
        im_input = sk.io.imread(args.url)

        im = im_input
        im = sk.filters.gaussian(im, sigma=0.5, channel_axis=2)
        im = rgb2gray(im)
        im = cv.Canny((255*im).astype(np.uint8), 150, 200)

        descriptor_extractor = ORB(n_keypoints=300)

        descriptor_extractor.detect_and_extract(im_to_match)
        keypoints_to_match = descriptor_extractor.keypoints
        descriptors_to_match = descriptor_extractor.descriptors
        #
        descriptor_extractor.detect_and_extract(im)
        keypoints = descriptor_extractor.keypoints
        descriptors = descriptor_extractor.descriptors

        matches = match_descriptors(descriptors, descriptors_to_match, cross_check=True)

        fig, ax = plt.subplots(nrows=1, ncols=1)

        plt.gray()
        plot_matches(ax, im, im_to_match, keypoints, keypoints_to_match, matches)
        ax.axis('off')

        plt.show()

        # process image
        # im = im_input

        # # img1 = rgb2gray
        # # im = denoise_tv_chambolle(im)
        # # grayscale
        # # display
        # cv.imshow('input', im_input)

        # # wait for the frame interval duration, and handle keypresses
        # # key = cv.waitKey(MIN_FRAME_INTERVAL_S * 1000) # ms
        # key = cv.waitKey(1) # ms
        # if key != -1 and chr(key) == 'q': exit(0)