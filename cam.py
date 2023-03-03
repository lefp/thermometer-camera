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
    
    ref = sk.io.imread('thermometer_reference.jpg')
    orig_ref = ref

    ref = sk.filters.gaussian(ref, sigma=0.5, channel_axis=2)
    ref = rgb2gray(ref)
    ref = cv.Canny((255*ref).astype(np.uint8), 150, 200)

    while True:
        # fetch and decode image
        # im_input = bytearray(urlopen(args.url).read())
        # im_input = cv.imdecode(np.asarray(im_input), cv.IMREAD_COLOR)
        im = sk.io.imread(args.url)
        orig_im = im

        im = sk.filters.gaussian(im, sigma=0.5, channel_axis=2)
        im = rgb2gray(im)
        im = cv.Canny((255*im).astype(np.uint8), 150, 200)

        descriptor_extractor = ORB(n_keypoints=300)

        descriptor_extractor.detect_and_extract(ref)
        keypoints_ref   = descriptor_extractor.keypoints
        descriptors_ref = descriptor_extractor.descriptors
        #
        descriptor_extractor.detect_and_extract(im)
        keypoints_im   = descriptor_extractor.keypoints
        descriptors_im = descriptor_extractor.descriptors

        matches = match_descriptors(descriptors_im, descriptors_ref, cross_check=True)

        # src_descriptors = descriptors[matches[:, 0]]
        # print(keypoints_to_match); print(np.shape(keypoints_to_match));
        # print(np.shape(matches)); exit(0)
        n_matches = matches.shape[0]
        im_points  = keypoints_im [matches[:, 0], :].round().astype(np.int32)
        ref_points = keypoints_ref[matches[:, 1], :].round().astype(np.int32)
        print(f"{n_matches} matched keypoints")
        # for i in range(n_matches):
        #     orig_ref[sk.draw.circle_perimeter(ref_points[i,0], ref_points[i, 1], 4)] = (255, 0, 0)
        #     orig_im [sk.draw.circle_perimeter(im_points [i,0], im_points [i, 1], 4)] = (255, 0, 0)

        fig, ax = plt.subplots(nrows=2, ncols=1)
        for a in ax: a.axis('off')
        plt.gray()

        plot_matches(ax[0], orig_im, orig_ref, keypoints_im, keypoints_ref, matches)

        # find the translation from the reference to the image
        # @todo this assumes no rotation, shear, etc! a more robust solution would compute the homography from
        # the src to matched keypoints; we'd also need a different method of discarding outliers
        translations = im_points - ref_points
        # assume the median translation is correct
        # @todo didn't think hard when choosing the median; but most translations are within a pixel of it
        translation = np.median(translations, axis=0)
        # discard matches whose translation is too far from the "correct" one
        # @todo this step isn't actually required; we don't use the result
        translations = translations[
            np.linalg.norm(translations - translation, axis=1) < 10,
            :
        ]
        # isolate the thermometer in the input image
        top_left = translation
        bottom_right = np.shape(orig_ref)[:2] + translation
        im_rect = orig_im
        im_rect[sk.draw.rectangle_perimeter(top_left, bottom_right)] = (255, 0, 0)
        ax[1].imshow(im_rect)

        plt.show()

        # @todo next step: get the numbers off the thermometer
