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
import itertools

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('url')
    return parser.parse_args()

"""Uses ORB to find matching points between the input `im` and reference image `ref`.
Expects inputs to be 3-channel integer images in [0, 255].
Returns (im_points, ref_points) where
    im_points[i, :] is a point in the input image matching ref_points[i, :]
"""
def get_matching_points(im, ref): # -> (np.ndarray, np.ndarray)
    # clone so we don't accidentally mutate the input
    im  = im .copy()
    ref = ref.copy()

    # perform edge detection on both images first; keypoint matching has been much more accurate this way
    ref = sk.filters.gaussian(ref, sigma=0.5, channel_axis=2)
    im  = sk.filters.gaussian(im , sigma=0.5, channel_axis=2)
    ref = rgb2gray(ref)
    im  = rgb2gray(im )
    ref = cv.Canny((255*ref).astype(np.uint8), 150, 200)
    im  = cv.Canny((255*im ).astype(np.uint8), 150, 200)

    descriptor_extractor = ORB(n_keypoints=300)

    # get keypoints and descriptors
    descriptor_extractor.detect_and_extract(ref)
    keypoints_ref   = descriptor_extractor.keypoints
    descriptors_ref = descriptor_extractor.descriptors
    #
    descriptor_extractor.detect_and_extract(im )
    keypoints_im    = descriptor_extractor.keypoints
    descriptors_im  = descriptor_extractor.descriptors

    # get indices of keypoints with matching descriptors
    matches = match_descriptors(descriptors_im, descriptors_ref, cross_check=True)

    # use the indices to get the points
    im_points  = keypoints_im [matches[:, 0], :].round().astype(np.int32)
    ref_points = keypoints_ref[matches[:, 1], :].round().astype(np.int32)

    return (im_points, ref_points)

if __name__ == '__main__':
    args = parse_args()
    
    orig_ref = sk.io.imread('thermometer_reference.jpg')
    ref_mask = sk.io.imread('thermometer_reference_mask.png', as_gray=True)
    assert(np.unique(ref_mask).tolist() == [0, 1]) # had issues with the mask not actually being binary
    ref_mask = ref_mask.astype(bool)

    while True:
        # fetch and decode image
        orig_im = sk.io.imread(args.url)

        im_points, ref_points = get_matching_points(orig_im, orig_ref)
        n_matches = np.shape(im_points)[0]
        print(f"{n_matches} matched keypoints")

        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax = list(itertools.chain.from_iterable(ax)) # flatten 2D array of axes
        for a in ax: a.axis('off')
        plt.gray()

        plot_matches(ax[0], orig_im, orig_ref, im_points, ref_points,
            np.repeat(np.arange(n_matches)[:, np.newaxis], repeats=2, axis=1) # hack to plot all points
        )

        # find the translation from the reference to the image
        # @todo this assumes no rotation, shear, etc! a more robust solution would compute the homography from
        # the src to matched keypoints; we'd also need a different method of discarding outliers
        translations = im_points - ref_points
        # assume the median translation is correct
        # @todo didn't think hard when choosing the median; but most translations are within a pixel of it
        translation = np.median(translations, axis=0).astype(np.int32)
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
        # crop
        im_thermo = orig_im[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]

        # im_thermo=sk.filters.gaussian(im_thermo, sigma=0.5, channel_axis=2)
        im_thermo = (rgb2gray(im_thermo)*255).astype(np.uint8) # rgb2gray returns floats
        im_thermo[~ref_mask] = 0 # set the background to black
        ax[2].imshow(im_thermo)
        thresh, im_thermo = cv.threshold(
            im_thermo, thresh=None, maxval=255, type=cv.THRESH_BINARY|cv.THRESH_OTSU
        )
        print(f"Otsu threshold: {thresh}")

        # get rid of black specks
        im_thermo = sk.morphology.remove_small_holes(im_thermo.astype(bool), area_threshold=10)

        # get the numbers from the image
        im = sk.color.gray2rgb(im_thermo)
        ax[3].imshow(im_thermo)
        print(tess.image_to_data(im_thermo, lang='eng', config='--psm 12 --oem 1'))

        plt.show()
