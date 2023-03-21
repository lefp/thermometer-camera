#!/bin/env python3

from argparse import ArgumentParser
from urllib.request import urlopen
import cv2 as cv
import numpy as np
import skimage as skim
import sklearn
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, corner_harris, corner_peaks, ORB, plot_matches
import scipy
import toml
import matplotlib.pyplot as plt
import itertools
from math import sin, cos, pi

REF_DATA_DIR = "reference-data"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('url')
    return parser.parse_args()

"""
Parses the toml file containing the thermometer's reference points;
Returns a function `temperature(pointer_direction: [float, float]) -> float64` which:
    Takes as input any vector parallel to the temperature pointer's direction (under skimage coordinates).
    Returns the temperature the pointer is indicating.
"""
def function_mapping_pointer_vector_to_temperature(reference_points_fname):
    ref_points = toml.load(reference_points_fname)
    pivot_pos = np.array(ref_points["pivot_position"])
    temperature_positions = ref_points["temperature_positions"]

    # TOML assumes all keys are strings, so we need to convert them to integers
    temperature_positions = {int(key): val for (key, val) in temperature_positions.items()}

    # for each sample temperature, compute its angle
    angle_to_temperature = dict()
    for (temperature, position) in temperature_positions.items():
        direction = np.array(position) - pivot_pos
        angle = np.arctan2(direction[1], direction[0])
        # `np.arctan2` returns is in [-pi, pi]; but we want the line of discontinuity to be from the pivot
        # downwards (in order for interpolation to work), so we remap to `[0, 2*pi]`
        if angle < 0: angle += 2*pi
        angle_to_temperature[angle] = temperature

    # convert the dict to arrays, the format expected by `np.interp`.
    # Note: `np.interp` expects the angles to be sorted!
    sample_angles = np.array(sorted(angle_to_temperature.keys()))
    sample_temperatures = np.array([angle_to_temperature[ang] for ang in sample_angles])

    # @debug
    print('(temperature, angle) sample points:')
    for (t, a) in zip(sample_temperatures, sample_angles): print((t, a))

    def temperature_from_pointer_vector(pointer_vector): # [float, float] -> float64
        print(f'pointer vector {pointer_vector}') # @debug
        angle = np.arctan2(pointer_vector[1], pointer_vector[0])
        # remap `np.arctan2` range from [-pi, pi] to [0, 2*pi]
        if angle < 0: angle += 2*pi
        print(f'pointer angle {angle}') # @debug
        return np.interp([angle], sample_angles, sample_temperatures)[0]

    return temperature_from_pointer_vector


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
    ref = skim.filters.gaussian(ref, sigma=0.5, channel_axis=2)
    im  = skim.filters.gaussian(im , sigma=0.5, channel_axis=2)
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

    # get reference data
    #
    # reference image of the thermometer
    orig_ref = skim.io.imread(REF_DATA_DIR + '/thermometer.jpg')
    # mask of the thermometer in the image
    ref_mask = skim.io.imread(REF_DATA_DIR + '/thermometer_mask.png', as_gray=True)
    assert(set(np.unique(ref_mask)) == {0, 1}) # sanity check; have exported the mask incorrectly in the past
    ref_mask = ref_mask.astype(bool)
    # position of the red temperature pointer's pivot
    pivot_pos = toml.load(REF_DATA_DIR + '/positions.toml')["pivot_position"]

    # this gives us a function
    temperature_from_pointer_vector = function_mapping_pointer_vector_to_temperature(
        REF_DATA_DIR + '/positions.toml'
    )

    # main loop
    while True:
        # fetch and decode image
        orig_im = skim.io.imread(args.url)

        im_points, ref_points = get_matching_points(orig_im, orig_ref)
        n_matches = np.shape(im_points)[0]
        print(f"{n_matches} matched keypoints")

        fig, ax = plt.subplots(nrows=3, ncols=2)
        plt.gray() # every 1-channel image should be displayed using the "gray" colormap
        ax = list(itertools.chain.from_iterable(ax)) # flatten 2D array of axes
        for a in ax: a.axis('off') # don't draw the axis lines
        ax = iter(ax) # from now, every attempt to add a new plot should call next(ax)

        plot_matches(
            next(ax), orig_im, orig_ref, im_points, ref_points,
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
        im_rect[skim.draw.rectangle_perimeter(top_left, bottom_right)] = (255, 0, 0)
        next(ax).imshow(im_rect)
        # crop
        im_thermo = orig_im[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]

        # isolate the pointer in the image
        #
        # convert to hsv
        im_thermo_hsv = skim.color.rgb2hsv(im_thermo)
        # next(ax).imshow(im_thermo_hsv)
        im_thermo_hue = im_thermo_hsv[:, :, 0]
        im_thermo_sat = im_thermo_hsv[:, :, 1]
        im_thermo_val = im_thermo_hsv[:, :, 2]
        # isolate red in image
        im_pointer = (
            ref_mask & # remove background
            ((0.9 < im_thermo_hue) | (im_thermo_hue < 0.1)) & # red hue
            (im_thermo_val > 0.3) & # filter out black
            (im_thermo_sat > 0.3) # filter out white
        )
        next(ax).imshow(im_pointer)
        # get rid of any holes or cracks in the pointer.
        # Note we don't use `remove_small_holes` because the cracks may be connected to the background.
        footprint = np.array([
            [0, 0, 0, 1, 0, 0, 0,],
            [0, 0, 1, 1, 1, 0, 0,],
            [0, 1, 1, 1, 1, 1, 0,],
            [1, 1, 1, 1, 1, 1, 1,],
            [0, 1, 1, 1, 1, 1, 0,],
            [0, 0, 1, 1, 1, 0, 0,],
            [0, 0, 0, 1, 0, 0, 0 ]
        ])
        skim.morphology.binary_dilation(im_pointer, footprint=footprint, out=im_pointer)
        skim.morphology.binary_erosion (im_pointer, footprint=footprint, out=im_pointer)
        # delete noise spots by only keeping the largest connected component (i.e. the pointer)
        # the "weights" part prevents the background from being counted
        labeled = skim.measure.label(im_pointer, background=0)
        largest_component_label = np.argmax(np.bincount(labeled.flat, weights=im_pointer.flat))
        im_pointer = labeled == largest_component_label

        # get the pointer's orientation
        #
        props = skim.measure.regionprops(im_pointer.astype(np.uint8))[0] # [0] because there's only one region
        # this direction is parallel to the pointer, but may be in the opposite direction
        direction = np.array([cos(props.orientation), sin(props.orientation)])
        # The pivot is offset from the pointer's centroid.
        # Starting at the pivot, if we travel half the total length of the pointer in the correct direction,
        # we will remain on the pointer. But if we travel backwards from the pivot by the same amount, we will
        # leave the pointer.
        # Therefore, iff we leave the pointer (i.e. reach the background) by such travel, then it's pointing
        # in the direction opposite to what we thought.
        half_pointer_len = 0.5 * props.axis_major_length
        endpoint = pivot_pos + half_pointer_len * direction
        if im_pointer[endpoint[0].astype(int), endpoint[1].astype(int)] == 0: direction = -direction

        a = next(ax)
        a.imshow(im_thermo)
        a.plot( # point order is reversed because matplotlib's coordinate system is opposite to skimage's
            (pivot_pos[1], pivot_pos[1] + half_pointer_len*direction[1]),
            (pivot_pos[0], pivot_pos[0] + half_pointer_len*direction[0]),
            '-',
            color='lime'
        )

        temperature = temperature_from_pointer_vector(direction)
        print(f'temperature: {temperature} degF')

        plt.show()
