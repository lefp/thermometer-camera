#!/bin/env python3

from argparse import ArgumentParser
from urllib.request import urlopen
import cv2 as cv
import numpy as np


MIN_FRAME_INTERVAL_S = 1

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('url')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    while True:
        # fetch and decode image
        im = bytearray(urlopen(args.url).read())
        im = cv.imdecode(np.asarray(im), cv.IMREAD_COLOR)

        # display
        cv.imshow('frame', im)
        key = cv.waitKey(MIN_FRAME_INTERVAL_S * 1000) # ms
        if key != -1 and chr(key) == 'q': exit(0)