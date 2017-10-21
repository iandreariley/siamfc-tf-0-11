import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2


def show_frame(frame, bbox, fig_n):
    rev_frame = np.copy(frame[...,::-1]) # copy necessary to keep cv2 from throwing a fit.
    draw_bbox(rev_frame, bbox, (0, 0, 255))
    cv2.imshow('main', rev_frame)
    cv2.waitKey(1)


def draw_bbox(image, bbox, color):
    top_left = intify((bbox[0], bbox[1]))
    bottom_right = intify((bbox[0] + bbox[2], bbox[1] + bbox[3]))
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    # draw bounding box.
    cv2.line(image, top_left, top_right, color, 2)
    cv2.line(image, top_right, bottom_right, color, 2)
    cv2.line(image, bottom_right, bottom_left, color, 2)
    cv2.line(image, bottom_left, top_left, color, 2)


def intify(tup):
    return tuple(map(int, tup))


def show_crops(crops, fig_n):
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(np.uint8(crops[0,:,:,:]))
    ax2.imshow(np.uint8(crops[1,:,:,:]))
    ax3.imshow(np.uint8(crops[2,:,:,:]))
    plt.ion()
    plt.show()
    plt.pause(0.001)


def show_scores(scores, fig_n):
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(scores[0,:,:], interpolation='none', cmap='hot')
    ax2.imshow(scores[1,:,:], interpolation='none', cmap='hot')
    ax3.imshow(scores[2,:,:], interpolation='none', cmap='hot')
    plt.ion()
    plt.show()
    plt.pause(0.001)
