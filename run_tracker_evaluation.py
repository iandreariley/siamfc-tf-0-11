from __future__ import division
import sys
import os
import numpy as np
import pickle
import collections
import logging
from evaluation import TrackingResults
from evaluation import BboxFormats
from PIL import Image
import src.siamese as siam
from src.tracker import Tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox


def main():
    # set log level
    logging.basicConfig(level=logging.INFO)
    # avoid printing TF debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # TODO: allow parameters from command line or leave everything in json files?
    hp, evaluation, run, env, design = parse_arguments()
    # Set size for use with tf.image.resize_images with align_corners=True.
    # For example,
    #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
    # instead of
    # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    # build TF graph once for all
    detector = siam.SiameseNetwork(hp, design, env)

    # iterate through all videos of evaluation.dataset
    if evaluation.video == 'all':
        dataset_folder = os.path.join(env.root_dataset, evaluation.dataset)
        videos_list = [v for v in os.listdir(dataset_folder)]
        videos_list.sort()
        nv = np.size(videos_list)
        print nv
        speed = np.zeros(nv)
        precisions = np.zeros(nv)
        precisions_auc = np.zeros(nv)
        ious = np.zeros(nv)
        lengths = np.zeros(nv)
        results_dir = 'all_results_w_image_load'
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        for i in range(nv):
            logging.info("Starting evaluation of video {0}.".format(videos_list[i]))
            if not os.path.isdir(os.path.join(env.root_dataset, evaluation.dataset, videos_list[i])):
                logging.info("Path {0} is not a directory. Ignoring.".format(videos_list[i]))
                continue

            results_file = os.path.join(results_dir, videos_list[i] + "_results.p")
            if os.path.exists(results_file):
                logging.info("Results exist for video {0}. Skipping.".format(videos_list[i]))
                continue

            try:
                gt, frame_name_list, frame_sz, n_frames = _init_video(env, evaluation, videos_list[i])
                bbox = region_to_bbox(gt[0], center=False)
                tracker = Tracker(hp, run, design, frame_name_list, bbox, detector)
                bboxes, speed[i] = tracker.track()
                lengths[i], precisions[i], precisions_auc[i], ious[i], gt_bboxes = _compile_results(gt, bboxes, evaluation.dist_threshold)
                pred = collections.OrderedDict(zip(frame_name_list, bboxes))
                res = TrackingResults(pred, bbox, lengths[i] / speed[i], gt_bboxes, BboxFormats.CCWH)
                res.save(results_file)
                print str(i) + ' -- ' + videos_list[i] + \
                ' -- Precision: ' + "%.2f" % precisions[i] + \
                ' -- Precisions AUC: ' + "%.2f" % precisions_auc[i] + \
                ' -- IOU: ' + "%.2f" % ious[i] + \
                ' -- Speed: ' + "%.2f" % speed[i] + ' --'
                print
            except Exception as e:
                logging.warn("Tracking of video {0} threw the following exception: {1}".format(videos_list[i], e))

        tot_frames = np.sum(lengths)
        mean_precision = np.sum(precisions * lengths) / tot_frames
        mean_precision_auc = np.sum(precisions_auc * lengths) / tot_frames
        mean_iou = np.sum(ious * lengths) / tot_frames
        mean_speed = np.sum(speed * lengths) / tot_frames
        print '-- Overall stats (averaged per frame) on ' + str(nv) + ' videos (' + str(tot_frames) + ' frames) --'
        print ' -- Precision ' + "(%d px)" % evaluation.dist_threshold + ': ' + "%.2f" % mean_precision +\
              ' -- Precisions AUC: ' + "%.2f" % mean_precision_auc +\
              ' -- IOU: ' + "%.2f" % mean_iou +\
              ' -- Speed: ' + "%.2f" % mean_speed + ' --'
        print

    else:
        gt, frame_name_list, _, _ = _init_video(env, evaluation, evaluation.video)
        bbox = region_to_bbox(gt[evaluation.start_frame], center=False)
        tracker = Tracker(hp, run, design, frame_name_list, bbox, detector)
        bboxes, speed = tracker.track()
        _, precision, precision_auc, iou, _ = _compile_results(gt, bboxes, evaluation.dist_threshold)
        print evaluation.video + \
              ' -- Precision ' + "(%d px)" % evaluation.dist_threshold + ': ' + "%.2f" % precision +\
              ' -- Precision AUC: ' + "%.2f" % precision_auc + \
              ' -- IOU: ' + "%.2f" % iou + \
              ' -- Speed: ' + "%.2f" % speed + ' --'
        print


def _compile_results(gt, bboxes, dist_threshold):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = sum(new_distances < dist_threshold)/np.size(new_distances) * 100

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds+1)
    thresholds = thresholds[-n_thresholds:]
    # reverse it so that higher values of precision goes at the beginning
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = sum(new_distances < thresholds[i])/np.size(new_distances)

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou, gt4


def _init_video(env, evaluation, video):
    video_folder = os.path.join(env.root_dataset, evaluation.dataset, video)
    frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
    frame_name_list = [os.path.join(env.root_dataset, evaluation.dataset, video, '') + s for s in frame_name_list]
    frame_name_list.sort()
    with Image.open(frame_name_list[0]) as img:
        frame_sz = np.asarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth
    gt_file = os.path.join(video_folder, 'groundtruth.txt')
    gt = np.genfromtxt(gt_file, delimiter=',')
    n_frames = len(frame_name_list)
    assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'

    return gt, frame_name_list, frame_sz, n_frames


def _compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou


if __name__ == '__main__':
    sys.exit(main())
