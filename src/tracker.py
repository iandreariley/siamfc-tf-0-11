
import tensorflow as tf
print('Using Tensorflow '+tf.__version__)
import matplotlib.pyplot as plt
import sys
# sys.path.append('../')
import os
import csv
import numpy as np
from PIL import Image
import collections
import time
import scipy.ndimage as ndimage

import src.siamese as siam
from src.visualization import show_frame, show_crops, show_scores


class Tracker:

    def __init__(self, hp, run, design, frame_name_list, bbox, image, templates_z, scores, start_frame=0):

        self.frame_name_list = frame_name_list[start_frame:]
        self.hp = hp
        self.run = run
        self.design = design
        self.image = image
        self.templates_z = templates_z
        self.scores = scores
        self.final_score_sz = hp.response_up * (design.score_sz - 1) + 1
        self.pos_x, self.pos_y, self.target_w, self.target_h = bbox

    def track(self):
        num_frames = np.size(self.frame_name_list)
        # stores tracker's output for evaluation
        bboxes = np.zeros((num_frames,4))

        scale_factors = self.hp.scale_step**np.linspace(-np.ceil(self.hp.scale_num/2), np.ceil(self.hp.scale_num/2), self.hp.scale_num)
        # cosine window to penalize large displacements
        hann_1d = np.expand_dims(np.hanning(self.final_score_sz), axis=0)
        penalty = np.transpose(hann_1d) * hann_1d
        penalty = penalty / np.sum(penalty)

        context = self.design.context*(self.target_w+self.target_h)
        z_sz = np.sqrt(np.prod((self.target_w+context)*(self.target_h+context)))
        x_sz = float(self.design.search_sz) / self.design.exemplar_sz * z_sz
        run_opts = {}

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # save first frame position (from ground-truth)
            bboxes[0,:] = self.pos_x - self.target_w/2, self.pos_y - self.target_h/2, self.target_w, self.target_h
            image_ = ndimage.imread(self.frame_name_list[0])
            templates_z_ = sess.run([self.templates_z], feed_dict={
                siam.pos_x_ph: self.pos_x,
                siam.pos_y_ph: self.pos_y,
                siam.z_sz_ph: z_sz,
                self.image: image_})

            t_start = time.time()

            # Get an image from the queue
            for i in range(1, num_frames):
                scaled_exemplar = z_sz * scale_factors
                scaled_search_area = x_sz * scale_factors
                scaled_target_w = self.target_w * scale_factors
                scaled_target_h = self.target_h * scale_factors
                image_ = ndimage.imread(self.frame_name_list[i])
                scores_ = sess.run(
                    [self.scores],
                    feed_dict={
                        siam.pos_x_ph: self.pos_x,
                        siam.pos_y_ph: self.pos_y,
                        siam.x_sz0_ph: scaled_search_area[0],
                        siam.x_sz1_ph: scaled_search_area[1],
                        siam.x_sz2_ph: scaled_search_area[2],
                        self.templates_z: np.squeeze(templates_z_),
                        self.image: image_,
                    }, **run_opts)
                scores_ = np.squeeze(scores_)
                # penalize change of scale
                scores_[0,:,:] = self.hp.scale_penalty*scores_[0,:,:]
                scores_[2,:,:] = self.hp.scale_penalty*scores_[2,:,:]
                # find scale with highest peak (after penalty)
                new_scale_id = np.argmax(np.amax(scores_, axis=(1,2)))
                # update scaled sizes
                x_sz = (1-self.hp.scale_lr)*x_sz + self.hp.scale_lr*scaled_search_area[new_scale_id]
                target_w = (1-self.hp.scale_lr)*self.target_w + self.hp.scale_lr*scaled_target_w[new_scale_id]
                target_h = (1-self.hp.scale_lr)*self.target_h + self.hp.scale_lr*scaled_target_h[new_scale_id]
                # select response with new_scale_id
                score_ = scores_[new_scale_id,:,:]
                score_ = score_ - np.min(score_)
                score_ = score_/np.sum(score_)
                # apply displacement penalty
                score_ = (1-self.hp.window_influence)*score_ + self.hp.window_influence*penalty
                self.pos_x, self.pos_y = _update_target_position(self.pos_x, self.pos_y, score_, self.final_score_sz, self.design.tot_stride, self.design.search_sz, self.hp.response_up, x_sz)
                # convert <cx,cy,w,h> to <x,y,w,h> and save output
                bboxes[i,:] = self.pos_x-target_w/2, self.pos_y-target_h/2, target_w, target_h
                # update the target representation with a rolling average
                if self.hp.z_lr>0:
                    new_templates_z_ = sess.run([self.templates_z], feed_dict={
                        siam.pos_x_ph: self.pos_x,
                        siam.pos_y_ph: self.pos_y,
                        siam.z_sz_ph: z_sz,
                        self.image: image_
                    })

                    templates_z_=(1-self.hp.z_lr)*np.asarray(templates_z_) + self.hp.z_lr*np.asarray(new_templates_z_)

                # update template patch size
                z_sz = (1-self.hp.scale_lr)*z_sz + self.hp.scale_lr*scaled_exemplar[new_scale_id]

                if self.run.visualization:
                    show_frame(image_, bboxes[i,:], 1)

            t_elapsed = time.time() - t_start
            speed = num_frames/t_elapsed

            # Finish off the filename queue coordinator.
            coord.request_stop()
            coord.join(threads)

        plt.close('all')

        return bboxes, speed





# read default parameters and override with custom ones
def tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz, image, templates_z, scores, start_frame):
    num_frames = np.size(frame_name_list)
    # stores tracker's output for evaluation
    bboxes = np.zeros((num_frames,4))

    scale_factors = hp.scale_step**np.linspace(-np.ceil(hp.scale_num/2), np.ceil(hp.scale_num/2), hp.scale_num)
    # cosine window to penalize large displacements
    hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
    penalty = np.transpose(hann_1d) * hann_1d
    penalty = penalty / np.sum(penalty)

    context = design.context*(target_w+target_h)
    z_sz = np.sqrt(np.prod((target_w+context)*(target_h+context)))
    x_sz = float(design.search_sz) / design.exemplar_sz * z_sz
    run_opts = {}

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # save first frame position (from ground-truth)
        bboxes[0,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
        image_ = ndimage.imread(frame_name_list[0])
        templates_z_ = sess.run([templates_z], feed_dict={
                                                                        siam.pos_x_ph: pos_x,
                                                                        siam.pos_y_ph: pos_y,
                                                                        siam.z_sz_ph: z_sz,
                                                                        image: image_})

        t_start = time.time()

        # Get an image from the queue
        for i in range(1, num_frames):
            scaled_exemplar = z_sz * scale_factors
            scaled_search_area = x_sz * scale_factors
            scaled_target_w = target_w * scale_factors
            scaled_target_h = target_h * scale_factors
            image_ = cv2.imread(frame_name_list[i])[:,:,::-1]
            scores_ = sess.run(
                [scores],
                feed_dict={
                    siam.pos_x_ph: pos_x,
                    siam.pos_y_ph: pos_y,
                    siam.x_sz0_ph: scaled_search_area[0],
                    siam.x_sz1_ph: scaled_search_area[1],
                    siam.x_sz2_ph: scaled_search_area[2],
                    templates_z: np.squeeze(templates_z_),
                    image: image_,
                }, **run_opts)
            scores_ = np.squeeze(scores_)
            # penalize change of scale
            scores_[0,:,:] = hp.scale_penalty*scores_[0,:,:]
            scores_[2,:,:] = hp.scale_penalty*scores_[2,:,:]
            # find scale with highest peak (after penalty)
            new_scale_id = np.argmax(np.amax(scores_, axis=(1,2)))
            # update scaled sizes
            x_sz = (1-hp.scale_lr)*x_sz + hp.scale_lr*scaled_search_area[new_scale_id]
            target_w = (1-hp.scale_lr)*target_w + hp.scale_lr*scaled_target_w[new_scale_id]
            target_h = (1-hp.scale_lr)*target_h + hp.scale_lr*scaled_target_h[new_scale_id]
            # select response with new_scale_id
            score_ = scores_[new_scale_id,:,:]
            score_ = score_ - np.min(score_)
            score_ = score_/np.sum(score_)
            # apply displacement penalty
            score_ = (1-hp.window_influence)*score_ + hp.window_influence*penalty
            pos_x, pos_y = _update_target_position(pos_x, pos_y, score_, final_score_sz, design.tot_stride, design.search_sz, hp.response_up, x_sz)
            # convert <cx,cy,w,h> to <x,y,w,h> and save output
            bboxes[i,:] = pos_x-target_w/2, pos_y-target_h/2, target_w, target_h
            # update the target representation with a rolling average
            if hp.z_lr>0:
                new_templates_z_ = sess.run([templates_z], feed_dict={
                                                                siam.pos_x_ph: pos_x,
                                                                siam.pos_y_ph: pos_y,
                                                                siam.z_sz_ph: z_sz,
                                                                image: image_
                                                                })

                templates_z_=(1-hp.z_lr)*np.asarray(templates_z_) + hp.z_lr*np.asarray(new_templates_z_)

            # update template patch size
            z_sz = (1-hp.scale_lr)*z_sz + hp.scale_lr*scaled_exemplar[new_scale_id]

            if run.visualization:
                show_frame(image_, bboxes[i,:], 1)

        t_elapsed = time.time() - t_start
        speed = num_frames/t_elapsed

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)

    plt.close('all')

    return bboxes, speed


def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y


