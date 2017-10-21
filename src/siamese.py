import tensorflow as tf
import numpy as np
import scipy.io
import sys
import os.path
from region_to_bbox import region_to_bbox
from src.convolutional import set_convolutional
from src.crops import extract_crops_z, extract_crops_x, pad_frame, resize_images
sys.path.append('../')

pos_x_ph = tf.placeholder(tf.float64)
pos_y_ph = tf.placeholder(tf.float64)
z_sz_ph = tf.placeholder(tf.float64)
x_sz0_ph = tf.placeholder(tf.float64)
x_sz1_ph = tf.placeholder(tf.float64)
x_sz2_ph = tf.placeholder(tf.float64)

# the follow parameters *have to* reflect the design of the network to be imported
_conv_stride = np.array([2,1,1,1,1])
_filtergroup_yn = np.array([0,1,0,1,1], dtype=bool)
_bnorm_yn = np.array([1,1,1,1,0], dtype=bool)
_relu_yn = np.array([1,1,1,1,0], dtype=bool)
_pool_stride = np.array([2,1,0,0,0]) # 0 means no pool
_pool_sz = 3
_bnorm_adjust = True
assert len(_conv_stride) == len(_filtergroup_yn) == len(_bnorm_yn) == len(_relu_yn) == len(_pool_stride), ('These arrays of flags should have same length')
assert all(_conv_stride) >= True, ('The number of conv layers is assumed to define the depth of the network')
_num_layers = len(_conv_stride)


class SiameseNetwork:
    """Siamese network for object detection."""

    def __init__(self, hp, design, env, run_opts = None):
        self.hp = hp
        self.design = design
        self.env = env
        self.final_score_sz = hp.response_up * (design.score_sz - 1) + 1
        self.image, self.templates_z, self.scores = self._build_tracking_graph()
        self.run_opts = {} if run_opts is None else run_opts
        self.templates_z_ = None
        self.z_sz = None
        self.x_sz = None
        self.pos_x = None
        self.pos_y = None
        self.target_w = None
        self.target_h = None
        # TODO: Clarify scale_factors calculation.
        self.scale_factors = self.hp.scale_step**np.linspace(-np.ceil(self.hp.scale_num/2), np.ceil(self.hp.scale_num/2), self.hp.scale_num)
        hann_1d = np.expand_dims(np.hanning(self.final_score_sz), axis=0)
        penalty = np.transpose(hann_1d) * hann_1d
        self.penalty = penalty / np.sum(penalty)
        self.sess = tf.Session()
        tf.initialize_all_variables().run(session=self.sess)

    def set_target(self, image, bbox):
        self.pos_x, self.pos_y, self.target_w, self.target_h = region_to_bbox(bbox, center=True)
        context = self.design.context*(self.target_w + self.target_h)
        self.z_sz = np.sqrt(np.prod((self.target_w + context) * (self.target_h + context)))
        self.x_sz = float(self.design.search_sz) / self.design.exemplar_sz * self.z_sz
        self.templates_z_ = self.get_template(image)

    def get_template(self, image):
        return self.sess.run(self.templates_z, feed_dict={
            pos_x_ph: self.pos_x,
            pos_y_ph: self.pos_y,
            z_sz_ph: self.z_sz,
            self.image: image
        })

    def detect(self, image):
        if self.templates_z_ is None:
            raise ValueError("SiameseNetwork.set_target must be called before any calls to SiameseNetwork.detect!")

        scaled_exemplar = self.z_sz * self.scale_factors
        scaled_search_area = self.x_sz * self.scale_factors
        scaled_target_w = self.target_w * self.scale_factors
        scaled_target_h = self.target_h * self.scale_factors
        scores_ = self.get_scores(image, self.pos_x, self.pos_y, scaled_search_area,
                                  np.squeeze(self.templates_z_), self.run_opts)
        scores_ = np.squeeze(scores_)
        # penalize change of scale
        scores_[0,:,:] = self.hp.scale_penalty*scores_[0,:,:]
        scores_[2,:,:] = self.hp.scale_penalty*scores_[2,:,:]
        # find scale with highest peak (after penalty)
        new_scale_id = np.argmax(np.amax(scores_, axis=(1,2)))
        # update scaled sizes
        self.x_sz = (1-self.hp.scale_lr)*self.x_sz + self.hp.scale_lr*scaled_search_area[new_scale_id]
        self.target_w = (1-self.hp.scale_lr) * self.target_w + self.hp.scale_lr*scaled_target_w[new_scale_id]
        self.target_h = (1-self.hp.scale_lr) * self.target_h + self.hp.scale_lr*scaled_target_h[new_scale_id]
        # select response with new_scale_id
        score_ = scores_[new_scale_id,:,:]
        score_ = score_ - np.min(score_)
        score_ = score_/np.sum(score_)
        # apply displacement penalty
        score_ = (1-self.hp.window_influence) * score_ + self.hp.window_influence * self.penalty
        self._update_target_position(score_)
        # update the target representation with a rolling average
        if self.hp.z_lr > 0:
            new_templates_z_ = self.get_template(image)
            self.templates_z_=(1-self.hp.z_lr) * np.asarray(self.templates_z_) + self.hp.z_lr * np.asarray(new_templates_z_)

        # update template patch size
        self.z_sz = (1-self.hp.scale_lr) * self.z_sz + self.hp.scale_lr * scaled_exemplar[new_scale_id]

        return self._get_bbox()

    def _get_bbox(self):
        return self.pos_x - self.target_w / 2, self.pos_y - self.target_h / 2, self.target_w, self.target_h

    def _update_target_position(self, score):
        # find location of score maximizer
        p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
        # displacement from the center in search area final representation ...
        center = float(self.final_score_sz - 1) / 2
        disp_in_area = p - center
        # displacement from the center in instance crop
        disp_in_xcrop = disp_in_area * float(self.design.tot_stride) / self.hp.response_up
        # displacement from the center in instance crop (in frame coordinates)
        disp_in_frame = disp_in_xcrop *  self.x_sz / self.design.search_sz
        # *position* within frame in frame coordinates
        self.pos_y, self.pos_x = self.pos_y + disp_in_frame[0], self.pos_x + disp_in_frame[1]

    def get_scores(self, image, pos_x, pos_y, scaled_search_area, template, run_opts):
        return self.sess.run([self.scores], feed_dict={
                        pos_x_ph: pos_x,
                        pos_y_ph: pos_y,
                        x_sz0_ph: scaled_search_area[0],
                        x_sz1_ph: scaled_search_area[1],
                        x_sz2_ph: scaled_search_area[2],
                        self.templates_z: template,
                        self.image: image,
                    }, **run_opts)

    def _build_tracking_graph(self):
        image = tf.placeholder(dtype=tf.uint8, shape=(None,None,None), name='image')
        image = 255.0 * tf.image.convert_image_dtype(image, tf.float32)
        frame_sz = tf.shape(image)
        # used to pad the crops
        if self.design.pad_with_image_mean:
            avg_chan = tf.reduce_mean(image, reduction_indices=(0,1), name='avg_chan')
        else:
            avg_chan = None
        # pad with if necessary
        frame_padded_z, npad_z = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, z_sz_ph, avg_chan)
        frame_padded_z = tf.cast(frame_padded_z, tf.float32)
        # extract tensor of z_crops
        z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x_ph, pos_y_ph, z_sz_ph, self.design.exemplar_sz)
        frame_padded_x, npad_x = pad_frame(image, frame_sz, pos_x_ph, pos_y_ph, x_sz2_ph, avg_chan)
        frame_padded_x = tf.cast(frame_padded_x, tf.float32)
        # extract tensor of x_crops (3 scales)
        x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x_ph, pos_y_ph, x_sz0_ph, x_sz1_ph, x_sz2_ph, self.design.search_sz)
        # use crops as input of (MatConvnet imported) pre-trained fully-convolutional Siamese net
        template_z, templates_x, p_names_list, p_val_list = self._create_siamese(os.path.join(self.env.root_pretrained,
                                                                                         self.design.net), x_crops,
                                                                            z_crops)
        template_z = tf.squeeze(template_z)
        templates_z = tf.pack([template_z, template_z, template_z])
        # compare templates via cross-correlation
        scores = self._match_templates(templates_z, templates_x, p_names_list, p_val_list)
        # upsample the score maps
        scores_up = tf.image.resize_images(scores, [self.final_score_sz, self.final_score_sz],
                                           method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
        return image, templates_z, scores_up

    # import pretrained Siamese network from matconvnet
    def _create_siamese(self, net_path, net_x, net_z):
        # read mat file from net_path and start TF Siamese graph from placeholders X and Z
        params_names_list, params_values_list = self._import_from_matconvnet(net_path)

        # loop through the flag arrays and re-construct network, reading parameters of conv and bnorm layers
        for i in xrange(_num_layers):
            print '> Layer '+str(i+1)
            # conv
            conv_W_name = self._find_params('conv'+str(i+1)+'f', params_names_list)[0]
            conv_b_name = self._find_params('conv'+str(i+1)+'b', params_names_list)[0]
            print '\t\tCONV: setting '+conv_W_name+' '+conv_b_name
            print '\t\tCONV: stride '+str(_conv_stride[i])+', filter-group '+str(_filtergroup_yn[i])
            conv_W = params_values_list[params_names_list.index(conv_W_name)]
            conv_b = params_values_list[params_names_list.index(conv_b_name)]
            # batchnorm
            if _bnorm_yn[i]:
                bn_beta_name = self._find_params('bn'+str(i+1)+'b', params_names_list)[0]
                bn_gamma_name = self._find_params('bn'+str(i+1)+'m', params_names_list)[0]
                bn_moments_name = self._find_params('bn'+str(i+1)+'x', params_names_list)[0]
                print '\t\tBNORM: setting '+bn_beta_name+' '+bn_gamma_name+' '+bn_moments_name
                bn_beta = params_values_list[params_names_list.index(bn_beta_name)]
                bn_gamma = params_values_list[params_names_list.index(bn_gamma_name)]
                bn_moments = params_values_list[params_names_list.index(bn_moments_name)]
                bn_moving_mean = bn_moments[:,0]
                bn_moving_variance = bn_moments[:,1]**2 # saved as std in matconvnet
            else:
                bn_beta = bn_gamma = bn_moving_mean = bn_moving_variance = []

            # set up conv "block" with bnorm and activation
            net_x = set_convolutional(net_x, conv_W, np.swapaxes(conv_b,0,1), _conv_stride[i], \
                                      bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance, \
                                      filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i], \
                                      scope='conv'+str(i+1), reuse=False)

            # notice reuse=True for Siamese parameters sharing
            net_z = set_convolutional(net_z, conv_W, np.swapaxes(conv_b,0,1), _conv_stride[i], \
                                      bn_beta, bn_gamma, bn_moving_mean, bn_moving_variance, \
                                      filtergroup=_filtergroup_yn[i], batchnorm=_bnorm_yn[i], activation=_relu_yn[i], \
                                      scope='conv'+str(i+1), reuse=True)

            # add max pool if required
            if _pool_stride[i]>0:
                print '\t\tMAX-POOL: size '+str(_pool_sz)+ ' and stride '+str(_pool_stride[i])
                net_x = tf.nn.max_pool(net_x, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))
                net_z = tf.nn.max_pool(net_z, [1,_pool_sz,_pool_sz,1], strides=[1,_pool_stride[i],_pool_stride[i],1], padding='VALID', name='pool'+str(i+1))

        print

        return net_z, net_x, params_names_list, params_values_list

    def _import_from_matconvnet(self, net_path):
        mat = scipy.io.loadmat(net_path)
        net_dot_mat = mat.get('net')
        # organize parameters to import
        params = net_dot_mat['params']
        params = params[0][0]
        params_names = params['name'][0]
        params_names_list = [params_names[p][0] for p in xrange(params_names.size)]
        params_values = params['value'][0]
        params_values_list = [params_values[p] for p in xrange(params_values.size)]
        return params_names_list, params_values_list


    # find all parameters matching the codename (there should be only one)
    def _find_params(self, x, params):
        matching = [s for s in params if x in s]
        assert len(matching)==1, ('Ambiguous param name found')
        return matching

    def _match_templates(self, net_z, net_x, params_names_list, params_values_list):
        # finalize network
        # z, x are [B, H, W, C]
        net_z = tf.transpose(net_z, perm=[1,2,0,3])
        net_x = tf.transpose(net_x, perm=[1,2,0,3])
        # z, x are [H, W, B, C]
        Hz, Wz, B, C = tf.unpack(tf.shape(net_z))
        Hx, Wx, Bx, Cx = tf.unpack(tf.shape(net_x))
        # assert B==Bx, ('Z and X should have same Batch size')
        # assert C==Cx, ('Z and X should have same Channels number')
        net_z = tf.reshape(net_z, (Hz, Wz, B*C, 1))
        net_x = tf.reshape(net_x, (1, Hx, Wx, B*C))
        net_final = tf.nn.depthwise_conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID')
        # final is [1, Hf, Wf, BC]
        net_final = tf.concat(0, tf.split(3, 3, net_final))
        # final is [B, Hf, Wf, C]
        net_final = tf.expand_dims(tf.reduce_sum(net_final, reduction_indices=3), dim=3)
        # final is [B, Hf, Wf, 1]
        if _bnorm_adjust:
            bn_beta = params_values_list[params_names_list.index('fin_adjust_bnb')]
            bn_gamma = params_values_list[params_names_list.index('fin_adjust_bnm')]
            bn_moments = params_values_list[params_names_list.index('fin_adjust_bnx')]
            bn_moving_mean = bn_moments[:,0]
            bn_moving_variance = bn_moments[:,1]**2
            param_initializer = {
                'beta': tf.constant_initializer(bn_beta),
                'gamma':  tf.constant_initializer(bn_gamma),
                'moving_mean': tf.constant_initializer(bn_moving_mean),
                'moving_variance': tf.constant_initializer(bn_moving_variance)
            }
            net_final = tf.contrib.layers.batch_norm(net_final, initializers=param_initializer,
                                                     is_training=False, trainable=False)

        return net_final
