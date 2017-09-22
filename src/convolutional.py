import tensorflow as tf


def set_convolutional(X, W, b, stride, bn_beta, bn_gamma, bn_mm, bn_mv, filtergroup=False, batchnorm=True,
                      activation=True, scope=None, reuse=False):
    # for var, var_name in zip([bn_beta, bn_gamma, bn_mm, bn_mv], ['bn_beta', 'bn_gamma', 'bn_mm', 'bn_mv']:
    # use the input scope or default to "conv"
    with tf.variable_scope(scope or 'conv', reuse=reuse):
        # sanity check    
	print "W SHAPE: {0}".format(W.shape)
	print "X SHAPE: {0}".format(X.get_shape())
        W = tf.get_variable("W", W.shape, trainable=False, initializer=tf.constant_initializer(W))
        b = tf.get_variable("b", b.shape, trainable=False, initializer=tf.constant_initializer(b))

	# "filtergroup" Is a bizarre name for this variable. This is a holdover from the original
	# alexnet, and is used to split the network across two separate GPUs. The 
	# input tensor is split along the in_channel axis, and the filter is
	# split along out_channel axis (in both cases this ends up being dimension 4, which is why
	# the leading '3' argument to tf.split. The '2' arg splits the tensor into 2 pieces
	# (surprise).
	# What remains unclear is how and why the channels argument ends up coming first on the
	# original input tensor (the image). While the documentation for tf r0.11 and r1.3 say that
	# the input tensor format must be (batch_size, width, height, channels), printing the input
	# tensor gives (3, 255, 255, ?) suggesting that the batch_size is at the end. It is also
	# unclear how tensorflow resolves the issue since during inference batch_size = 1, which
	# seems like it would break the convolution.
        if filtergroup:
            X0, X1 = tf.split(3, 2, X)
            W0, W1 = tf.split(3, 2, W)
            h0 = tf.nn.conv2d(X0, W0, strides=[1, stride, stride, 1], padding='VALID')
            h1 = tf.nn.conv2d(X1, W1, strides=[1, stride, stride, 1], padding='VALID')
            h = tf.concat(3, [h0, h1]) + b
        else:
            h = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='VALID') + b

        if batchnorm:
	    param_initializer = {
		'beta': tf.constant_initializer(bn_beta),
		'gamma':  tf.constant_initializer(bn_gamma),
		'moving_mean': tf.constant_initializer(bn_mm),
		'moving_variance': tf.constant_initializer(bn_mv)
	    }
            h = tf.contrib.layers.batch_norm(h, initializers=param_initializer,
					      is_training=False, trainable=False)

        if activation:
            h = tf.nn.relu(h)

        return h
