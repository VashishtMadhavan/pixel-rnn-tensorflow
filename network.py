import tensorflow as tf
from logging import getLogger

from ops import *
from utils import *

logger = getLogger(__name__)

class Network:
  def __init__(self, sess, args, height, width, channel):
    logger.info("Building %s starts!" % args.model)

    self.sess = sess
    self.color_dim = 3
    self.data = args.data
    self.height, self.width, self.channel = height, width, channel
    input_shape = [None, height, width, channel]

    self.inputs = tf.placeholder(tf.float32, [None, height, width, channel])

    # input of main reccurent layers
    self.first_mask = conv2d(self.inputs, args.hidden_dims, [7, 7], "A", scope='conv_inputs')
    
    # main reccurent layers
    x = self.first_mask
    for idx in xrange(args.recurrent_length):
        x = conv2d(x, 3, [1, 1], "B", scope='conv_{}'.format(idx))

    # output reccurent layers
    for idx in xrange(args.out_recurrent_length):
      x = tf.nn.relu(conv2d(x, args.out_hidden_dims, [1, 1], "B", scope='conv_out_{}'.format(idx)))

    self.out_logits = conv2d(x, self.color_dim, [1, 1], "B", scope='conv2d_out_logits')
    self.out_logits_flat = tf.reshape(self.out_logits, [-1, self.height*self.width, self.color_dim])
    self.inputs_flat = tf.reshape(self.inputs, [-1, self.height*self.width], self.color_dim)

    self.output = tf.nn.softmax(self.out_logits)
    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.out_logits, self.inputs, name='loss'))

    optimizer = tf.train.RMSPropOptimizer(args.learning_rate)
    grads_and_vars = optimizer.compute_gradients(self.loss)
    new_grads_and_vars = [(tf.clip_by_value(gv[0], -args.grad_clip, args.grad_clip), gv[1]) for gv in grads_and_vars]
    self.optim = optimizer.apply_gradients(new_grads_and_vars)
    show_all_variables()

  def predict(self, x):
    return self.sess.run(self.output, {self.inputs: x})

  def test(self, x, update=False):
    if update:
      _, cost = self.sess.run([self.optim, self.loss], feed_dict={ self.inputs: x})
    else:
      cost = self.sess.run(self.loss, feed_dict={self.inputs: x})
    return cost

  def generate(self):
    samples = np.zeros((100, self.height, self.width, 1), dtype='float32')
    for i in xrange(self.height):
      for j in xrange(self.width):
        for k in xrange(self.channel):
          next_sample = binarize(self.predict(samples))
          samples[:, i, j, k] = next_sample[:, i, j, k]
          if self.data == 'mnist':
            print "=" * (self.width/2), "(%2d, %2d)" % (i, j), "=" * (self.width/2)
            mprint(next_sample[0,:,:,:])
    return samples
