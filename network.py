import tensorflow as tf
from logging import getLogger

from ops import *
from utils import *

import pdb
logger = getLogger(__name__)

class Network:
  def __init__(self, sess, args, height, width, channel, color_dim=8):
    logger.info("Building %s starts!" % args.model)

    self.sess = sess
    self.color_dim = color_dim
    self.data = args.data
    self.height, self.width, self.channel = height, width, channel
    input_shape = [None, height, width, channel]

    self.inputs = tf.placeholder(tf.float32, [None, height, width, channel])

    # input of main reccurent layers
    self.first_mask = conv2d(self.inputs, args.hidden_dims, [7, 7], "A", scope='conv_inputs')
    
    # main reccurent layers
    x = self.first_mask
    for idx in range(args.recurrent_length):
        x = conv2d(x, 3, [1, 1], "B", scope='conv_{}'.format(idx))

    # output reccurent layers
    for idx in range(args.out_recurrent_length):
      x = tf.nn.relu(conv2d(x, args.out_hidden_dims, [1, 1], "B", scope='conv_out_{}'.format(idx)))

    self.out_logits = conv2d(x, self.color_dim, [1, 1], "B", scope='conv2d_out_logits')
    self.out_logits_flat = tf.reshape(self.out_logits, [-1, self.height*self.width, self.color_dim])

    self.inputs_one_hot =  tf.one_hot(tf.cast(tf.squeeze(self.inputs, axis=-1), tf.uint8), depth=self.color_dim)
    self.inputs_one_hot_flat = tf.reshape(self.inputs_one_hot, [-1, self.height*self.width, self.color_dim])

    self.output = tf.nn.softmax(self.out_logits)
    self.prediction = tf.argmax(self.output, axis=3)

    self.output_log_probs = tf.multiply(tf.nn.log_softmax(self.out_logits), self.inputs_one_hot)
    self.log_likelihood = tf.reduce_sum(self.output_log_probs)

    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.out_logits, self.inputs_one_hot, name='loss'))

    optimizer = tf.train.RMSPropOptimizer(args.learning_rate)
    grads_and_vars = optimizer.compute_gradients(self.loss)
    new_grads_and_vars = [(tf.clip_by_value(gv[0], -args.grad_clip, args.grad_clip), gv[1]) for gv in grads_and_vars]
    self.optim = optimizer.apply_gradients(new_grads_and_vars)
    show_all_variables()

  def predict(self, x):
    logl, pred = self.sess.run([self.log_likelihood, self.prediction], {self.inputs: x})
    return logl, pred

  def test(self, x, update=False):
    if update:
      _, logl, cost = self.sess.run([self.optim, self.log_likelihood, self.loss], feed_dict={ self.inputs: x})
    else:
      logl, cost = self.sess.run([self.log_likelihood, self.loss], feed_dict={self.inputs: x})
    return logl, cost

  def generate(self):
    samples = np.zeros((100, self.height, self.width, 1), dtype='float32')
    for i in range(self.height):
      for j in range(self.width):
        for k in range(self.channel):
          _, next_sample = self.predict(samples)
          #hack for now; need to generalize
          next_sample = next_sample[:,:,:,np.newaxis]
          samples[:, i, j, k] = next_sample[:, i, j, k]
          #if self.data == 'mnist':
          #  print("=" * (self.width // 2), "(%2d, %2d)" % (i, j), "=" * (self.width // 2))
          #  mprint(next_sample[0,:,:,:])
    return samples
