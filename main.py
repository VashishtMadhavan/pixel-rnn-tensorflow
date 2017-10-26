import os
import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import numpy as np
from tqdm import trange
import tensorflow as tf

from utils import *
from network import Network
from statistic import Statistic
import argparse
from tensorflow.examples.tutorials.mnist import input_data
import gym
import gym.wrappers as gym


def parse_args():
  parser = argparse.ArgumentParser()

  # network
  parser.add_argument("--model", type=str, default="pixel_cnn", help="name of model [pixel_rnn, pixel_cnn]")
  parser.add_argument("--batch_size", type=int, default=100, help="size of a batch")
  parser.add_argument("--hidden_dims", type=int, default=16, help="dimesion of hidden states of LSTM or Conv layers")
  parser.add_argument("--recurrent_length", type=int, default=7, help="the length of LSTM or Conv layers")
  parser.add_argument("--out_hidden_dims", type=int, default=32, help="dimesion of hidden states of output Conv layers")
  parser.add_argument("--out_recurrent_length", type=int, default=2, help="the length of output Conv layers")
  parser.add_argument("--use_residual",type=bool, default=False, help="whether to use residual connections or not")
  parser.add_argument("--use_dynamic_rnn", type=bool, default=False, help="whether to use dynamic_rnn or not")

  # training
  parser.add_argument("--max_epoch", type=int, default=100000, help="# of step in an epoch")
  parser.add_argument("--test_step", type=int, default=100, help="# of step to test a model")
  parser.add_argument("--save_step", type=int, default=1000, help="# of step to save a model")
  parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
  parser.add_argument("--grad_clip", type=int, default=1, help="value of gradient to be used for clipping")
  parser.add_argument("--gpu", type=int, default=-1, help="which gpu to use for training; -1 default")

  # data
  parser.add_argument("--data", type=str, default="mnist", help="name of dataset [mnist, gym]")
  parser.add_argument("--env_id", type=str, default=None, help='when using gym data, specify an env')
  parser.add_argument("--data_dir", type=str, default="data", help="name of data directory")
  parser.add_argument("--output_dir", type=str, default='model_output', help='where to store all model data')

  # Debug
  parser.add_argument("--is_train", type=str, default=True, help="training or testing")
  parser.add_argument("--display", type=bool, default=False, help="whether to display the training results or not")
  parser.add_argument("--log_level", type=str, default="INFO", help="log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
  parser.add_argument("--random_seed", type=int, default=123, help="random seed for python")
  return parser.parse_args()

def check_and_load_data(env, filename, phase='train'):
  if not os.path.exists(filename):
    data = gather_data(env, phase)
    pickle.dump(data, open(filename, 'wb'))
  else:
    data = pickle.load(open(filename, 'rb'))
  return data

def prepocess_frame(frame, bins=8):
  obs = frame[:,:,:1]
  obs = np.round(obs * bins).astype(np.uint8)
  return obs

def gather_data(env, phase='train'):
  if phase == 'train':
    num_eps = 60
  else:
    num_eps = 20
  num_actions = env.action_space.n
  eps = 0; obs = env.reset(); data = []
  while eps < num_eps:
    obs, rew, done, info = env.step(np.random.choice(env.action_space.n))
    data.append(preprocess_frame(obs))
    if done:
      obs = env.reset()
      eps += 1
  return np.array(data)

def main(args):

  # logging
  logger = logging.getLogger()
  logger.setLevel(args.log_level)

  # random seed
  tf.set_random_seed(args.random_seed)
  np.random.seed(args.random_seed)

  data_dir = os.path.join(args.data_dir, args.data)
  output_dir = os.path.join(args.output_dir, args.data)
  check_and_create_dir(data_dir)
  check_and_create_dir(output_dir)

  # 0. prepare datasets
  if args.data == "mnist":
    mnist = input_data.read_data_sets(data_dir, one_hot=True)

    height, width, channel = 28, 28, 1
    color_dim = 2
    next_train_batch = lambda x: binarize(mnist.train.next_batch(x)[0]).reshape([x, height, width, channel])
    next_test_batch = lambda x: binarize(mnist.test.next_batch(x)[0]).reshape([x, height, width, channel])

    train_step_per_epoch = mnist.train.num_examples // args.batch_size
    test_step_per_epoch = mnist.test.num_examples // args.batch_size

  elif args.data == 'gym':
    env = gym.make(args.env_id + "NoFrameskip-v4")
    env = wrappers.ScaledFloatFrame(wrappers.wrap_deepmind(env))
    train_data = check_and_load_data(env, data_dir + "/train_data.pkl", phase='train')
    test_data = check_and_load_data(env, data_dir + "/test_data.pkl", phase='test')

    next_train_batch = lambda x: train_data[np.random.randint(len(train_data), size=x)]
    next_test_batch = lambda x: test_data[np.random.randint(len(test_data), size=x)]
    height, width, channel = 84, 84, 1
    color_dim = 8

    train_step_per_epoch = train_data.shape[0] // args.batch_size
    test_step_per_epoch = test_data.shape[0] // args.batch


  with tf.Session() as sess:
    network = Network(sess, args, height, width, channel, color_dim=color_dim)
    stat = Statistic(sess, args.data, output_dir, tf.trainable_variables(), args.test_step)
    stat.load_model()

    if args.is_train:
      logger.info("Training starts!")
      initial_step = stat.get_t() if stat else 0
      iterator = trange(args.max_epoch, ncols=70, initial=initial_step)

      for epoch in iterator:
        # 1. train
        total_train_costs = []; total_train_logl = []
        for idx in range(train_step_per_epoch):
          images = next_train_batch(args.batch_size)
          logl, cost = network.test(images, update=True)
          total_train_costs.append(cost)
          total_train_logl.append(logl)
        
        # 2. test
        total_test_costs = []; total_test_logl = []
        for idx in range(test_step_per_epoch):
          images = next_test_batch(args.batch_size)
          logl, cost = network.test(images, update=False)
          total_test_costs.append(cost)
          total_test_logl.append(logl)

        avg_train_cost, avg_test_cost = np.mean(total_train_costs), np.mean(total_test_costs)
        stat.on_step(avg_train_cost, avg_test_cost)

        # 3. generate samples
        samples = network.generate()
        save_images(samples, height, width, 10, 10, directory=output_dir, prefix="epoch_%s" % epoch)
        iterator.set_description("train l: %.3f, test l: %.3f" % (avg_train_cost, avg_test_cost))
    else:
      logger.info("Image generation starts!")
      samples = network.generate()
      save_images(samples, height, width, 10, 10, directory=output_dir)


if __name__ == "__main__":
  args = parse_args()
  main(args)