## Code modified from : https://github.com/ajbrock/BigGAN-PyTorch/blob/master/inception_tf13.py
# Tensorflow inception score code
# Derived from https://github.com/openai/improved-gan
# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
# THIS CODE REQUIRES TENSORFLOW 1.3 or EARLIER to run in PARALLEL BATCH MODE 
# To use this code, run sample.py on your model with --sample_npz, and then 
# pass the experiment name in the --experiment_name.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import os.path
import sys
import tarfile
from argparse import ArgumentParser

import numpy as np
import tensorflow.compat.v1 as tf
from six.moves import urllib
from tqdm import trange

tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

MODEL_DIR = 'evaluation'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

def prepare_parser():
  usage = 'Parser for TF1.3- Inception Score scripts.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--experiment_root', type=str, default='samples',
    help='Default location where samples are stored (default: %(default)s)')
  parser.add_argument(
    '--batch_size', type=int, default=500,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--fid_folder', type=str, default='/scratch')
  parser.add_argument(
    '--doc', type=str, default='mymodel')
  parser.add_argument(
    '--string', type=str, default='Samples')
  return parser


def run(config):
  # Inception with TF1.3 or earlier.
  # Call this function with list of images. Each of elements should be a 
  # numpy array with values ranging from 0 to 255.
  def get_inception_score(images, splits=10):
    assert(type(images) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
      img = img.astype(np.float32)
      inps.append(np.expand_dims(img, 0))
    bs = config['batch_size']
    with tf.Session() as sess:
      preds = []
      n_batches = int(math.ceil(float(len(inps)) / float(bs)))
      for i in trange(n_batches):
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'InputTensor:0': inp})
        #pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)
      preds = np.concatenate(preds, 0)
      scores = []
      for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
      return np.mean(scores), np.std(scores)
  # Init inception
  def _init_inception():
    global softmax
    if not os.path.exists(MODEL_DIR):
      os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
      statinfo = os.stat(filepath)
      print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
        MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                                    name='InputTensor')
      _ = tf.import_graph_def(graph_def, name='',
                              input_map={'ExpandDims:0':input_tensor})
      #_ = tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
      pool3 = sess.graph.get_tensor_by_name('pool_3:0')
      ops = pool3.graph.get_operations()
      for op_idx, op in enumerate(ops):
        for o in op.outputs:
          shape = o.get_shape()
          shape = [s.value for s in shape]
          new_shape = []
          for j, s in enumerate(shape):
            if s == 1 and j == 0:
              new_shape.append(None)
            else:
              new_shape.append(s)
          o.set_shape(tf.TensorShape(new_shape))
          #o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
      w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
      logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
      softmax = tf.nn.softmax(logits)

  # if softmax is None: # No need to functionalize like this.
  _init_inception()

  fname = '%s/samples.npz' % config['experiment_root']
  print('loading %s ...'%fname)
  ims = np.load(fname)['x']
  import time
  t0 = time.time()
  print(ims.swapaxes(1,2).swapaxes(2,3).shape)
  inc_mean, inc_std = get_inception_score(list(ims.swapaxes(1,2).swapaxes(2,3)), splits=10)
  t1 = time.time()
  print('Saving pool to numpy file for FID calculations...')
  print('Inception took %3f seconds, score of %3f +/- %3f.'%(t1-t0, inc_mean, inc_std))

  config['fid_folder'] = config['fid_folder'] + "/" + os.environ["USER"] + "/Output" + os.environ["slot"] + "/Extra"
  os.makedirs(config['fid_folder'], exist_ok=True)
  log_output = open(f"{config['fid_folder']}/log_FID.txt", 'a+')
  print("({}) {} Inception-Mean: {}, Inception-Std: {}".format(config['string'], config['doc'], inc_mean, inc_std))
  print("({}) {} Inception-Mean: {}, Inception-Std: {}".format(config['string'],config['doc'], inc_mean, inc_std), file=log_output)
def main():
  # parse command line and run
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)

if __name__ == '__main__':
  main()