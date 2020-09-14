import math
import os.path
import tarfile
import time

import tensorflow.compat.v1 as tf
from tqdm import trange

from .abc_runner import *
from .sample_runner import *

__all__ = ['InceptionRunner']

softmax = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class InceptionRunner(SampleRunner):

    @torch.no_grad()
    def run(self):

        sigmas = self.get_sigmas(npy=True)
        dataloader = self.get_dataloader(bs=self.config.sampling.batch_size)
        score = self.load_score(eval=True)

        kwargs = {'scorenet': score, 'sigmas': sigmas, 'nsigma': self.config.sampling.nsigma,
                  'step_lr': self.config.sampling.step_lr, 'final_only': True, 'target': self.args.target,
                  'noise_first': self.config.sampling.noise_first}

        x, x_denoised = [], []

        bs = self.config.fast_fid.batch_size
        for k in range(self.config.fast_fid.num_samples // bs):

            all_samples, all_samples_denoised = self.sample(dataloader, saveimages=False, kwargs=kwargs, bs=bs)
            x += [*np.uint8(255 * all_samples[-1].cpu().numpy())]
            x_denoised += [*np.uint8(255 * all_samples_denoised[-1].cpu().numpy())]

        # if softmax is None: # No need to functionalize like this.
        tf.disable_v2_behavior()
        _init_inception()

        self.exec_inception(np.array(x), 'non-denoised')
        self.exec_inception(np.array(x_denoised), 'denoised')

    def exec_inception(self, ims, comment):
        # Inception with TF1.3 or earlier.
        # Call this function with list of images. Each of elements should be a
        # numpy array with values ranging from 0 to 255.

        def get_inception_score(images, splits=10):
            assert (type(images) == list)
            assert (type(images[0]) == np.ndarray)
            assert (len(images[0].shape) == 3)
            assert (np.max(images[0]) > 10)
            assert (np.min(images[0]) >= 0.0)
            inps = [np.expand_dims(img.astype(np.float32), 0) for img in images]

            bs = self.args.batch_size
            with tf.Session() as sess:
                preds = []
                n_batches = int(math.ceil(float(len(inps)) / float(bs)))

                for i in trange(n_batches):
                    inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
                    inp = np.concatenate(inp, 0)
                    pred = sess.run(softmax, {'InputTensor:0': inp})
                    preds.append(pred)

                preds = np.concatenate(preds, 0)
                scores = []

                for i in range(splits):
                    part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
                    kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
                    kl = np.mean(np.sum(kl, 1))
                    scores.append(np.exp(kl))
                return np.mean(scores), np.std(scores)

        t_start = time.time()
        ims_swaped = ims.swapaxes(1, 2).swapaxes(2, 3)

        inc_mean, inc_std = get_inception_score(list(ims_swaped), splits=10)
        print('Saving pool to numpy file for FID calculations...')
        print('Inception took {} seconds, score of {} +/- {}.'.format(time.time() - t_start, inc_mean, inc_std))

        fid_folder = os.path.join(self.args.fid_folder, os.environ["USER"], "Output" + str(os.environ["slot"]), "Extra")
        os.makedirs(fid_folder, exist_ok=True)

        o = "({}) {} Inception-Mean: {}, Inception-Std: {}".format(comment, self.args.doc, inc_mean, inc_std)
        print(o)
        print(o, file=open(f"{fid_folder}/log_FID.txt", 'a+'))

def _init_inception():
    global softmax

    MODEL_DIR = 'evaluation'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    filename = 'inception-2015-12-05.tgz'
    filepath = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(filepath):
        raise NotADirectoryError(f"Inception file '{filepath}' not found. Downloads have been disabled.")

    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='InputTensor')
        tf.import_graph_def(graph_def, name='', input_map={'ExpandDims:0': input_tensor})

    with tf.Session() as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for outp in op.outputs:
                shape = outp.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                outp.set_shape(tf.TensorShape(new_shape))
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        softmax = tf.nn.softmax(logits)
