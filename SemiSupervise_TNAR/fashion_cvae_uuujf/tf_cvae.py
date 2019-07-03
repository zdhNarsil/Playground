import os
import GPUtil
from datetime import datetime
import math
import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--iter-per-epoch', type=int, default=400)
parser.add_argument('--epoch', type=int, default=121)
parser.add_argument('--lr-decay-epoch', type=int, default=81)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--latent-dim', type=int, default=100)
parser.add_argument('--kld-coef', type=float, default=1.0)
parser.add_argument('--logdir', type=str, default='./cvae_logs')


def main():
    args = parser.parse_args()
    print(args)
    log_dir = args.logdir + '/latent' + str(args.latent_dim) + '_lr' + str(args.lr) + '_kld' + str(
        args.kld_coef) + '_seed' + str(args.seed) + '_' + str(datetime.now())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
    # set random seed
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    # set gpu
    deviceIDs = GPUtil.getAvailable(order='first', limit=4, maxLoad=0.1, maxMemory=0.1, excludeID=[], excludeUUID=[])
    print('Unloaded gpu:', deviceIDs)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(deviceIDs[0])

    # dataset
    Dataset = np.load(os.environ['HOME'] + '/datasets/fashion/fashion.npz')
    X_train = Dataset['xtrain']
    X_test = Dataset['xtest']
    Y_train = Dataset['ytrain']
    Y_test = Dataset['ytest']

    # build graph
    lr = tf.placeholder(tf.float32, [], name='lr')
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    vae = cVAE(args.latent_dim)
    vae.build_graph(x, y, lr, args.kld_coef)

    saver = tf.train.Saver(max_to_keep=100)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        for ep in range(args.epoch):
            if ep < args.lr_decay_epoch:
                decayed_lr = args.lr
            else:
                decayed_lr = args.lr * (args.epoch - ep) / float(args.epoch - args.lr_decay_epoch)

            for i in range(args.iter_per_epoch):
                mask = np.random.choice(len(X_train), args.batch_size, False)
                _ = sess.run(vae.train_step,
                             feed_dict={x: X_train[mask], y: Y_train[mask], lr: decayed_lr, vae.is_train: True})

            test_summary, bce, kld, mse = sess.run([vae.summary, vae.bce_loss, vae.kld_loss, vae.mse_loss],
                                                   feed_dict={x: X_test[0:200], y: Y_test[0:200], vae.is_train: False})
            writer.add_summary(test_summary, ep)
            print('epoch:', ep, 'loss_bce:', bce, 'loss_kld:', kld, 'mse_loss:', mse)
            if ep % 20 == 0:
                saver.save(sess, os.path.join(log_dir, 'model'), ep)
    print(args)


class cVAE(object):
    def __init__(self, latent_dim):
        super(cVAE, self).__init__()
        self.is_train = tf.placeholder(tf.bool, name='cvae_is_train')
        self.reuse = {}
        self.latent_dim = latent_dim

    def encode(self, x, y, name='cvae/encoder'):
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            w1 = tf.get_variable(name='w_conv1', shape=[5, 5, 1, 16], dtype=tf.float32)
            b1 = tf.get_variable(name='b_conv1', shape=[16], dtype=tf.float32)
            w2 = tf.get_variable(name='w_conv2', shape=[5, 5, 16, 32], dtype=tf.float32)
            b2 = tf.get_variable(name='b_conv2', shape=[32], dtype=tf.float32)
            w_y1 = tf.get_variable(name='w_y1', shape=[10, 128], dtype=tf.float32)
            b_y1 = tf.get_variable(name='b_y1', shape=[128], dtype=tf.float32)
            w_y2 = tf.get_variable(name='w_y2', shape=[128, 128], dtype=tf.float32)
            b_y2 = tf.get_variable(name='b_y2', shape=[128], dtype=tf.float32)
            w21 = tf.get_variable(name='w_fc21', shape=[49 * 32 + 128, self.latent_dim], dtype=tf.float32)
            b21 = tf.get_variable(name='b_fc21', shape=[self.latent_dim], dtype=tf.float32)
            w22 = tf.get_variable(name='w_fc22', shape=[49 * 32 + 128, self.latent_dim], dtype=tf.float32)
            b22 = tf.get_variable(name='b_fc22', shape=[self.latent_dim], dtype=tf.float32)

            x = tf.reshape(x, [-1, 28, 28, 1])
            x = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
            x = tf.nn.relu(tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))
            x = tf.nn.conv2d(x, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
            x = tf.nn.relu(tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))
            y = tf.matmul(y, w_y1) + b_y1
            y = tf.nn.relu(y)
            y = tf.matmul(y, w_y2) + b_y2
            y = tf.nn.relu(y)
            x = tf.reshape(x, [-1, 49 * 32])
            x = tf.concat([x, y], axis=1)
            mu = tf.matmul(x, w21) + b21
            logvar = tf.matmul(x, w22) + b22
            return mu, logvar

    def reparamenterize(self, mu, logvar):
        return tf.cond(self.is_train, lambda: tf.random_normal(tf.shape(mu), mean=mu, stddev=tf.exp(0.5 * logvar)),
                       lambda: mu)

    def decode(self, z, y, name='cvae/decoder'):
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            w3 = tf.get_variable(name='w_conv3', shape=[5, 5, 1, 16], dtype=tf.float32)
            b3 = tf.get_variable(name='b_conv3', shape=[1], dtype=tf.float32)
            w4 = tf.get_variable(name='w_conv4', shape=[5, 5, 16, 32], dtype=tf.float32)
            b4 = tf.get_variable(name='b_conv4', shape=[16], dtype=tf.float32)
            w5 = tf.get_variable(name='w_fc5', shape=[self.latent_dim + 10, 49 * 32], dtype=tf.float32)
            b5 = tf.get_variable(name='b_fc5', shape=[49 * 32], dtype=tf.float32)

            z = tf.concat([z, y], axis=1)
            z = tf.matmul(z, w5) + b5
            z = tf.nn.relu(z)
            z = tf.reshape(z, [-1, 7, 7, 32])
            bs = tf.shape(z)[0]
            z = tf.nn.conv2d_transpose(z, w4, output_shape=[bs, 14, 14, 16], strides=[1, 2, 2, 1], padding='SAME') + b4
            z = tf.nn.relu(z)
            z = tf.nn.conv2d_transpose(z, w3, output_shape=[bs, 28, 28, 1], strides=[1, 2, 2, 1], padding='SAME') + b3
            z = tf.nn.sigmoid(z)
            z = tf.reshape(z, [-1, 784])
            return z

    def BCE(self, recon_x, x, mu, logvar):
        ret = -tf.reduce_mean(tf.reduce_sum(x * tf.log(recon_x + 1e-8) + (1 - x) * tf.log(1 - recon_x + 1e-8), axis=1))
        return ret

    def KLD(self, mu, logvar):
        ret = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + logvar - tf.square(mu) - tf.exp(logvar)), axis=1))
        return ret

    def build_graph(self, x, y, lr, kld_coef=1.0):
        mu, logvar = self.encode(x, y)
        self.zx = self.reparamenterize(mu, logvar)
        self.xzx = self.decode(self.zx, y)

        self.xz = self.decode(tf.random_normal([tf.shape(y)[0], self.latent_dim]), y)

        self.bce_loss = self.BCE(self.xzx, x, mu, logvar)
        self.kld_loss = self.KLD(mu, logvar)
        self.total_loss = self.bce_loss + kld_coef * self.kld_loss
        self.mse_loss = tf.reduce_mean(tf.reduce_sum((x - self.xzx) ** 2, axis=1))

        self.train_step = tf.train.AdamOptimizer(lr).minimize(self.total_loss)

        loss_summary_list = [tf.summary.scalar('loss/bce', self.bce_loss),
                             tf.summary.scalar('loss/kld', self.kld_loss),
                             tf.summary.scalar('loss/mse', self.mse_loss)]
        image_summary_list = [tf.summary.image('x', tf.reshape(x, [-1, 28, 28, 1]), max_outputs=32),
                              tf.summary.image('recon_x', tf.reshape(self.xzx, [-1, 28, 28, 1]), max_outputs=32),
                              tf.summary.image('gene_x', tf.reshape(self.xz, [-1, 28, 28, 1]), max_outputs=32)]
        self.summary = tf.summary.merge(loss_summary_list + image_summary_list)
        return True


if __name__ == '__main__':
    main()
