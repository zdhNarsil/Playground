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
parser.add_argument('--epoch', type=int, default=31)
parser.add_argument('--lr-decay-epoch', type=int, default=21)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--batch-size-ul', type=int, default=128)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--epsilon', type=float, default=3.0)
parser.add_argument('--numX', type=int, default=200)
parser.add_argument('--logdir', type=str, default='./vat_logs')


def main():
    args = parser.parse_args()
    print(args)
    log_dir = args.logdir + '_' + str(args.numX) + '/lr' + str(args.lr) + '_epsilon' + str(args.epsilon) + str(
        args.alpha) + '_seed' + str(args.seed) + '_' + str(datetime.now())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
    # set random seed
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    # set gpu
    deviceIDs = GPUtil.getAvailable(order='first', limit=4, maxLoad=0.6, maxMemory=0.2, excludeID=[], excludeUUID=[])
    print('Unloaded gpu:', deviceIDs)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(deviceIDs[0])

    # dataset
    Dataset = np.load(os.environ['HOME'] + '/datasets/fashion/fashion.npz')
    Xul_train = Dataset['xtrain'][0:60000]
    Yul_train = Dataset['ytrain'][0:60000]
    X_test = Dataset['xtest']
    Y_test = Dataset['ytest']
    # mask = np.random.choice(60000, 500, False)
    mask = np.arange(0, args.numX)
    X_train = Xul_train[mask]
    Y_train = Yul_train[mask]
    for i in range(10):
        print('class:', i, np.sum((Y_train[:, i] == 1)))

    # build graph
    lr = tf.placeholder(tf.float32, [], name='lr')
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')
    x_ul = tf.placeholder(tf.float32, [None, 784], name='xul')

    net = Net()
    out = net.classifier(x)
    out_ul = net.classifier(x_ul)

    ###################################
    # power method - begin
    def normalizevector(r):
        r /= (1e-12 + tf.reduce_max(tf.abs(r), axis=1, keepdims=True))
        return r / tf.sqrt(tf.reduce_sum(r ** 2, axis=1, keepdims=True) + 1e-6)

    r_adv = normalizevector(tf.random_normal(shape=tf.shape(x_ul)))
    for j in range(1):
        r_adv = 1e-6 * r_adv
        out_r = net.classifier(x_ul + r_adv)
        kl = net.kldivergence(out_ul, out_r)
        r_adv = tf.stop_gradient(tf.gradients(kl, r_adv)[0])
        r_adv = normalizevector(r_adv)
    out_adv = net.classifier(x_ul + args.epsilon * r_adv)

    vat_loss = net.kldivergence(tf.stop_gradient(out_ul), out_adv)
    # power method - end
    ###################################

    en_loss = net.crossentropy(out_ul, out_ul)
    ce_loss = net.crossentropy(y, out)
    total_loss = ce_loss + args.alpha * vat_loss + args.beta * en_loss

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(out, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    weight_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net')
    train_step = tf.train.AdamOptimizer(lr).minimize(total_loss, var_list=weight_list)

    # summary
    train_summary_list = [tf.summary.scalar('loss/total', total_loss),
                          tf.summary.scalar('loss/crossentropy', ce_loss),
                          tf.summary.scalar('loss/entropy', en_loss),
                          tf.summary.scalar('loss/vat', vat_loss),
                          tf.summary.scalar('acc/train', accuracy)]
    _acc = tf.placeholder(tf.float32, name='acc_summary')
    test_summary_list = [tf.summary.scalar('acc/test', _acc)]
    train_summary_merged = tf.summary.merge(train_summary_list)
    test_summary_merged = tf.summary.merge(test_summary_list)

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
                mask_ul = np.random.choice(len(Xul_train), args.batch_size_ul, False)
                _, loss_ce, loss_vat, loss_en, train_summary = sess.run(
                    [train_step, ce_loss, vat_loss, en_loss, train_summary_merged],
                    feed_dict={x: X_train[mask], y: Y_train[mask], x_ul: Xul_train[mask_ul], lr: decayed_lr})
            acc = 0
            for j in range(20):
                acc += sess.run(accuracy,
                                feed_dict={x: X_test[500 * j:500 * (j + 1)], y: Y_test[500 * j:500 * (j + 1)]})
            acc /= 20
            test_summary = sess.run(test_summary_merged, feed_dict={_acc: acc})
            writer.add_summary(train_summary, ep)
            writer.add_summary(test_summary, ep)
            print('epoch', ep, 'ce', loss_ce, 'vat', loss_vat, 'ent', loss_en, 'acc', acc)
    print(args)


class Net(object):
    def __init__(self):
        super(Net, self).__init__()
        self.reuse = {}

    def classifier(self, x, name='net'):
        if name in self.reuse.keys():
            reuse = self.reuse[name]
        else:
            self.reuse[name] = True
            reuse = False

        with tf.variable_scope(name, reuse=reuse) as scope:
            w1 = tf.get_variable(name='w_conv1', shape=[3, 3, 1, 32], dtype=tf.float32)
            b1 = tf.get_variable(name='b_conv1', shape=[32], dtype=tf.float32)
            w2 = tf.get_variable(name='w_conv2', shape=[3, 3, 32, 32], dtype=tf.float32)
            b2 = tf.get_variable(name='b_conv2', shape=[32], dtype=tf.float32)
            w3 = tf.get_variable(name='w_conv3', shape=[3, 3, 32, 64], dtype=tf.float32)
            b3 = tf.get_variable(name='b_conv3', shape=[64], dtype=tf.float32)
            w4 = tf.get_variable(name='w_conv4', shape=[3, 3, 64, 64], dtype=tf.float32)
            b4 = tf.get_variable(name='b_conv4', shape=[64], dtype=tf.float32)
            w5 = tf.get_variable(name='w_fc1', shape=[49 * 64, 512], dtype=tf.float32)
            b5 = tf.get_variable(name='b_fc1', shape=[512], dtype=tf.float32)
            w6 = tf.get_variable(name='w_fc2', shape=[512, 10], dtype=tf.float32)
            b6 = tf.get_variable(name='b_fc2', shape=[10], dtype=tf.float32)

            x = tf.reshape(x, [-1, 28, 28, 1])
            x = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
            x = tf.nn.relu(x)
            x = tf.nn.conv2d(x, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            x = tf.nn.lrn(x)
            x = tf.nn.conv2d(x, w3, strides=[1, 1, 1, 1], padding='SAME') + b3
            x = tf.nn.relu(x)
            x = tf.nn.conv2d(x, w4, strides=[1, 1, 1, 1], padding='SAME') + b4
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            x = tf.nn.lrn(x)
            x = tf.reshape(x, [-1, 49 * 64])
            x = tf.matmul(x, w5) + b5
            x = tf.nn.relu(x)
            x = tf.matmul(x, w6) + b6
            x = tf.nn.softmax(x)
            return x

    def crossentropy(self, label, logits):
        return -tf.reduce_mean(tf.reduce_sum(label * tf.log(logits + 1e-8), axis=1))

    def kldivergence(self, label, logits):
        return tf.reduce_mean(tf.reduce_sum(label * (tf.log(label + 1e-8) - tf.log(logits + 1e-8)), axis=1))

    def kl_keepdims(self, label, logits):
        return tf.reduce_sum(label * (tf.log(label + 1e-8) - tf.log(logits + 1e-8)), axis=1, keepdims=True)


if __name__ == '__main__':
    main()
