import os
import GPUtil
from datetime import datetime
import math
import argparse
import numpy as np
import tensorflow as tf
from cvae import cVAE
from vat import Net

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--iter-per-epoch', type=int, default=400)
parser.add_argument('--epoch', type=int, default=31)
parser.add_argument('--lr-decay-epoch', type=int, default=21)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--batch-size-ul', type=int, default=128)
parser.add_argument('--coef-vat1', type=float, default=1)
parser.add_argument('--coef-vat2', type=float, default=1)
parser.add_argument('--coef-ent', type=float, default=1)
parser.add_argument('--zeta', type=float, default=0.001)
parser.add_argument('--epsilon1', type=float, default=0.5)
parser.add_argument('--epsilon2', type=float, default=0.05)
parser.add_argument('--numX', type=int, default=200)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--latent-dim', type=int, default=100)
parser.add_argument('--logdir', type=str, default='./cvaeVATz_logs')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    log_dir = args.logdir+'/'+str(args.numX)+'/ep1st'+str(args.epsilon1)+'_ep2nd'+str(args.epsilon2)\
            +'_zeta'+str(args.zeta)+'_seed'+str(args.seed)+'_'+str(datetime.now())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    # set random seed
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # set gpu
    # deviceIDs = GPUtil.getAvailable(order='first', limit=4, maxLoad=0.1, maxMemory=0.1, excludeID=[], excludeUUID=[])
    deviceIDs = GPUtil.getAvailable()
    print('Unloaded gpu:', deviceIDs)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(deviceIDs[0])

    # dataset
    Dataset = np.load('./fashion.npz')
    Xul_train = Dataset['xtrain']
    Yul_train = Dataset['ytrain']
    X_test = Dataset['xtest']
    Y_test = Dataset['ytest']
    # mask = np.random.choice(60000, 500, False)
    mask = np.arange(0, args.numX)
    X_train = Xul_train[mask]
    Y_train = Yul_train[mask]
    for i in range(10):
        print('class:', i, np.sum((Y_train[:,i] == 1)))


    # build graph
    lr = tf.placeholder(tf.float32, [], name='lr')
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')
    x_ul = tf.placeholder(tf.float32, [None, 784], name='xul')

    # 炼cvae
    vae = cVAE(args.latent_dim)
    net = Net()
    out = net.classifier(x)
    out_ul = net.classifier(x_ul)
    mu, logvar = vae.encode(x_ul, out_ul)
    z = vae.reparamenterize(mu, logvar)
    x_recon = vae.decode(z, out_ul)
    x_gen = vae.decode(tf.random_normal([tf.shape(out_ul)[0],args.latent_dim]), out_ul)

    # conditional vae graph
    vae_loss = vae.BCE(x_recon, x_ul, mu, logvar) + vae.KLD(mu, logvar)
    vae_weight_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cvae')
    vae_train_step = tf.train.AdamOptimizer(lr).minimize(vae_loss, var_list=vae_weight_list)


    # TNAR graph
    # 计算 out_adv
    r0 = tf.zeros_like(z, name='zero_holder')  # tensor
    x_recon_r0 = vae.decode(z+r0, out_ul)
    diff2 = 0.5 * tf.reduce_sum((x_recon - x_recon_r0)**2, axis=1)
    diffJaco = tf.gradients(diff2, r0)[0]


    # 无穷范归一化 + 二范归一化
    def normalizevector(r):
        r /= (1e-12+tf.reduce_max(tf.abs(r), axis=1, keepdims=True))
        return r / tf.sqrt(tf.reduce_sum(r**2, axis=1, keepdims=True)+1e-6)

    # power method
    with tf.variable_scope('tangent_reg') as scope:
        r_adv = normalizevector(tf.random_normal(shape=tf.shape(z)))
        for j in range(1):
            # 这里是paper中的 F 对 \eta 求一阶导 （21式）
            r_adv = 1e-6*r_adv
            x_r = vae.decode(z+r_adv, out_ul)
            out_r = net.classifier(x_r-x_recon+x_ul) # ？？
            kl = net.kldivergence(out_ul, out_r)
            r_adv = tf.stop_gradient(tf.gradients(kl, r_adv)[0]) / 1e-6
            r_adv = normalizevector(r_adv) # 这时的r_adv 就是文章中的v

            # begin cg
            rk = r_adv + 0
            pk = rk + 0
            xk = tf.zeros_like(rk)
            for k in range(4):
                Bpk = tf.stop_gradient(tf.gradients(diffJaco*pk, r0)[0])
                pkBpk = tf.reduce_sum(pk*Bpk, axis=1, keepdims=True)
                rk2 = tf.reduce_sum(rk*rk, axis=1, keepdims=True)
                alphak = (rk2 / (pkBpk+1e-8)) * tf.cast((rk2>1e-8), tf.float32)
                xk += alphak * pk
                rk -= alphak * Bpk
                betak = tf.reduce_sum(rk*rk, axis=1, keepdims=True) / (rk2+1e-8)
                pk = rk + betak * pk
            # end cg, get xk = (JtJ)^(-1)v = \mu in paper
            r_adv = normalizevector(xk)

        x_adv = vae.decode(z+r_adv*args.epsilon1, out_ul)
        r_x = x_adv - x_recon
        out_adv = net.classifier(x_ul + r_x)
        r_x = normalizevector(r_x)


    with tf.variable_scope('normal_reg') as scope:
        # 计算 out_adv_orth
        r_adv_orth = normalizevector(tf.random_normal(shape=tf.shape(x_ul)))
        for j in range(1):
            r_adv_orth1 = 1e-6*r_adv_orth
            out_r = net.classifier(x_ul+r_adv_orth1)
            kl = net.kldivergence(out_ul, out_r)
            r_adv_orth1 = tf.stop_gradient(tf.gradients(kl, r_adv_orth1)[0]) / 1e-6
            r_adv_orth = r_adv_orth1 \
                         - args.zeta*(tf.reduce_sum(r_x*r_adv_orth,axis=1,keepdims=True)*r_x) \
                         + args.zeta*r_adv_orth
            r_adv_orth = normalizevector(r_adv_orth)
        out_adv_orth = net.classifier(x_ul+r_adv_orth*args.epsilon2)

    with tf.variable_scope('loss') as scope:
        # TNAR loss
        vat_loss = net.kldivergence(tf.stop_gradient(out_ul), out_adv)
        vat_loss_orth = net.kldivergence(tf.stop_gradient(out_ul), out_adv_orth)
        en_loss = net.crossentropy(out_ul, out_ul)
        ce_loss = net.crossentropy(y, out)
        total_loss = ce_loss + args.coef_vat1*vat_loss + args.coef_vat2*vat_loss_orth + args.coef_ent*en_loss

    weight_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net')
    train_step = tf.train.AdamOptimizer(lr).minimize(total_loss, var_list=weight_list)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(out,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # summary
    train_summary_list = [tf.summary.scalar('loss/total', total_loss),
                          tf.summary.scalar('loss/crossentropy', ce_loss),
                          tf.summary.scalar('loss/entropy', en_loss),
                          tf.summary.scalar('loss/vat', vat_loss),
                          tf.summary.scalar('acc/train', accuracy),]
    image_summary_list = [tf.summary.image('x', tf.reshape(x_ul, [-1,28,28,1]), max_outputs=32),
                          tf.summary.image('x_recon', tf.reshape(x_recon, [-1,28,28,1]), max_outputs=32),
                          tf.summary.image('x_gen', tf.reshape(x_gen, [-1,28,28,1]), max_outputs=32),
                          tf.summary.image('x_adv', tf.reshape(x_adv, [-1,28,28,1]), max_outputs=32),
                          tf.summary.image('r_adv', tf.reshape(x_adv-x_recon, [-1,28,28,1]), max_outputs=32),
                          tf.summary.image('r_adv_orth', tf.reshape(r_adv_orth*args.epsilon2, [-1,28,28,1]), max_outputs=32),
                          tf.summary.image('x_adv_orth', tf.reshape(x_ul+r_adv_orth*args.epsilon2, [-1,28,28,1]), max_outputs=32)]

    _acc = tf.placeholder(tf.float32, name='acc_summary')
    test_summary_list = [tf.summary.scalar('acc/test', _acc)]
    train_summary_merged = tf.summary.merge(train_summary_list+image_summary_list)
    test_summary_merged = tf.summary.merge(test_summary_list)

    saver = tf.train.Saver(max_to_keep=100)

    # optimization
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        # feed lr 是因为adam的初始化需要这样做
        sess.run(tf.global_variables_initializer(), feed_dict={lr:args.lr})
        for ep in range(args.epoch):
            if ep < args.lr_decay_epoch:
                decayed_lr = args.lr
            else:
                decayed_lr = args.lr * (args.epoch-ep)/float(args.epoch-args.lr_decay_epoch)

            for i in range(args.iter_per_epoch):
                mask = np.random.choice(len(X_train), args.batch_size, False)
                mask_ul = np.random.choice(len(Xul_train), args.batch_size_ul, False)
                # optimize cls
                # train的accuracy是用一个batch内部的点计算的
                _, loss_ce, loss_vat, loss_vat2, train_summary = sess.run([train_step, ce_loss, vat_loss, vat_loss_orth, train_summary_merged],
                        feed_dict={x: X_train[mask], y: Y_train[mask], x_ul: Xul_train[mask_ul], lr: decayed_lr, vae.is_train: False})
                # optimize vae
                _, loss_vae = sess.run([vae_train_step, vae_loss], feed_dict={x_ul: Xul_train[mask_ul], lr: decayed_lr, vae.is_train: True})

            # compute for the whole test set, 防止爆显存
            acc = 0
            for j in range(20):
                acc += sess.run(accuracy, feed_dict={x: X_test[500*j:500*(j+1)], y: Y_test[500*j:500*(j+1)]})
            acc /= 20

            test_summary = sess.run(test_summary_merged, feed_dict={x: X_test[0:128], y: Y_test[0:128], _acc:acc})
            writer.add_summary(train_summary, ep)
            writer.add_summary(test_summary, ep)
            print('epoch', ep, 'ce', loss_ce, 'vat1', loss_vat, 'vat2', loss_vat2, 'vae', loss_vae)
            print('epoch', ep, 'acc', acc)
            if ep % 10 == 0:
                saver.save(sess, os.path.join(log_dir, 'model'), ep)

    print(args)


