import numpy as np
import pandas as pd
from os import path

import tensorflow as tf


class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_dataset(self):
        train_set = []
        with open(path.join(self.file_path, 'train_set.txt'), 'r') as f:
            lines = f.readlines()
            for l in lines:
                # read user_id, item_id, rate
                train_set.append(tuple(map(int, l.strip().split(','))))

        test_set = []
        with open(path.join(self.file_path, 'test_set.txt'), 'r') as f:
            lines = f.readlines()
            for l in lines:
                # read user_id, item_id, rate
                test_set.append(tuple(map(int, l.strip().split(','))))

        def convert_dataframe(data_set):
            user_list, item_list, rate_list = [], [], []
            for user, item, rate in data_set:
                user_list.append(user)
                item_list.append(item)
                rate_list.append(rate)

            data_df = pd.DataFrame({
                'user': user_list,
                'item': item_list,
                'rate': rate_list
            })

            return data_df

        return convert_dataframe(train_set), convert_dataframe(test_set)


class fm:
    def __init__(self, config):
        self.lr = config['lr']
        self.reg = config['reg']
        self.latent_factors = config['latent_factors']
        self.features = config['features']
        # build graph for model
        self.define_model()

    def model(self):
        # define placeholders
        self.X = tf.sparse_placeholder('float32', [None, self.features])
        self.y = tf.placeholder('int64', [None, ])
        self.keep_prob = tf.placeholder('float32')

        # forward propagation
        with tf.variable_scope('linear_terms'):
            b = tf.get_variable('bias', shape=[2], initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w1', shape=[self.features, 2], initializer=tf.truncated_normal_initializer(stddev=0.2))
            self.linear_terms = tf.add(tf.sparse_tensor_dense_matmul(self.X, w1), b)

        with tf.variable_scope('interaction_terms'):
            v = tf.get_variable('v', shape=[self.features, self.latent_factors], initializer=tf.truncated_normal_initializer(stddev=0.2))
            self.interaction_terms = tf.multiply(0.5,
                                                 tf.reduce_mean(
                                                     tf.subtract(
                                                         tf.pow(tf.sparse_tensor_dense_matmul(self.X, v), 2),
                                                         tf.sparse_tensor_dense_matmul(self.X, tf.pow(v, 2))),
                                                     1, keep_dims=True))
        self.y_out = tf.add(self.linear_terms, self.interaction_terms)
        self.y_out_prob = tf.nn.softmax(self.y_out)

        # loss
        with tf.variable_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
            error = tf.reduce_mean(cross_entropy)
            regularizer = tf.keras.regularizers.l2(l2=self.reg)
            self.loss = tf.add(error, regularizer)
            tf.summary.scalar('loss', self.loss)

        # accuracy
        with tf.variable_scope('accuracy'):
            self.correct_pre = tf.equal(tf.cast(tf.argmax(self.y_out, 1), tf.int64), self.y)
            self.accuracy = tf.reduce_mean(tf.case(self.correct_pre, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        # train
        with tf.name_scope('train'):
            self.global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(lr, self.global_step, 10000, 0.99, staircase=True)
            optimizer = tf.train.AdagradOptimizer(self.lr)
            self.train_step = optimizer.minimize(self.loss, global_step=self.global_step)


def check_restore_parameters(sess, saver):
    # restore previous parameters if exist
    ckpt = tf.train.get_checkpoint_state('checkpoints')
    if ckpt and ckpt.model_checkpoint_path:
        print('Loading parameters...')
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Initializing parameters...')


if __name__ == '__main__':
    file_path = ''
    dataset = Dataset(file_path)
    train_df, test_df = dataset.load_dataset()

    # initialize parameters
    config = {
        'lr': 0.01,
        'batch_size': 100,
        'epochs': 100,
        'reg': 0.01,
        'latent_factors': 40,
        'features': 2
    }
    batches_per_epoch = len(train_df) // config['batch_size']
    save_path = './mf_model.ckpt'
    # initialize model
    model = fm(config)

    # train
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # restore parameters
        check_restore_parameters(sess, saver)
        # train model
        for i in range(config['epochs']):
            num_samples = 0
            losses = []
            for j in range(batches_per_epoch):
                # sample batch training set
                batch_train = train_df.sample(n=config['batch_size'])
                users, items, rates = (batch_train.user.values, batch_train.item.values, batch_train.rate.values)
                batch_values = np.array([users, items], dtype=np.float32)
                batch_y = np.array(rates, dtype=np.int64)
                loss, accuracy, global_step, train_step = sess.run([model.loss, model.accuracy, model.global_step, model.train_step],
                                                                   feed_dict={
                                                                       model.X: batch_values,
                                                                       model.y: batch_y,
                                                                       model.keep_prob: 1.0})
                losses.append(loss)
                num_samples += len(batch_y)
                # print loss and accuracy
                if j > 0 and j % 100 == 0:
                    print("Epoch %s of batch %s: loss = %s, accuracy = %s" % (i, j, loss, accuracy))
                    saver.save(sess, save_path, global_step=global_step)

            # print total loss
            total_loss = np.sum(losses) / num_samples
            print("Epoch %s: overall loss = %s" % (i, total_loss))
