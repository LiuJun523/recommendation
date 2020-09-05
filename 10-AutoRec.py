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

        user_dict = {}
        item_dict = {}
        def get_user_idx(user):
            if user not in user_dict.keys():
                user_dict[user] = len(user_dict)
            return user_dict[user]

        def get_item_idx(item):
            if item not in item_dict.keys():
                item_dict[item] = len(item_dict)
            return item_dict[item]

        def convert_dict(data_set):
            user_list, item_list, rate_list = [], [], []
            for user, item, rate in data_set:
                user_idx = get_user_idx(user)
                item_idx = get_item_idx(item)
                user_list.append(user_idx)
                item_list.append(item_idx)
                rate_list.append(rate)

            data_df = pd.DataFrame({
                'user': user_list,
                'item': item_list,
                'rate': rate_list
            })
            return data_df

        return convert_dict(train_set), convert_dict(test_set), user_dict, item_dict


class AutoRec:
    def __init__(self, params):
        self.num_items = params['num_items']
        self.hidden_size = params['hidden_size']
        self.lambda_value = params['lambda_value']
        self.learning_rate = params['learning_rate']
        self.model()

    def model(self):
        tf.reset_default_graph()

        # define placeholders
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name='X')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name='y')

        # forward propagation
        with tf.variable_scope('forward'):
            # define encoder
            V = tf.get_variable(name='V', initializer=tf.truncated_normal(shape=[self.num_items, self.hidden_size], stddev=0.02, mean=0))
            hidden_b = tf.get_variable(name='hidden_b', initializer=tf.zeros(shape=self.hidden_size), dtype=tf.float32)
            encoder = tf.sigmoid(tf.add(tf.matmul(self.X, V), hidden_b))
            # define decoder
            W = tf.get_variable(name='W', initializer=tf.truncated_normal(shape=[self.hidden_size, self.hidden_size], stddev=0.02, mean=0))
            output_b = tf.get_variable(name='output_b', initializer=tf.zeros(shape=self.num_items), dtype=tf.float32)
            self.y_out = tf.sigmoid(tf.add(tf.matmul(encoder, W), output_b))

        # loss
        with tf.variable_scope('loss'):
            rec_loss = tf.reduce_mean(tf.square(tf.subtract(self.y, self.y_out)))
            reg_loss = self.lambda_value * 0.5 * (tf.reduce_mean(tf.square(W)) + tf.reduce_mean(tf.square(V)))
            self.loss = rec_loss + reg_loss

        # train
        with tf.variable_scope('train'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 10000, 0.99, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_step = optimizer.minimize(self.loss, global_step=global_step)


if __name__ == '__main__':
    file_path = ''
    dataset = Dataset(file_path)
    train_df, test_df, user_dict, item_dict = dataset.load_dataset()

    params = {
        'num_items': len(item_dict),
        'hidden_size': 256,
        'lambda_value': 0.1,
        'batch_size': 100,
        'learning_rate': 0.01,
        'epochs': 10
    }
    batches_per_epoch = len(train_df) // params['batch_size']
    save_path = './autorec_model.ckpt'
    # initial model
    model = AutoRec(params)

    # train
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # train model
        for i in range(params['epochs']):
            num_samples = 0
            losses = []
            for j in range(batches_per_epoch):
                # sample batch training set
                batch_train = train_df.sample(n=params['batch_size'])
                batch_y = np.array(batch_train.pop('rate').values, dtype=np.float32)
                batch_x = np.array(batch_train.values, dtype=np.float32)
                loss, train_step = sess.run([model.loss, model.train_step],
                                            feed_dict={model.X: batch_x,
                                                       model.y: batch_y
                                                       })
                losses.append(loss)
                num_samples += len(batch_y)
                # print loss
                if j > 0 and j % 100 == 0:
                    print("Epoch %s of batch %s: loss = %s" % (i, j, loss))
                    saver.save(sess, save_path)

            # print total loss
            total_loss = np.sum(losses) / num_samples
            print("Epoch %s: overall loss = %s" % (i, total_loss))
