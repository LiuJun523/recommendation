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


class mf:
    def __init__(self, train_df, test_df, user2id, item2id):
        self.train_df = train_df
        self.test_df = test_df
        self.num_user = len(user2id)
        self.num_item = len(item2id)

        self.user_batch = tf.placeholder(tf.int32, shape=[None], name="user_batch")
        self.item_batch = tf.placeholder(tf.int32, shape=[None], name="item_batch")
        self.rate_batch = tf.placeholder(tf.float32, shape=[None], name="rate_batch")

    def model(self, dim=10, lr=0.01, reg=0.05):
        # Initialize the matrix factors from random normals with mean 0.
        # W represents users and H represents items.
        W = tf.Variable(tf.truncated_normal([self.num_user, dim], stddev=0.02, mean=0), name="user")
        H = tf.Variable(tf.truncated_normal([dim, self.num_item], stddev=0.02, mean=0), name="item")

        W_bias = tf.Variable(tf.truncated_normal([self.num_user, 1], stddev=0.02, mean=0), name="user_bias")
        H_bias = tf.Variable(tf.truncated_normal([1, self.num_item], stddev=0.02, mean=0), name="item_bias")

        # Add bias to the user matrix, and add another column of 1 to be multiplied by the item matrix
        W_plus_bias = tf.concat([W, W_bias, tf.ones((self.num_user, 1), dtype=tf.float32, name="item_bias_ones")], 1)
        # Add bias to the item matrix, and add another row of 1 to be multiplied by the user matrix
        H_plus_bias = tf.concat([H, tf.ones((1, self.num_item), name="user_bias_ones", dtype=tf.float32), H_bias], 0)

        # Regularization
        regularizer = tf.multiply(tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(H))), reg, name="regularizer")

        # Multiply the factors to get our result as a dense matrix
        result = tf.matmul(W_plus_bias, H_plus_bias)
        pred_rate = tf.gather(tf.reshape(result, [-1]), self.user_batch * tf.shape(result)[1] + self.item_batch, name="pred_rate")

        # Calculate the difference between the predicted rates and the actual rates
        diff = tf.subtract(pred_rate, self.rate_batch, name="diff")

        with tf.name_scope("cost"):
            base_cost = tf.reduce_sum(tf.square(diff), name="sum_squared_error")
            cost = tf.div(tf.add(base_cost, regularizer), tf.to_float(tf.shape(self.rate_batch)[0] * 2), name="average_error")

        with tf.name_scope("train"):
            # Use an exponentially decaying learning rate.
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.99, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_step = optimizer.minimize(cost, global_step=global_step)

        with tf.name_scope("rmse"):
            rmse = tf.sqrt(tf.reduce_sum(tf.square(diff)) / tf.to_float(tf.shape(self.rate_batch)[0]))

        return pred_rate, cost, train_step, rmse


if __name__ == '__main__':
    file_path = ''
    dataset = Dataset(file_path)
    train_df, test_df = dataset.load_dataset()

    dim = 10
    lr = 0.01
    regularization = 0.1
    epochs = 300
    batch_size = 1000
    save_path = './mf_model.ckpt'

    batches_per_epoch = len(train_df) // batch_size
    # reindex user and item
    users = sorted(set(list(train_df.user.unique()) + list(test_df.user.unique())))
    user2id = {users[i]: i for i in range(len(users))}
    items = sorted(set(list(train_df.item.unique()) + list(test_df.item.unique())))
    item2id = {items[i]: i for i in range(len(items))}

    mf_model = mf(train_df, test_df, user2id, item2id)
    pred_rate, cost, train_step, rmse = mf_model.model(dim, lr, regularization)

    # training
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            for j in range(batches_per_epoch):
                # sample batch training set
                batch_train = train_df.sample(n=batch_size)
                users, items, rates = (batch_train.user.values, batch_train.item.values, batch_train.rate.values)
                users = np.array([user2id[u] for u in users if u in user2id.keys()])
                items = np.array([item2id[i] for i in items if i in item2id.keys()])
                rates = np.array(rates, dtype=np.float32)

                # train
                if j > 0 and j % 100 == 0:
                    tr_rmse, tr_cost = sess.run([rmse, cost], feed_dict={mf_model.user_batch: users,
                                                                         mf_model.item_batch: items,
                                                                         mf_model.rate_batch: rates})
                    print("Training RMSE at epoch %s of batch %s: %s" % (i, j, tr_rmse))
                else:
                    sess.run(train_step, feed_dict={mf_model.user_batch: users,
                                                    mf_model.item_batch: items,
                                                    mf_model.rate_batch: rates})

        saver.save(sess, save_path)
        print("Train done, model saved at path: ", save_path)

    # prediction
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_path)

        users, items, rates = (test_df.user.values, test_df.item.values, test_df.rate.values)
        users = np.array([user2id[u] for u in users if u in user2id.keys()])
        items = np.array([item2id[i] for i in items if i in item2id.keys()])
        rates = np.array(rates, dtype=np.float32)

        pred_rate, _ = sess.run(pred_rate, feed_dict={mf_model.user_batch: users,
                                                      mf_model.item_batch: items,
                                                      mf_model.rate_batch: rates})

        print("Predict\Actual")
        for i in range(len(rates)):
            print("%.3f\t%.3f" % (pred_rate[i], rates[i]))
