import numpy as np
import pandas as pd
from os import path

import tensorflow as tf


class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_dataset(self):
        train_df = pd.read_csv(path.join(self.file_path, 'train.csv'))
        test_df = pd.read_csv(path.join(self.file_path, 'test.csv'))

        # delete Id
        train_df.drop(['Id'], axis=1, inplace=True)
        test_df.drop(['Id'], axis=1, inplace=True)
        test_df['Label'] = -1

        data_df = pd.concat([train_df, test_df])
        data_df = data_df.fillna(-1)
        return data_df


class mlr:
    def __init__(self, params):
        self.num_features = params['num_features']
        self.num_separators = params['num_separators']
        self.learning_rate = params['learning_rate']
        self.decay_step = params['decay_step']
        self.decay_rate = params['decay_rate']
        self.model()

    def model(self):
        tf.reset_default_graph()

        # define placeholders
        self.X = tf.placeholder('float32', shape=[None, self.num_features])
        self.y = tf.placeholder('float32', shape=[None, ])

        # forward propagation
        with tf.variable_scope('forward'):
            seperator_w = tf.get_variable('separator_w', (self.num_features, self.num_separators), initializer=tf.random_normal_initializer())
            fitter_w = tf.get_variable('fitter_w', (self.num_features, self.num_separators), initializer=tf.random_normal_initializer())
            output_w = tf.get_variable('output_w', (self.num_separators * 2, 1), initializer=tf.random_normal_initializer())
            output_b = tf.get_variable('output_b', (1), initializer=tf.random_normal_initializer())

            seperator_tensor = tf.nn.softmax(tf.matmul(self.X, seperator_w))
            fitter_tensor = tf.sigmoid(tf.matmul(self.X, fitter_w))
            concat_tensor = tf.concat([fitter_tensor, seperator_tensor], axis=1)
            self.y_out = tf.sigmoid(tf.add(tf.matmul(concat_tensor, output_w), output_b))

        # loss
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.y, self.y_out)))

        # train
        with tf.variable_scope('train'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 10000, 0.99, staircase=True)
            optimizer = tf.train.AdagradOptimizer(learning_rate)
            self.train_step = optimizer.minimize(self.loss, global_step=global_step)


if __name__ == '__main__':
    file_path = ''
    dataset = Dataset(file_path)
    data_df = dataset.load_dataset()

    # one-hot
    for col in ['C' + str(i + 1) for i in range(26)]:
        onehot_features = pd.get_dummies(data_df[col], prefix=col)
        data_df.drop([col], axis=1, inplace=True)
        data_df = pd.concat([data_df, onehot_features], axis=1)

    train_df = data_df[data_df['Label'] != -1]
    test_df = data_df[data_df['Label'] == 1]
    test_df.drop(['Label'], axis=1, inplace=True)
    train_df = train_df.fillna(-1)
    test_df = test_df.fillna(-1)

    params = {
        'num_features': train_df.shape[1] - 1,
        'num_separators': 10,
        'batch_size': 100,
        'learning_rate': 0.01,
        'decay_step': 1000.0,
        'decay_rate': 0.99,
        'epochs': 1000
    }
    batches_per_epoch = len(train_df) // params['batch_size']
    save_path = './mlr_model.ckpt'
    # initial model
    model = mlr(params)

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
                batch_y = np.array(batch_train.pop('Label').values, dtype=np.float32)
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
