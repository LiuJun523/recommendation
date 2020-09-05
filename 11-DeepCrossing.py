import numpy as np
import pandas as pd
from os import path

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split


class DeepCrossing:
    def __init__(self, params, cate_list):
        self.num_cate = params['num_cate']
        self.num_cont = params['num_cont']
        self.num_residual = params['num_residual']
        self.embed_dim = params['embed_dim']
        self.residual_dim = params['residual_dim']
        self.lambda_value = params['lambda_value']
        self.dropout = params['dropout']
        self.learning_rate = params['learning_rate']
        self.l2_reg = tf.contrib.layers.l2_regularizer(self.lambda_value)
        self.model(cate_list)

    def residual_unit(self, x, i):
        in_dim = self.num_cate * self.embed_dim + self.num_cont
        out_dim = self.residual_dim
        w0 = tf.get_variable(name='residual_w0_%d' % i, shape=[in_dim, out_dim], dtype=tf.float32, regularizer=self.l2_reg)
        b0 = tf.get_variable(name='residual_b0_%d' % i, shape=[out_dim], dtype=tf.float32)
        residual = tf.nn.relu(tf.matmul(x, w0) + b0)
        w1 = tf.get_variable(name='residual_w1_%d' % i, shape=[out_dim, in_dim], dtype=tf.float32, regularizer=self.l2_reg)
        b1 = tf.get_variable(name='residual_b1_%d' % i, shape=[in_dim], dtype=tf.float32)
        residual = tf.matmul(residual, w1) + b1
        return tf.nn.relu(residual + x)

    def model(self, cate_list):
        tf.reset_default_graph()

        # define placeholders
        self.X_cate = tf.placeholder(tf.int32, shape=[None, self.num_cate])
        self.X_cont = tf.placeholder(tf.float32, shape=[None, self.num_cont])
        self.y = tf.placeholder(tf.float32, shape=[None, ])
        self.is_train = tf.placeholder(tf.bool)

        # Embedding
        with tf.variable_scope('embedding'):
            cate_embed = []
            for i in range(self.num_cate):
                in_dim = cate_list[i]
                out_dim = self.embed_dim
                embed = tf.get_variable(name='emb_%d' % i, shape=[in_dim, out_dim], dtype=tf.float32, regularizer=self.l2_reg)
                cate_embed.append(tf.nn.embedding_lookup(embed, self.X_cate[:, i]))
            input_embed = tf.concat(cate_embed, axis=1)
        # Stacking
        with tf.variable_scope('stacking'):
            input_embed = tf.concat([input_embed, self.X_cont], axis=1)
        # Multiple Residual Units
        with tf.variable_scope('residual'):
            x = input_embed
            for i in range(self.num_residual):
                x = self.residual_unit(x, i)
                x = tf.layers.dropout(x, rate=self.dropout, training=self.is_train)
            in_dim = self.num_cate * self.embed_dim + self.num_cont
            w = tf.get_variable(name='residual_w', shape=[in_dim, 1], dtype=tf.float32, regularizer=self.l2_reg)
            b = tf.get_variable(name='residual_b', shape=[1], dtype=tf.float32)
            self.y_out = tf.sigmoid(tf.matmul(x, w) + b)
            self.pred_label = tf.arg_max(self.y_out, 1)
        # Scoring
        with tf.variable_scope('scoring'):
            self.loss = -1 * tf.reduce_mean(self.y * tf.log(self.y_out + 1e-8) + (1 - self.y) * tf.log(1 - self.y_out + 1e-8))
            reg_variable = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if len(reg_variable) > 0:
                self.loss += tf.add_n(reg_variable)

        # train
        with tf.variable_scope('train'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 10000, 0.99, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_step = optimizer.minimize(self.loss, global_step=global_step)


def data_preprocessing(data):
    # category columns
    cate_columns = ['user_type', 'item_type']
    data_cate = data[cate_columns]
    # continus columns
    cont_columns = [col for col in data.columns if col not in cate_columns]
    cont_columns.remove('userid')
    cont_columns.remove('itemid')
    scaler = StandardScaler()
    data_cont = pd.DataFrame(StandardScaler().fit_transform(data[cont_columns]), columns=cont_columns)
    data = pd.concat([data_cate, data_cont], axis=1)
    # label
    label = data['label'].values
    cate_values = data[cate_columns].values
    cont_values = data[cont_columns].values
    return data, label, cate_values, cont_values


if __name__ == '__main__':
    # preprocessing
    file_path = ''
    raw = pd.read_csv(path.join(file_path, 'data.csv'))
    data, label, cate_values, cont_values = data_preprocessing(raw)

    cate_list = []
    for i in range(cate_values.shape[1]):
        nunique = np.unique(cate_values[:, i]).shape[0] + 1
        cate_list.append(nunique)

    params = {
        'num_cate': cate_values.shape[1],
        'num_cont': cont_values.shape[1],
        'num_residual': 2,
        'embed_dim': 10,
        'residual_dim': 256,
        'lambda_value': 0.01,
        'dropout': 0.2,
        'learning_rate': 0.01,
        'batch_size': 100,
        'epochs': 10
    }
    X_train_cate, X_test_cate, X_train_cont, X_test_cont, y_train, y_test = train_test_split(cate_values, cont_values, label, test_size=0.3, random_state=1024)
    batches_per_epoch = len(X_train_cate) // params['batch_size']
    save_path = './deep_crossing_model.ckpt'
    # initial model
    model = DeepCrossing(params, cate_list)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # train
        for i in range(params['epochs']):
            losses = []
            for j in range(batches_per_epoch):
                # sample batch training set
                start = j * params['batch_size']
                end = start + params['batch_size']
                loss, train_step = sess.run([model.loss, model.train_step],
                                            feed_dict={
                                                model.X_cate: X_train_cate[start:end],
                                                model.X_cont: X_train_cont[start:end],
                                                model.y: y_train[start:end],
                                                model.is_train: True})
                losses.append(loss)
                # print loss
                if j > 0 and j % 100 == 0:
                    print("Epoch %s of batch %s: loss = %s" % (i, j, loss))
                    saver.save(sess, save_path)
            print("Epoch %s: overall loss = %s" % (i, np.mean(losses)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # test
        y_pred = sess.run(model.pred_label, feed_dict={model.X_cate: X_test_cate,
                                                       model.X_cont: X_test_cont,
                                                       model.is_train: False})
