import numpy as np
import tensorflow as tf


class PNN:
    def __init__(self, args, cate_num, cont_num, cate_list):
        self.cate_num = cate_num
        self.cont_num = cont_num
        self.inner_product = args.inner_product
        self.embed_size = args.embed_size
        self.product_size = args.product_size
        self.hidden_layers = args.hidden_layers
        self.regular_rate = args.regular_rate
        self.dropout = args.dropout
        self.learning_rate = args.learning_rate
        self.decay_steps = args.decay_steps
        self.decay_rate = args.decay_rate
        self.cate_list = cate_list

    def build(self):
        self.initial()
        self.define_model()
        self.evaluation()
        self.summary()
        self.saver = tf.train.Saver(tf.global_variables())

    def initial(self):
        # Initial setting
        self.regularizer = tf.contrib.layers.l2_regularizer(self.regular_rate)
        self.initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        self.activation_func = tf.nn.relu
        self.optimizer = tf.train.AdamOptimizer

    def define_model(self):
        # define input
        with tf.variable_scope('input'):
            self.X_cate = tf.placeholder(tf.int32, shape=[None, self.cate_num])
            self.X_cont = tf.placeholder(tf.float32, shape=[None, self.cont_num])
            self.y = tf.placeholder(tf.float32, shape=[None, ])
            self.is_train = tf.placeholder(tf.bool)
        # embedding layer
        with tf.variable_scope('embedding'):
            embed_cate = []
            for i in range(self.cate_num):
                in_dim = self.cate_list[i]
                out_dim = self.embed_size
                embed_w = tf.get_variable(name='embed_cate_w_%d' % i, shape=[in_dim, out_dim], dtype=tf.float32,
                                          regularizer=self.regularizer, initializer=self.initializer)
                onehot = tf.one_hot(self.X_cate[:, i], in_dim, dtype=tf.int32)
                embed = tf.nn.embedding_lookup(embed_w, onehot)
                value_cate = tf.cast(tf.reshape(onehot, shape=[-1, in_dim, 1]), dtype=tf.float32)
                embed_cate.append(tf.multiply(embed, value_cate))
            embed_cate = tf.concat(embed_cate, axis=1)

            embed_w = tf.get_variable(name='embed_cont_w_%d' % i, shape=[self.X_cont.shape[1], out_dim], dtype=tf.float32,
                                      regularizer=self.regularizer, initializer=self.initializer)
            value_cont = tf.reshape(self.X_cont, shape=[-1, self.X_cont.shape[1], 1])
            embed_cont = tf.multiply(embed_w, value_cont)
            embed_output = tf.concat([embed_cate, embed_cont], axis=1)
        # product layer
        with tf.variable_scope('product'):
            # lz part
            lz = []
            in_dim = np.sum(self.cate_list) + self.cont_num
            out_dim = self.embed_size
            lz_w = tf.get_variable(name='lz_w', shape=[self.product_size, in_dim, out_dim], dtype=tf.float32,
                                   regularizer=self.regularizer, initializer=self.initializer)
            for i in range(self.product_size):
                theta = tf.multiply(embed_output, lz_w[i])
                lz.append(tf.reshape(tf.reduce_sum(theta, axis=[1, 2]), shape=[-1, 1]))
            lz_output = tf.concat(lz, axis=1)

            # lp part
            lp = []
            if self.inner_product:
                dim = np.sum(self.cate_list) + self.cont_num
                lp_inner_w = tf.get_variable(name='lp_inner_w', shape=[self.product_size, dim], dtype=tf.float32,
                                             regularizer=self.regularizer, initializer=self.initializer)
                for i in range(self.product_size):
                    theta = tf.multiply(embed_output, tf.reshape(lp_inner_w[i], shape=[1, -1, 1]))
                    lp.append(tf.reshape(tf.norm(tf.reduce_sum(theta, axis=1)), shape=[-1, 1]))
            else:
                embed_sum = tf.reduce_sum(embed_output, axis=1)
                embed_sum_expend = tf.matmul(tf.expand_dims(embed_sum, 2), tf.expand_dims(embed_sum, 1))
                dim = self.embed_size
                lp_outer_w = tf.get_variable(name='lp_outer_w', shape=[self.product_size, dim, dim], dtype=tf.float32,
                                             regularizer=self.regularizer, initializer=self.initializer)
                for i in range(self.product_size):
                    theta = tf.multiply(embed_sum_expend, tf.expand_dims(lp_outer_w[i], 0))
                    lp.append(tf.reshape(tf.reduce_sum(theta, axis=[1, 2]), shape=[-1, 1]))
            lp_output = tf.concat(lp, axis=1)

            bias = tf.get_variable(name='product_bias', shape=[self.product_size, ], dtype=tf.float32,
                                   initializer=self.initializer)
            product_output = self.activation_func(lz_output + lp_output + bias)
            product_output = tf.layers.dropout(product_output, rate=self.dropout, training=self.is_train)
        # hidden layer
        with tf.variable_scope('hidden'):
            w = tf.get_variable(name='hidden_w_0', shape=[self.product_size, self.hidden_layers[0]], dtype=tf.float32,
                                regularizer=self.regularizer, initializer=self.initializer)
            b = tf.get_variable(name='hidden_b_0', shape=[1, self.hidden_layers[0]], dtype=tf.float32,
                                initializer=self.initializer)
            x = self.activation_func(tf.matmul(product_output, w) + b)
            x = tf.layers.dropout(x, rate=self.dropout, training=self.is_train)
            for i in range(1, len(self.hidden_layers)):
                w = tf.get_variable(name='hidden_w_%d' % i, shape=[self.hidden_layers[i - 1], self.hidden_layers[i]], dtype=tf.float32,
                                    regularizer=self.regularizer, initializer=self.initializer)
                b = tf.get_variable(name='hidden_b_%d' % i, shape=[1, self.hidden_layers[i]], dtype=tf.float32,
                                    initializer=self.initializer)
                x = self.activation_func(tf.matmul(x, w) + b)
                x = tf.layers.dropout(x, rate=self.dropout, training=self.is_train)

            w = tf.get_variable(name='output_w', shape=[self.hidden_layers[-1], 1], dtype=tf.float32,
                                regularizer=self.regularizer, initializer=self.initializer)
            b = tf.get_variable(name='output_b', shape=[1, ], dtype=tf.float32,
                                initializer=self.initializer)
            self.y_out = tf.reshape(tf.sigmoid(tf.matmul(x, w) + b), shape=[-1, ])
        # loss
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,
                                                                               logits=self.y_out,
                                                                               name='loss'))
        # optimization
        with tf.variable_scope('optimization'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=self.decay_steps,
                                                       decay_rate=self.decay_rate,
                                                       staircase=True)
            self.optim_op = self.optimizer(learning_rate).minimize(self.loss, global_step=global_step)

    def evaluation(self):
        with tf.variable_scope('evaluation'):
            self.actual = self.y
            self.predict = tf.round(self.y_out)

    def summary(self):
        self.writer = tf.summary.FileWriter('./graphs/DeepCrossing', tf.get_default_graph())
        with tf.variable_scope('summary', reuse=tf.AUTO_REUSE):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def run(self, session, X_cate, X_cont, y, is_train=False, step=None):
        if is_train:
            loss, optim_op, summary_op = session.run([self.loss, self.optim_op, self.summary_op],
                                                     feed_dict={
                                                         self.X_cate: X_cate,
                                                         self.X_cont: X_cont,
                                                         self.y: y,
                                                         self.is_train: is_train})
            self.writer.add_summary(summary_op, global_step=step)
            return loss
        else:
            actual, predict = session.run([self.actual, self.predict],
                                          feed_dict={
                                              self.X_cate: X_cate,
                                              self.X_cont: X_cont,
                                              self.y: y,
                                              self.is_train: is_train})
            return actual, predict
