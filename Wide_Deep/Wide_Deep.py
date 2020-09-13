import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


class Wide_Deep:
    def __init__(self, args, cross_num, cate_num, cont_num, cross_list, cate_list):
        self.cross_num = cross_num
        self.cate_num = cate_num
        self.cont_num = cont_num
        self.embed_size = args.embed_size
        self.hidden_layers = args.hidden_layers
        self.regular_rate = args.regular_rate
        self.dropout = args.dropout
        self.learning_rate = args.learning_rate
        self.decay_steps = args.decay_steps
        self.decay_rate = args.decay_rate
        self.cross_list = cross_list
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

    def define_model(self):
        # define input
        with tf.variable_scope('input'):
            self.X_cross = tf.placeholder(tf.int32, shape=[None, self.cross_num])
            self.X_cate = tf.placeholder(tf.int32, shape=[None, self.cate_num])
            self.X_cont = tf.placeholder(tf.float32, shape=[None, self.cont_num])
            self.y = tf.placeholder(tf.float32, shape=[None, ])
            self.is_train = tf.placeholder(tf.bool)
        # wide net
        with tf.variable_scope('wide_net'):
            wide_embed = []
            for i in range(self.cross_num):
                in_dim = self.cross_list[i]
                out_dim = self.embed_size
                embed_w = tf.get_variable(name='wide_cross_w_%d' % i, shape=[in_dim, out_dim], dtype=tf.float32,
                                          regularizer=self.regularizer, initializer=self.initializer)
                b = tf.get_variable(name='wide_cross_b_%d' % i, shape=[out_dim], dtype=tf.float32,
                                    initializer=self.initializer)
                wide_embed.append(tf.nn.embedding_lookup(embed_w, self.X_cross[:, i]))
            for i in range(self.cate_num):
                in_dim = self.cate_list[i]
                out_dim = self.embed_size
                embed_w = tf.get_variable(name='wide_cate_w_%d' % i, shape=[in_dim, out_dim], dtype=tf.float32,
                                          regularizer=self.regularizer, initializer=self.initializer)
                wide_embed.append(tf.nn.embedding_lookup(embed_w, self.X_cate[:, i]))
            wide_input = tf.concat(wide_embed, axis=1)
            # output
            w = tf.get_variable(name='wide_output_w', shape=[wide_input.shape[1], 1], dtype=tf.float32,
                                regularizer=self.regularizer, initializer=self.initializer)
            b = tf.get_variable(name='wide_output_b', shape=[1], dtype=tf.float32,
                                initializer=self.initializer)
            wide_output = tf.matmul(wide_input, w) + b
        # deep net
        with tf.variable_scope('deep_net'):
            # embedding
            embed_cate = []
            for i in range(self.cate_num):
                in_dim = self.cate_list[i]
                out_dim = self.embed_size
                embed_w = tf.get_variable(name='deep_cate_w_%d' % i, shape=[in_dim, out_dim], dtype=tf.float32,
                                          regularizer=self.regularizer, initializer=self.initializer)
                embed_cate.append(tf.nn.embedding_lookup(embed_w, self.X_cate[:, i]))
            embed_output = tf.concat(embed_cate, axis=1)
            input_embed = tf.concat([embed_output, self.X_cont], axis=1)
            # hidden
            x = input_embed
            for i in range(len(self.hidden_layers)):
                w = tf.get_variable(name='deep_hidden_w_%d' % i, shape=[x.shape[1], self.hidden_layers[i]], dtype=tf.float32,
                                    regularizer=self.regularizer, initializer=self.initializer)
                b = tf.get_variable(name='deep_hidden_b_%d' % i, shape=[1, self.hidden_layers[i]], dtype=tf.float32,
                                    initializer=self.initializer)
                x = self.activation_func(tf.matmul(x, w) + b)
                x = tf.layers.dropout(x, rate=self.dropout, training=self.is_train)
            # output
            w = tf.get_variable(name='deep_output_w', shape=[self.hidden_layers[-1], 1], dtype=tf.float32,
                                regularizer=self.regularizer, initializer=self.initializer)
            b = tf.get_variable(name='deep_output_b', shape=[1, ], dtype=tf.float32,
                                initializer=self.initializer)
            deep_output = tf.matmul(x, w) + b
        # loss
        with tf.variable_scope('loss'):
            self.y_out = tf.reshape(tf.sigmoid(wide_output + deep_output), shape=[-1, ])
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
            # wide part - FTRL + L1
            wide_op = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                             l1_regularization_strength=0.1)
            wide_optim_op = wide_op.minimize(loss=self.loss,
                                             global_step=global_step,
                                             var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'wide_net'))
            # deep part - Adagrad
            deep_op = tf.train.AdagradOptimizer(learning_rate)
            deep_optim_op = deep_op.minimize(loss=self.loss,
                                             global_step=global_step,
                                             var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'deep_net'))
            self.optim_op = control_flow_ops.group([wide_optim_op, deep_optim_op])

    def evaluation(self):
        with tf.variable_scope('evaluation'):
            self.actual = self.y
            self.predict = tf.round(self.y_out)

    def summary(self):
        self.writer = tf.summary.FileWriter('./graphs/Wide_Deep', tf.get_default_graph())
        with tf.variable_scope('summary', reuse=tf.AUTO_REUSE):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def run(self, session, X_cross, X_cate, X_cont, y, is_train=False, step=None):
        if is_train:
            loss, optim_op, summary_op = session.run([self.loss, self.optim_op, self.summary_op],
                                                     feed_dict={
                                                         self.X_cross: X_cross,
                                                         self.X_cate: X_cate,
                                                         self.X_cont: X_cont,
                                                         self.y: y,
                                                         self.is_train: is_train})
            self.writer.add_summary(summary_op, global_step=step)
            return loss
        else:
            actual, predict = session.run([self.actual, self.predict],
                                          feed_dict={
                                              self.X_cross: X_cross,
                                              self.X_cate: X_cate,
                                              self.X_cont: X_cont,
                                              self.y: y,
                                              self.is_train: is_train})
            return actual, predict
