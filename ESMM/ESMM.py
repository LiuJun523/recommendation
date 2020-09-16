import tensorflow as tf


class ESMM:
    def __init__(self, args, cate_num, cont_num, cate_list):
        self.cate_num = cate_num
        self.cont_num = cont_num
        self.embed_size = args.embed_size
        self.deep_layers = args.deep_layers
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
        self.optimizer = tf.train.AdagradOptimizer

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
                onehot = tf.one_hot(self.X_cate[:, i], in_dim, dtype=tf.int32, name='onehot')
                embed = tf.nn.embedding_lookup(embed_w, onehot)
                value_cate = tf.cast(tf.reshape(onehot, shape=[-1, in_dim, 1]), dtype=tf.float32)
                embed_cate.append(tf.multiply(embed, value_cate))
            embed_output = tf.reduce_sum(tf.concat(embed_cate, axis=1), 2)
        # CTR deep part
        with tf.variable_scope('ctr_deep_part'):
            ctr_input = tf.concat([embed_output, self.X_cont], axis=1)
            w = tf.get_variable(name='ctr_deep_w_0', shape=[ctr_input.shape[1], self.deep_layers[0]], dtype=tf.float32,
                                regularizer=self.regularizer, initializer=self.initializer)
            b = tf.get_variable(name='ctr_deep_b_0', shape=[1, self.deep_layers[0]], dtype=tf.float32,
                                initializer=self.initializer)
            ctr_x = self.activation_func(tf.matmul(ctr_input, w) + b)
            ctr_x = tf.layers.dropout(ctr_x, rate=self.dropout, training=self.is_train)
            for i in range(1, len(self.deep_layers)):
                w = tf.get_variable(name='ctr_deep_w_%d' % i, shape=[self.deep_layers[i - 1], self.deep_layers[i]], dtype=tf.float32,
                                    regularizer=self.regularizer, initializer=self.initializer)
                b = tf.get_variable(name='ctr_deep_b_%d' % i, shape=[1, self.deep_layers[i]], dtype=tf.float32,
                                    initializer=self.initializer)
                ctr_x = self.activation_func(tf.matmul(ctr_x, w) + b)
                ctr_x = tf.layers.dropout(ctr_x, rate=self.dropout, training=self.is_train)
            w = tf.get_variable(name='ctr_output_w', shape=[self.deep_layers[-1], 1], dtype=tf.float32,
                                regularizer=self.regularizer, initializer=self.initializer)
            b = tf.get_variable(name='ctr_output_b', shape=[1, ], dtype=tf.float32,
                                initializer=self.initializer)
            ctr_output = tf.matmul(ctr_x, w) + b
        # CVR deep part
        with tf.variable_scope('cvr_deep_part'):
            cvr_input = tf.concat([embed_output, self.X_cont], axis=1)
            w = tf.get_variable(name='cvr_deep_w_0', shape=[cvr_input.shape[1], self.deep_layers[0]], dtype=tf.float32,
                                regularizer=self.regularizer, initializer=self.initializer)
            b = tf.get_variable(name='cvr_deep_b_0', shape=[1, self.deep_layers[0]], dtype=tf.float32,
                                initializer=self.initializer)
            cvr_x = self.activation_func(tf.matmul(cvr_input, w) + b)
            cvr_x = tf.layers.dropout(cvr_x, rate=self.dropout, training=self.is_train)
            for i in range(1, len(self.deep_layers)):
                w = tf.get_variable(name='cvr_deep_w_%d' % i, shape=[self.deep_layers[i - 1], self.deep_layers[i]], dtype=tf.float32,
                                    regularizer=self.regularizer, initializer=self.initializer)
                b = tf.get_variable(name='cvr_deep_b_%d' % i, shape=[1, self.deep_layers[i]], dtype=tf.float32,
                                    initializer=self.initializer)
                cvr_x = self.activation_func(tf.matmul(cvr_x, w) + b)
                cvr_x = tf.layers.dropout(cvr_x, rate=self.dropout, training=self.is_train)
            w = tf.get_variable(name='cvr_output_w', shape=[self.deep_layers[-1], 1], dtype=tf.float32,
                                regularizer=self.regularizer, initializer=self.initializer)
            b = tf.get_variable(name='cvr_output_b', shape=[1, ], dtype=tf.float32,
                                initializer=self.initializer)
            cvr_output = tf.matmul(cvr_x, w) + b
        # loss
        with tf.variable_scope('loss'):
            self.y_out = tf.reshape(tf.multiply(tf.sigmoid(ctr_output), tf.sigmoid(cvr_output)), shape=[-1, ])
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
        self.writer = tf.summary.FileWriter('./graphs/ESMM', tf.get_default_graph())
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
