import tensorflow as tf


class DCN:
    def __init__(self, args, cate_num, cont_num, cate_list):
        self.cate_num = cate_num
        self.cont_num = cont_num
        self.embed_size = args.embed_size
        self.cross_layer_num = args.cross_layer_num
        self.deep_layers = args.deep_layers
        self.regular_rate = args.regular_rate
        self.dropout = args.dropout
        self.learning_rate = args.learning_rate
        self.decay_steps = args.decay_steps
        self.decay_rate = args.decay_rate
        self.cate_list = cate_list
        self.total_size = np.sum(self.cate_list) * self.embed_size + self.cont_num

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
                onehot = tf.one_hot(self.X_cate[:, i], in_dim, dtype=tf.int32, name='onehot')
                embed = tf.nn.embedding_lookup(embed_w, onehot)
                value_cate = tf.cast(tf.reshape(onehot, shape=[-1, in_dim, 1]), dtype=tf.float32)
                embed_cate.append(tf.multiply(embed, value_cate))
            embed_cate = tf.concat(embed_cate, axis=1)
            embed_output = tf.concat([tf.reshape(embed_cate, shape=[-1, embed_cate.shape[1] * embed_cate.shape[2]]), self.X_cont], axis=1)
        # cross part
        with tf.variable_scope('cross_part'):
            cross_x0 = tf.reshape(embed_output, shape=[-1, self.total_size, 1])
            cross_xi = cross_x0
            for i in range(self.cross_layer_num):
                w = tf.get_variable(name='cross_w_%d' % i, shape=[self.total_size, 1], dtype=tf.float32,
                                    regularizer=self.regularizer, initializer=self.initializer)
                b = tf.get_variable(name='cross_b_%d' % i, shape=[self.total_size, 1], dtype=tf.float32,
                                    initializer=self.initializer)
                cross_xi = tf.tensordot(tf.matmul(cross_x0, cross_xi, transpose_b=True), w, 1) + b + cross_xi
            cross_output = tf.reshape(cross_xi, [-1, self.total_size])
        # deep part
        with tf.variable_scope('deep_part'):
            w = tf.get_variable(name='deep_w_0', shape=[self.total_size, self.deep_layers[0]], dtype=tf.float32,
                                regularizer=self.regularizer, initializer=self.initializer)
            b = tf.get_variable(name='deep_b_0', shape=[1, self.deep_layers[0]], dtype=tf.float32,
                                initializer=self.initializer)
            deep_x = self.activation_func(tf.matmul(embed_output, w) + b)
            deep_x = tf.layers.dropout(deep_x, rate=self.dropout, training=self.is_train)
            for i in range(1, len(self.deep_layers)):
                w = tf.get_variable(name='deep_w_%d' % i, shape=[self.deep_layers[i - 1], self.deep_layers[i]], dtype=tf.float32,
                                    regularizer=self.regularizer, initializer=self.initializer)
                b = tf.get_variable(name='deep_b_%d' % i, shape=[1, self.deep_layers[i]], dtype=tf.float32,
                                    initializer=self.initializer)
                deep_x = self.activation_func(tf.matmul(deep_x, w) + b)
                deep_x = tf.layers.dropout(deep_x, rate=self.dropout, training=self.is_train)
            deep_output = deep_x
        # stack layer
        with tf.variable_scope('stack_layer'):
            stack_input = tf.concat([cross_output, deep_output], axis=1)
            w = tf.get_variable(name='stack_w_%d' % i, shape=[stack_input.shape[1], 1], dtype=tf.float32,
                                regularizer=self.regularizer, initializer=self.initializer)
            b = tf.get_variable(name='stack_b_%d' % i, shape=[1, ], dtype=tf.float32,
                                initializer=self.initializer)
            self.y_out = tf.reshape(tf.sigmoid(tf.matmul(stack_input, w) + b), shape=[-1, ])
        # loss
        with tf.variable_scope('loss'):
            self.loss = tf.losses.log_loss(self.y, self.y_out)
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
        self.writer = tf.summary.FileWriter('./graphs/DCN', tf.get_default_graph())
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
