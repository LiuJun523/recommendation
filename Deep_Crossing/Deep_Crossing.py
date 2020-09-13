import tensorflow as tf


class Deep_Crossing:
    def __init__(self, args, cate_num, cont_num, cate_list):
        self.cate_num = cate_num
        self.cont_num = cont_num
        self.residual_num = args.redisual_num
        self.embed_size = args.embed_size
        self.residual_size = args.residual_size
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

    def residual_unit(self, x, i):
        in_dim = self.cate_num * self.embed_size + self.cont_num
        out_dim = self.residual_size
        w0 = tf.get_variable(name='residual_w0_%d' % i, shape=[in_dim, out_dim], dtype=tf.float32, regularizer=self.regularizer)
        b0 = tf.get_variable(name='residual_b0_%d' % i, shape=[out_dim], dtype=tf.float32)
        residual = self.activation_func(tf.matmul(x, w0) + b0)
        w1 = tf.get_variable(name='residual_w1_%d' % i, shape=[out_dim, in_dim], dtype=tf.float32, regularizer=self.regularizer)
        b1 = tf.get_variable(name='residual_b1_%d' % i, shape=[in_dim], dtype=tf.float32)
        residual = tf.matmul(residual, w1) + b1
        return tf.nn.relu(residual + x)

    def define_model(self):
        # define placeholders
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
                embed_w = tf.get_variable(name='emb_w_%d' % i, shape=[in_dim, out_dim], dtype=tf.float32,
                                          regularizer=self.regularizer, initializer=self.initializer)
                embed_cate.append(tf.nn.embedding_lookup_sparse(embed_w, self.X_cate[:, i]))
            embed_output = tf.concat(embed_cate, axis=1)
        # stacking layer
        with tf.variable_scope('stacking'):
            input_embed = tf.concat([embed_output, self.X_cont], axis=1)
        # residual layer
        with tf.variable_scope('residual'):
            x = input_embed
            for i in range(self.redisual_num):
                x = self.residual_unit(x, i)
                x = tf.layers.dropout(x, rate=self.dropout, training=self.is_train)
            in_dim = self.cate_num * self.embed_size + self.cont_num
            w = tf.get_variable(name='residual_w', shape=[in_dim, 1], dtype=tf.float32,
                                regularizer=self.regularizer, initializer=self.initializer)
            b = tf.get_variable(name='residual_b', shape=[1], dtype=tf.float32, initializer=self.initializer)
            self.y_out = tf.sigmoid(tf.matmul(x, w) + b)
        # scoring layer
        with tf.variable_scope('scoring'):
            self.loss = -1 * tf.reduce_mean(self.y * tf.log(self.y_out + 1e-8) + (1 - self.y) * tf.log(1 - self.y_out + 1e-8))
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
        with tf.variable_scope('evaluation', reuse=tf.AUTO_REUSE):
            self.actual = self.y
            self.predict = tf.arg_max(self.y_out, 1)

    def summary(self):
        self.writer = tf.summary.FileWriter('./graphs/Deep_Crossing', tf.get_default_graph())
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
