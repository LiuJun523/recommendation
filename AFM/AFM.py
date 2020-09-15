import tensorflow as tf


class AFM:
    def __init__(self, args, cate_num, cont_num, cate_list):
        self.cate_num = cate_num
        self.cont_num = cont_num
        self.embed_size = args.embed_size
        self.attention_size = args.attention_size
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
            embed_output = tf.concat(embed_cate, axis=1)

            in_dim = self.X_cont.shape[1]
            embed_w = tf.get_variable(name='embed_cont_w_%d' % i, shape=[in_dim, out_dim], dtype=tf.float32,
                                      regularizer=self.regularizer, initializer=self.initializer)
            value_cont = tf.reshape(self.X_cont, shape=[-1, in_dim, 1])
            embed_cont = tf.multiply(embed_w, value_cont)
            embed_output = tf.concat([embed_output, embed_cont], axis=1)

        # pair-wise interaction layer
        with tf.variable_scope('pair_wise_interaction'):
            element_wise_product_list = []
            for i in range(embed_output.shape[1]):
                for j in range(i + 1, embed_output.shape[1]):
                    element_wise_product_list.append(tf.multiply(embed_output[:, i, :], embed_output[:, j, :]))
            element_wise_product = tf.stack(element_wise_product_list)
            element_wise_product = tf.transpose(element_wise_product, perm=[1, 0, 2])
        # attention-based pooling layer
        with tf.variable_scope('attention_based_pooling'):
            num_interaction = int(embed_output.shape[1] * (embed_output.shape[1] - 1) // 2)
            w = tf.get_variable(name='attention_w', shape=[self.embed_size, self.attention_size], dtype=tf.float32,
                                regularizer=self.regularizer, initializer=self.initializer)
            b = tf.get_variable(name='attention_b', shape=[1, self.attention_size], dtype=tf.float32,
                                initializer=self.initializer)
            attention_mul = tf.matmul(tf.reshape(element_wise_product, shape=[-1, self.embed_size]), w) + b
            attention_mul = tf.reshape(attention_mul, shape=[-1, num_interaction, self.attention_size])
            attention_act = self.activation_func(attention_mul)
            h = tf.get_variable(name='attention_h', shape=[1, self.attention_size], dtype=tf.float32,
                                regularizer=self.regularizer, initializer=self.initializer)
            attention_output = tf.nn.softmax(tf.reduce_sum(tf.multiply(attention_act, h), axis=2, keep_dims=True))
            attention_output = tf.layers.dropout(attention_output, rate=self.dropout, training=self.is_train)
        # output
        with tf.variable_scope('output'):
            x = tf.reduce_sum(tf.multiply(attention_output, element_wise_product), 1)
            x = tf.layers.dropout(x, rate=self.dropout, training=self.is_train)
            self.y_out = tf.reshape(tf.reduce_sum(x, 1, keep_dims=True), shape=[-1, ])
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
        self.writer = tf.summary.FileWriter('./graphs/AFM', tf.get_default_graph())
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
