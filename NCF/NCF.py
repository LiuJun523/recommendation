import tensorflow as tf


class NCF:
    def __init__(self, args, user_size, item_size):
        self.user_size = user_size
        self.item_size = item_size
        self.embed_size = args.embed_size
        self.regular_rate = args.regular_rate
        self.dropout = args.dropout
        self.learning_rate = args.learning_rate
        self.decay_steps = args.decay_steps
        self.decay_rate = args.decay_rate

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
        # input
        with tf.variable_scope('input'):
            self.X_user = tf.placeholder(tf.int32, shape=[None, ])
            self.X_item = tf.placeholder(tf.int32, shape=[None, ])
            self.y = tf.placeholder(tf.float32, shape=[None, ])
            self.is_train = tf.placeholder(tf.bool)
        # embedding
        with tf.variable_scope('embedding'):
            self.onehot_user = tf.one_hot(self.X_user, self.user_size, name='onehot_user')
            self.onehot_item = tf.one_hot(self.X_item, self.item_size, name='onehot_item')
            user_embed_GMF = tf.layers.dense(inputs=self.onehot_user,
                                             units=self.embed_size,
                                             activation=self.activation_func,
                                             kernel_initializer=self.initializer,
                                             kernel_regularizer=self.regularizer,
                                             name='user_embed_GMF')
            item_embed_GMF = tf.layers.dense(inputs=self.onehot_item,
                                             units=self.embed_size,
                                             activation=self.activation_func,
                                             kernel_initializer=self.initializer,
                                             kernel_regularizer=self.regularizer,
                                             name='item_embed_GMF')
            user_embed_MLP = tf.layers.dense(inputs=self.onehot_user,
                                             units=self.embed_size,
                                             activation=self.activation_func,
                                             kernel_initializer=self.initializer,
                                             kernel_regularizer=self.regularizer,
                                             name='user_embed_MLP')
            item_embed_MLP = tf.layers.dense(inputs=self.onehot_item,
                                             units=self.embed_size,
                                             activation=self.activation_func,
                                             kernel_initializer=self.initializer,
                                             kernel_regularizer=self.regularizer,
                                             name='item_embed_MLP')
        # GMF
        with tf.variable_scope('GMF_layer'):
            GMF_layer = tf.multiply(user_embed_GMF, item_embed_GMF, name='GMF_layer')
        # MLP
        with tf.variable_scope('MLP'):
            MLP_concat = tf.concat([user_embed_MLP, item_embed_MLP], axis=1, name='MLP_concat')
            MLP_layer1 = tf.layers.dense(inputs=MLP_concat,
                                         units=self.embed_size * 2,
                                         activation=self.activation_func,
                                         kernel_initializer=self.initializer,
                                         kernel_regularizer=self.regularizer,
                                         name='MLP_layer1')
            MLP_layer1 = tf.layers.dropout(MLP_layer1,
                                           rate=self.dropout,
                                           training=self.is_train)
            MLP_layer2 = tf.layers.dense(inputs=MLP_layer1,
                                         units=self.embed_size,
                                         activation=self.activation_func,
                                         kernel_initializer=self.initializer,
                                         kernel_regularizer=self.regularizer,
                                         name='MLP_layer2')
            MLP_layer2 = tf.layers.dropout(MLP_layer2,
                                           rate=self.dropout,
                                           training=self.is_train)
            MLP_layer3 = tf.layers.dense(inputs=MLP_layer2,
                                         units=self.embed_size // 2,
                                         activation=self.activation_func,
                                         kernel_initializer=self.initializer,
                                         kernel_regularizer=self.regularizer,
                                         name='MLP_layer3')
            MLP_layer3 = tf.layers.dropout(MLP_layer3,
                                           rate=self.dropout,
                                           training=self.is_train)
        # concat
        with tf.variable_scope('concat'):
            concat = tf.concat([GMF_layer, MLP_layer3], axis=-1, name='concat')
            logits = tf.layers.dense(inputs=concat,
                                     units=1,
                                     activation=self.activation_func,
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer,
                                     name='logits')
            self.y_out = tf.sigmoid(tf.reshape(logits, [-1]))
        # loss
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,
                                                                               logits=self.y_out,
                                                                               name='loss'))
        # optimization
        with tf.variable_scope('optim_op'):
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=self.decay_steps,
                                                       decay_rate=self.decay_rate,
                                                       staircase=True)
            self.optim_op = self.optimizer(learning_rate).minimize(self.loss, global_step=global_step)

    def evaluation(self):
        with tf.variable_scope('eval_op'):
            self.actual = self.y
            self.predict = tf.round(self.y_out)

    def summary(self):
        self.writer = tf.summary.FileWriter('./graphs/NCF', tf.get_default_graph())
        with tf.variable_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def run(self, session, X, y, is_train=False, step=None):
        if is_train:
            loss, optim_op, summary_op = session.run([self.loss, self.optim_op, self.summary_op],
                                                     feed_dict={
                                                         self.X_user: X[:, 0],
                                                         self.X_item: X[:, 1],
                                                         self.y: y,
                                                         self.is_train: is_train})
            self.writer.add_summary(summary_op, global_step=step)
            return loss
        else:
            actual, predict = session.run([self.actual, self.predict],
                                          feed_dict={
                                              self.X_user: X[:, 0],
                                              self.X_item: X[:, 1],
                                              self.y: y,
                                              self.is_train: is_train})
            return actual, predict
