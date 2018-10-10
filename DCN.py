#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:  2018/10/10 10:36
# @Author:  jessetjiang
import tensorflow as tf

from sklearn.metrics import roc_auc_score


class DCN(object):
    """
    Deep and Cross (DCN) models
    """

    def __init__(self, feature_size, field_size, numeric_feature_size,
                 embedding_size=32, num_epochs=1, batch_size=256,
                 deep_layers=[32, 32, 32], deep_layer_activation=tf.nn.relu,
                 learning_rate=0.001, optimizer_type='adam', random_seed=2018,
                 loss_type='logloss', eval_metric=roc_auc_score, cross_layer_num=3,
                 l2_reg=0.001, dropout=[0.5, 0.5, 0.5],
                 verbose=False):
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"
        self.feature_szie = feature_size
        self.field_size = field_size
        self.numeric_feature_size = numeric_feature_size
        self.embedding_size = embedding_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.deep_layers = deep_layers
        self.deep_layer_activation = deep_layer_activation
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.cross_layer_num = cross_layer_num
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.verbose = verbose
        self.train_result, self.valid_result = [], []

    def get_data(self, filename, perform_shuffle=False):
        """
        parsing file and get data batch
        :param filename:
        :return:
        """
        print("Parsing ", filename)

        # 1 1:0.5 2:0.03519 3:1 4:0.02567 7:0.03708 8:0.01705 9:0.06296 10:0.18185 11:0.02497 12:1 14:0.02565...
        def decode_libsvm(line):
            """
            decode the libsvm format data
            :param line:
            :return:
            """
            columns = tf.string_split([line], ' ')
            # tf.string_split 通过分隔符分割source，结果是一个SparseTensor, indices, shape, values
            labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
            splits = tf.string_split(columns.values[1:], ':')
            id_vals = tf.reshape(splits.values, splits.dense_shape)
            feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
            feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
            feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
            return {'feat_ids': feat_ids, 'feat_vals': feat_vals}, labels

        dataset = tf.data.TextLineDataset(filename).map(decode_libsvm, num_parallel_calls=1).prefetch(50000)
        #  epochs
        dataset = dataset.repeat(self.num_epochs)
        # Batch size to use
        dataset = dataset.batch(self.batch_size)
        print(dataset)
        #  # Randomizes input using a window of 256 elements (read into memory)
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)

        iterator = dataset.make_one_shot_iterator()  # iterator
        return iterator
        # batch_features, batch_labels = iterator.get_next()
        # return batch_features, batch_labels

    def add_placeholders(self):
        self.X = tf.placeholder(tf.float32, [None, self.feature_szie])
        self.y = tf.placeholder(tf.int32, [None, ])
        # index of none zero features
        self.feature_inds = tf.placeholder(tf.int32, [None, self.field_size])

    def inference(self):
        """
        forward propagation
        :return:
        """
        self.graph = tf.Graph()

        # with self.graph.as_default():
        tf.set_random_seed(self.random_seed)

        with tf.variable_scope('DCN'):
            # build weights
            Cross_B = tf.get_variable(name="cross_b",
                                      shape=[self.cross_layer_num, self.field_size * self.embedding_size],
                                      initializer=tf.glorot_normal_initializer())
            Cross_W = tf.get_variable(name='cross_w',
                                      shape=[self.cross_layer_num, self.field_size * self.embedding_size],
                                      initializer=tf.glorot_normal_initializer())
            Feat_Emb = tf.get_variable(name='feat_emb', shape=[self.feature_szie, self.embedding_size],
                                       initializer=tf.glorot_normal_initializer())

            # build features
            feat_ids = self.feature_inds
            feat_vals = self.X

            # build f(x)
            with tf.variable_scope("Embedding-layyer"):

                embeddings = tf.nn.embedding_lookup(Feat_Emb, feat_ids)
                feat_vals = tf.reshape(feat_vals, shape=[-1, self.field_size, 1])
                embeddings = tf.multiply(embeddings, feat_vals)
                x0 = tf.reshape(embeddings, shape=[-1, self.field_size * self.embedding_size])

            with tf.variable_scope("Cross-Network"):
                xl = x0
                for l in range(self.cross_layer_num):
                    wl = tf.reshape(Cross_W[l], shape=[-1, 1])
                    xlw = tf.matmul(xl, wl)
                    xl = x0 * xlw + Cross_B[l]

            with tf.variable_scope("Deep-Network"):
                x_deep = x0
                for i in range(len(self.deep_layers)):
                    x_deep = tf.contrib.layers.fully_connected(inputs=x_deep, num_outputs=self.deep_layers[i],
                                                               weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                   self.l2_reg), scope="mlp_{0}".format(i))
                    x_deep = tf.nn.dropout(x_deep, keep_prob=self.dropout[i])

            with tf.variable_scope("DCN-Out"):
                # concat cross layer and deep layer
                x_stack = tf.concat([xl, x_deep], 1)
                y = tf.contrib.layers.fully_connected(inputs=x_stack, num_outputs=1, activation_fn=tf.identity,
                                                      weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                          self.l2_reg),
                                                      scope='out_layer')
                y = tf.reshape(y, shape=[-1])
                self.out = y

    def add_loss(self):
        if self.loss_type == 'logloss':
            self.out = tf.nn.sigmoid(self.out)
            self.loss = tf.losses.log_loss(self.y, self.out)
        elif self.loss_type == 'mse':
            self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

    def add_optimize(self):
        """
        bulid optimizer
        :return:
        """
        if self.optimizer_type == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99,
                                                    epsilon=1e-8).minimize(self.loss)
        elif self.optimizer_type == "adagrade":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=1e-8).minimize(self.loss)
        elif self.optimizer_type == "gd":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        elif self.optimizer_type == "moment":
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                self.loss)

    def add_accuracy(self):
        self.auc, self.auc_op = tf.metrics.auc(labels=self.y, predictions=self.out)
        self.correct_pred = tf.equal(tf.cast(self.out, tf.int32), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def build_graph(self):
        self.add_placeholders()
        self.inference()
        self.add_loss()
        self.add_optimize()
        self.add_accuracy()


if __name__ == '__main__':
    model = DCN(feature_size=117581, field_size=39, numeric_feature_size=13)
    filename = './output/va.libsvm'
    iterator = model.get_data(filename=filename)
    with tf.Session() as sess:
        try:
            while True:
                batch_features, batch_labels = iterator.get_next()
                bat, lab = sess.run([batch_features, batch_labels])
                print(type(bat))
                # print(bat)
                # print(batch_features, batch_labels)
        except tf.errors.OutOfRangeError:
            print("end!")
    model.build_graph()
    # model.inference()
