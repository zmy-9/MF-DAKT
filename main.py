import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
import argparse
from data_loader import Data_loader
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import logging
from sklearn.metrics import mean_squared_error


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MF-DAKT.")
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='Pre-train flag. 0: train from scratch; 1: load from pretrain file')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=256,
                        help='Number of hidden factors.')
    parser.add_argument('--layers', nargs='?', default='[128]',
                        help="Size of each layer.")
    parser.add_argument('--keep_prob', nargs='?', default='[0.3,0.3]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--diff_layers', nargs='?', default='[256]',
                        help="Size of each layer.")
    parser.add_argument('--keep_diff', nargs='?', default='[0.3]',
                        help='Keep probability (i.e., 1-dropout_ratio) for')
    parser.add_argument('--fenlayers', nargs='?', default='[256]',
                        help="Size of FEN layer.")
    parser.add_argument('--keep_attention', nargs='?', default='[0.4,0.3]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each dee')
    parser.add_argument('--fenlayersv', nargs='?', default='[256]',
                        help="Size of FEN layer.")
    parser.add_argument('--keep_attentionv', nargs='?', default='[0.4,0.3]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each dee')
    parser.add_argument('--deep_layers', nargs='?', default='[32]',
                        help="Size of each layer.")
    parser.add_argument('--keep_deep', nargs='?', default='[0.4,0.2]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer')
    parser.add_argument('--deep_layers_v', nargs='?', default='[32]',
                        help="Size of each layer.")
    parser.add_argument('--keep_deep_v', nargs='?', default='[0.4]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer')
    parser.add_argument('--lamda', type=float, default=0.1,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu',
                        help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--early_stop', type=int, default=0,
                        help='Whether to perform early stop (0 or 1)')
    parser.add_argument('--net_channel', nargs='?', default='[256,256]',
                        help='net_channel, should be 2 layers here')

    return parser.parse_args()


class DAKT(BaseEstimator, TransformerMixin):
    def __init__(self, features_user, features_item, features_skills, hidden_factor, loss_type, pretrain_flag, epoch,
                 batch_size, learning_rate,
                 lamda_bilinear, keep_prob, optimizer_type, batch_norm, activation_function, verbose, early_stop,
                 layers, random_seed=2016):
        # bind params to class
        self.batch_size = batch_size
        self.hidden_factor = hidden_factor
        self.loss_type = loss_type
        self.pretrain_flag = pretrain_flag
        self.features_user = features_user
        self.features_item = features_item
        self.features_skills = features_skills
        self.lamda_bilinear = lamda_bilinear
        self.epoch = epoch
        self.random_seed = random_seed

        ## neurons which transforms the interaction vectors into difficulty
        self.layers = layers
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for i in range(len(keep_prob))])

        ## neurons which transforms the feature vectors into difficulty
        self.diff_layers = np.array(eval(args.diff_layers))
        self.keep_diff = np.array(eval(args.keep_diff))
        self.no_dropout_diff = np.array([1 for i in range(len(eval(args.keep_diff)))])

        ## attention settings for feature vectors
        self.atten_layers = np.array(eval(args.fenlayers))
        self.keep_attention = np.array(eval(args.keep_attention))
        self.no_atten_dropout = np.array([1 for i in range(len(eval(args.keep_attention)))])

        ## deep networks settings for feature vectorst
        self.deep_layers = np.array(eval(args.deep_layers))
        self.keep_deep = np.array(eval(args.keep_deep))
        self.no_deep_dropout = np.array([1 for i in range(len(eval(args.keep_deep)))])

        ## attention settings for interaction vectors
        self.atten_layers_v = np.array(eval(args.fenlayersv))
        self.keep_attention_v = np.array(eval(args.keep_attentionv))
        self.no_atten_dropout_v = np.array([1 for i in range(len(eval(args.keep_attentionv)))])

        ## deep networks settings for interaction vectorst
        self.deep_layers_v = np.array(eval(args.deep_layers_v))
        self.keep_deep_v = np.array(eval(args.keep_deep_v))
        self.no_deep_dropout_v = np.array([1 for i in range(len(eval(args.keep_deep_v)))])

        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.activation_function = activation_function
        self.early_stop = early_stop
        self.num_field = 4

        # performance of each epoch
        self.train_acc, self.test_acc, self.train_auc, self.test_auc = [], [], [], []

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.user_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.item_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.skill_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.skill_nums = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M
            self.wins_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.wins_nums = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M
            self.fails_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.fails_nums = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M
            self.last_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.last_nums = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M

            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.difficulty_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.dropout_keep_attention = tf.placeholder(tf.float32, shape=[None])
            self.dropout_keep_attention_v = tf.placeholder(tf.float32, shape=[None])
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None])
            self.dropout_keep_deep_v = tf.placeholder(tf.float32, shape=[None])
            self.dropout_keep_diff = tf.placeholder(tf.float32, shape=[None])
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights, self.weights2, self.weights3 = self._initialize_weights()

            # Model.
            # _________ sum_square part _____________
            # get the summed up embeddings of features.
            nonzero_embeddings_user = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.user_features)
            nonzero_embeddings_item = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.item_features)
            nonzero_embeddings_skill = tf.multiply(
                tf.nn.embedding_lookup(self.weights['skill_embeddings'], self.skill_features),
                tf.expand_dims(self.skill_nums, 2))
            nonzero_embeddings_wins = tf.multiply(
                tf.nn.embedding_lookup(self.weights['wins_embeddings'], self.wins_features),
                tf.expand_dims(self.wins_nums, 2))
            nonzero_embeddings_fails = tf.multiply(
                tf.nn.embedding_lookup(self.weights['fails_embeddings'], self.fails_features),
                tf.expand_dims(self.fails_nums, 2))
            nonzero_embeddings_last = tf.multiply(
                tf.nn.embedding_lookup(self.weights['last_embeddings'], self.last_features),
                tf.expand_dims(self.last_nums, 2))

            self.skill_nums_count = tf.expand_dims(tf.reduce_sum(self.skill_nums, 1, keepdims=True), 2)

            nonzero_embeddings_skill = tf.reduce_sum(nonzero_embeddings_skill, 1, keepdims=True)
            nonzero_embeddings_skill_mean = tf.div(nonzero_embeddings_skill, self.skill_nums_count)

            nonzero_embeddings_wins = tf.reduce_sum(nonzero_embeddings_wins, 1, keepdims=True)
            nonzero_embeddings_fails = tf.reduce_sum(nonzero_embeddings_fails, 1, keepdims=True)

            self.skill_nums_count_last = tf.expand_dims(tf.reduce_sum(self.last_nums, 1, keepdims=True), 2)
            nonzero_embeddings_last = tf.reduce_sum(nonzero_embeddings_last, 1, keepdims=True)
            nonzero_embeddings_last_mean = tf.div(nonzero_embeddings_last, self.skill_nums_count_last)

            ## difficulty component
            self.item = nonzero_embeddings_item
            self.feature = self.item
            self.diff = tf.reshape(self.feature, [-1, self.hidden_factor])
            # ________ Deep Layers __________
            for i in range(0, len(self.layers)):
                self.diff = tf.add(tf.matmul(self.diff, self.weights2['layer_%d' % i]),
                                   self.weights2['bias_%d' % i])  # None * layer[i] * 1

                if self.batch_norm:
                    self.diff = self.batch_norm_layer(self.diff, train_phase=self.train_phase,
                                                      scope_bn='bn_diff%d' % i)  # None * layer[i] * 1

                self.diff = self.activation_function(self.diff)
                self.diff = tf.nn.dropout(self.diff, self.dropout_keep[i])  # dropout at each Deep layer
            self.difficulty = tf.matmul(self.diff, self.weights2['prediction_diff'])  # None * 1
            self.mse = tf.reduce_mean(tf.square(self.difficulty - self.difficulty_labels))

            # conv1d attempt embedding operation
            attempt_embeddings = tf.concat([nonzero_embeddings_wins, nonzero_embeddings_fails], 1)
            attempt_embeddings = tf.concat([attempt_embeddings, nonzero_embeddings_last_mean], 1)
            conv_embeddings = tf.layers.conv1d(attempt_embeddings, filters=256, kernel_size=2, strides=1, padding='VALID')
            conv_embeddings = tf.nn.relu(conv_embeddings)
            conv_embeddings = tf.layers.conv1d(conv_embeddings, filters=256, kernel_size=2, strides=1, padding='VALID')
            conv_embeddings = tf.nn.relu(conv_embeddings)

            nonzero_embeddings = tf.concat([nonzero_embeddings_user, nonzero_embeddings_item], 1)
            nonzero_embeddings = tf.concat([nonzero_embeddings, nonzero_embeddings_skill_mean], 1)
            nonzero_embeddings = tf.concat([nonzero_embeddings, conv_embeddings], 1)

            ## attention component
            dnn_nonzero_embeddings_v = tf.reshape(nonzero_embeddings,
                                                shape=[-1, self.num_field * self.hidden_factor])
            self.dnn_v = tf.add(tf.matmul(dnn_nonzero_embeddings_v, self.weights3['fenlayer_v_0']),
                              self.weights3['fenbias_v_0'])  # None * layer[i] * 1
            if self.batch_norm:
                self.dnn_v = self.batch_norm_layer(self.dnn_v, train_phase=self.train_phase,
                                                 scope_bn='bn_v_0')  # None * layer[i] * 1
            self.dnn_v = tf.nn.relu(self.dnn_v)
            self.dnn_v = tf.nn.dropout(self.dnn_v, self.dropout_keep_attention_v[0])  # dropout at each Deep layer

            for i in range(1, len(self.atten_layers_v)):
                self.dnn_v = tf.add(tf.matmul(self.dnn_v, self.weights3['fenlayer_v_%d' % i]),
                                  self.weights3['fenbias_v_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.dnn_v = self.batch_norm_layer(self.dnn_v, train_phase=self.train_phase,
                                                     scope_bn='bn_v_%d' % i)  # None * layer[i] * 1
                self.dnn_v = tf.nn.relu(self.dnn_v)
                self.dnn_v = tf.nn.dropout(self.dnn_v, self.dropout_keep_attention_v[i])  # dropout at each Deep layer
            self.dnn_out_v = tf.matmul(self.dnn_v, self.weights3['prediction_attention_v'])  # None * 10
            self.outm_v = tf.constant(float(5)) * tf.nn.softmax(self.dnn_out_v)
            self.nonzero_embeddings = tf.multiply(nonzero_embeddings, tf.expand_dims(self.outm_v, 2))

            # second-order interaction
            self.summed_features_emb = tf.reduce_sum(self.nonzero_embeddings, 1)  # None * K
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            self.squared_features_emb = tf.square(self.nonzero_embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # ________ FM __________
            self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                scope_bn='bn_fm')  # None * layer[i] * 1
            self.FM_v = tf.nn.dropout(self.FM, self.dropout_keep[-1])  # dropout at each Deep layer

            ## DNN component
            for i in range(0, len(self.deep_layers)):
                self.FM_v = tf.add(tf.matmul(self.FM_v, self.weights3['layer_v_%d' % i]),
                                 self.weights3['bias_v_%d' % i])  # None * layer[i] * 1

                if self.batch_norm:
                    self.FM_v = self.batch_norm_layer(self.FM_v, train_phase=self.train_phase,
                                                    scope_bn='bn_v_deep%d' % i)  # None * layer[i] * 1

                self.FM_v = self.activation_function(self.FM_v)
                self.FM_v = tf.nn.dropout(self.FM_v, self.dropout_keep_deep_v[i])  # dropout at each Deep layer
            self.FM_v = tf.matmul(self.FM_v, self.weights3['prediction_dnn_v'])  # None * 1
            # _________out _________
            Bilinear = tf.reduce_sum(self.FM_v, 1, keep_dims=True)  # None * 1

            # __________________特征向量部分________________________________
            self.Feature_bias_user = tf.nn.embedding_lookup(self.weights['user_bias'], self.user_features)
            self.Feature_bias_item = tf.nn.embedding_lookup(self.weights['item_bias'], self.item_features)
            self.Feature_bias_skill = tf.reduce_sum(
                tf.multiply(tf.nn.embedding_lookup(self.weights['skill_bias'], self.skill_features),
                            tf.expand_dims(self.skill_nums, 2)), 1, keepdims=True)
            self.Feature_bias_wins = tf.reduce_sum(
                tf.multiply(tf.nn.embedding_lookup(self.weights['wins_bias'], self.wins_features),
                            tf.expand_dims(self.wins_nums, 2)), 1, keepdims=True)
            self.Feature_bias_fails = tf.reduce_sum(
                tf.multiply(tf.nn.embedding_lookup(self.weights['fails_bias'], self.fails_features),
                            tf.expand_dims(self.fails_nums, 2)), 1, keepdims=True)
            self.Feature_bias_last = tf.reduce_sum(
                tf.multiply(tf.nn.embedding_lookup(self.weights['last_bias'], self.last_features),
                            tf.expand_dims(self.last_nums, 2)), 1, keepdims=True)

            self.mean_skill_bias = tf.div(self.Feature_bias_skill, self.skill_nums_count)
            self.mean_last_bias = tf.div(self.Feature_bias_last, self.skill_nums_count_last)

            ## difficulty component
            self.item_bias = self.Feature_bias_item
            self.feature_bias = self.item_bias
            self.diff_bias = tf.reshape(self.feature_bias, [-1, self.hidden_factor])
            # ________ Deep Layers __________
            for i in range(0, len(self.diff_layers)):
                self.diff_bias = tf.add(tf.matmul(self.diff_bias, self.weights2['diff_layer_%d' % i]),
                                        self.weights2['diff_bias_%d' % i])  # None * layer[i] * 1

                if self.batch_norm:
                    self.diff_bias = self.batch_norm_layer(self.diff_bias, train_phase=self.train_phase,
                                                           scope_bn='bn_diff_bias%d' % i)  # None * layer[i] * 1

                self.diff_bias = self.activation_function(self.diff_bias)
                self.diff_bias = tf.nn.dropout(self.diff_bias, self.dropout_keep_diff[i])  # dropout at each Deep layer
            self.difficulty_bias = tf.matmul(self.diff_bias, self.weights2['prediction_diff_bias'])  # None * 1
            self.mse_bias = tf.reduce_mean(tf.square(self.difficulty_bias - self.difficulty_labels))

            # conv1d attempt embedding operation
            attempt_bias = tf.concat([self.Feature_bias_wins, self.Feature_bias_fails], 1)
            attempt_bias = tf.concat([attempt_bias, self.mean_last_bias], 1)
            conv_bias = tf.layers.conv1d(attempt_bias, filters=256, kernel_size=2, strides=1, padding='VALID')
            conv_bias = tf.nn.relu(conv_bias)
            conv_bias = tf.layers.conv1d(conv_bias, filters=256, kernel_size=2, strides=1, padding='VALID')
            conv_bias = tf.nn.relu(conv_bias)

            self.bias = tf.concat([self.Feature_bias_user, self.Feature_bias_item], 1)
            self.bias = tf.concat([self.bias, self.mean_skill_bias], 1)
            self.bias = tf.concat([self.bias, conv_bias], 1)

            ## attention component
            dnn_nonzero_embeddings = tf.reshape(self.bias,
                                                shape=[-1, self.num_field * self.hidden_factor])
            self.dnn = tf.add(tf.matmul(dnn_nonzero_embeddings, self.weights3['fenlayer_0']),
                              self.weights3['fenbias_0'])  # None * layer[i] * 1
            if self.batch_norm:
                self.dnn = self.batch_norm_layer(self.dnn, train_phase=self.train_phase,
                                                 scope_bn='bn_0')  # None * layer[i] * 1
            self.dnn = tf.nn.relu(self.dnn)
            self.dnn = tf.nn.dropout(self.dnn, self.dropout_keep_attention[0])  # dropout at each Deep layer

            for i in range(1, len(self.atten_layers)):
                self.dnn = tf.add(tf.matmul(self.dnn, self.weights3['fenlayer_%d' % i]),
                                  self.weights3['fenbias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.dnn = self.batch_norm_layer(self.dnn, train_phase=self.train_phase,
                                                     scope_bn='bn_%d' % i)  # None * layer[i] * 1
                self.dnn = tf.nn.relu(self.dnn)
                self.dnn = tf.nn.dropout(self.dnn, self.dropout_keep_attention[i])  # dropout at each Deep layer
            self.dnn_out = tf.matmul(self.dnn, self.weights3['prediction_attention'])  # None * 10
            self.outm = tf.constant(float(5)) * tf.nn.softmax(self.dnn_out)

            ## DNN component
            self.nonzero_embeddings_m = tf.multiply(self.bias, tf.expand_dims(self.outm, 2))
            self.bias = tf.reduce_sum(self.nonzero_embeddings_m, 1)
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.bias, train_phase=self.train_phase, scope_bn='bn_bias')

            self.FM = tf.nn.dropout(self.FM, self.dropout_keep_deep[-1])  # dropout at the bilinear interactin layer
            for i in range(0, len(self.deep_layers)):
                self.FM = tf.add(tf.matmul(self.FM, self.weights3['layer_%d' % i]),
                                 self.weights3['bias_%d' % i])  # None * layer[i] * 1

                if self.batch_norm:
                    self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                                    scope_bn='bn_deep%d' % i)  # None * layer[i] * 1

                self.FM = self.activation_function(self.FM)
                self.FM = tf.nn.dropout(self.FM, self.dropout_keep_deep[i])  # dropout at each Deep layer
            self.FM = tf.matmul(self.FM, self.weights3['prediction_dnn'])  # None * 1
            self.Bilinear_bias = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1

            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out = tf.add_n([Bilinear, self.Bilinear_bias, Bias])  # None * 1

            # Compute the loss.
            if self.lamda_bilinear > 0:
                self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels)+ tf.add_n(
                    [tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights3[v]) for v in
                     self.weights3])
            else:
                self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels)
            self.out = tf.nn.sigmoid(self.out)
            self.total_loss = tf.reduce_mean(self.loss) + self.mse + self.mse_bias
            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.total_loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.total_loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.total_loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _conv_weight(self, deep, isz, osz):
        return self.weight_variable([deep, isz, osz]), self.bias_variable([osz])

    def _initialize_weights(self):
        all_weights = dict()
        weights1 = dict()
        weights2 = dict()
        if self.pretrain_flag > 0:  # with pretrain
            pretrain_file = 'assist2009/pre_train/problem_embedding_inner_diff'
            weight_saver = tf.train.import_meta_graph(pretrain_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            item_bias = pretrain_graph.get_tensor_by_name('item_bias:0')
            item_embeddings = pretrain_graph.get_tensor_by_name('item_embeddings:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, pretrain_file)
                ie, ib = sess.run([item_embeddings, item_bias])
            all_weights['item_embeddings'] = tf.Variable(ie, dtype=tf.float32)
            all_weights['item_bias'] = tf.Variable(ib, dtype=tf.float32)
        else:  # without pretrain
            all_weights['item_embeddings'] = tf.Variable(
                tf.random_normal([self.features_item, self.hidden_factor], 0.0, 0.01),
                name='item_embeddings')  # features_M * K
            # feature bias vectors initialization
            all_weights['item_bias'] = tf.Variable(
                tf.random_normal([self.features_item, self.hidden_factor], 0.0, 0.01),
                name='item_bias')  # features_M * K

        # parameters initialization
        all_weights['user_embeddings'] = tf.Variable(
            tf.random_normal([self.features_user, self.hidden_factor], 0.0, 0.01),
            name='user_embeddings')  # features_M * K
        all_weights['skill_embeddings'] = tf.Variable(
            tf.random_normal([self.features_skills, self.hidden_factor], 0.0, 0.01),
            name='skill_embeddings')  # features_M * K
        mn = tf.Variable(tf.zeros([1, self.hidden_factor]))
        all_weights['skill_embeddings'] = tf.concat([all_weights['skill_embeddings'], mn], 0)
        all_weights['wins_embeddings'] = tf.Variable(
            tf.random_normal([self.features_skills, self.hidden_factor], 0.0, 0.01),
            name='wins_embeddings')  # features_M * K
        all_weights['wins_embeddings'] = tf.concat([all_weights['wins_embeddings'], mn], 0)
        all_weights['fails_embeddings'] = tf.Variable(
            tf.random_normal([self.features_skills, self.hidden_factor], 0.0, 0.01),
            name='fails_embeddings')  # features_M * K
        all_weights['fails_embeddings'] = tf.concat([all_weights['fails_embeddings'], mn], 0)
        all_weights['last_embeddings'] = tf.Variable(
            tf.random_normal([2 * self.features_skills, self.hidden_factor], 0.0, 0.01),
            name='last_embeddings')  # features_M * K
        all_weights['last_embeddings'] = tf.concat([all_weights['last_embeddings'], mn], 0)

        # feature bias vectors initialization
        all_weights['user_bias'] = tf.Variable(
            tf.random_normal([self.features_user, self.hidden_factor], 0.0, 0.01),
            name='user_bias')  # features_M * K
        all_weights['skill_bias'] = tf.Variable(
            tf.random_normal([self.features_skills, self.hidden_factor], 0.0, 0.01),
            name='skill_bias')  # features_M * K
        mn1 = tf.Variable(tf.zeros([1, self.hidden_factor]))
        all_weights['skill_bias'] = tf.concat([all_weights['skill_bias'], mn1], 0)
        all_weights['wins_bias'] = tf.Variable(
            tf.random_normal([self.features_skills, self.hidden_factor], 0.0, 0.01),
            name='wins_bias')  # features_M * K
        all_weights['wins_bias'] = tf.concat([all_weights['wins_bias'], mn1], 0)
        all_weights['fails_bias'] = tf.Variable(
            tf.random_normal([self.features_skills, self.hidden_factor], 0.0, 0.01),
            name='fails_bias')  # features_M * K
        all_weights['fails_bias'] = tf.concat([all_weights['fails_bias'], mn1], 0)
        all_weights['last_bias'] = tf.Variable(
            tf.random_normal([2 * self.features_skills, self.hidden_factor], 0.0, 0.01),
            name='last_bias')  # features_M * K
        all_weights['last_bias'] = tf.concat([all_weights['last_bias'], mn1], 0)

        # global bias initialization
        all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1

        num_layer = len(self.layers)
        if num_layer > 0:
            glorot = np.sqrt(2.0 / (1 * self.hidden_factor + self.layers[0]))
            weights1['layer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1 * self.hidden_factor, self.layers[0])), dtype=np.float32)
            weights1['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[0])),
                                             dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                weights1['layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights1['bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.layers[-1] + 1))
            weights1['prediction_diff'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[-1], 1)),
                                                      dtype=np.float32)  # layers[-1] * 1
        else:
            weights1['prediction_diff'] = tf.Variable(
                np.ones((2 * self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1

        num_diff_layer = len(self.diff_layers)
        if num_diff_layer > 0:
            glorot = np.sqrt(2.0 / (1 * self.hidden_factor + self.diff_layers[0]))
            weights1['diff_layer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1 * self.hidden_factor, self.diff_layers[0])),
                dtype=np.float32)
            weights1['diff_bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.diff_layers[0])),
                                                  dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.diff_layers[i - 1] + self.diff_layers[i]))
                weights1['diff_layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.diff_layers[i - 1], self.diff_layers[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights1['diff_bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.diff_layers[i])),
                    dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.diff_layers[-1] + 1))
            weights1['prediction_diff_bias'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.diff_layers[-1], 1)),
                dtype=np.float32)  # layers[-1] * 1
        else:
            weights1['prediction_diff_bias'] = tf.Variable(
                np.ones((2 * self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1

        num_fenlayer_v = len(self.atten_layers_v)
        if num_fenlayer_v > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor * self.num_field + self.atten_layers_v[0]))

            weights2['fenlayer_v_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot,
                                 size=(self.hidden_factor * self.num_field, self.atten_layers_v[0])),
                dtype=np.float32)
            weights2['fenbias_v_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.atten_layers_v[0])),
                dtype=np.float32)  # 1 * layers[0]

            for i in range(1, num_fenlayer_v):
                glorot = np.sqrt(2.0 / (self.atten_layers_v[i - 1] + self.atten_layers_v[i]))
                weights2['fenlayer_v_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.atten_layers_v[i - 1], self.atten_layers_v[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights2['fenbias_v_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.atten_layers_v[i])),
                    dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.atten_layers_v[-1] + 1))

            weights2['prediction_attention_v'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.atten_layers_v[-1], self.num_field)),
                dtype=np.float32)  # layers[-1] * 1
        num_layer_v = len(self.deep_layers_v)
        if num_layer_v > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.deep_layers_v[0]))
            weights2['layer_v_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor, self.deep_layers_v[0])), dtype=np.float32)
            weights2['bias_v_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers_v[0])),
                                             dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer_v):
                glorot = np.sqrt(2.0 / (self.deep_layers_v[i - 1] + self.deep_layers_v[i]))
                weights2['layer_v_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers_v[i - 1], self.deep_layers_v[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights2['bias_v_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers_v[i])),
                    dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.deep_layers_v[-1] + 1))
            weights2['prediction_dnn_v'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers_v[-1], 1)),
                dtype=np.float32)  # layers[-1] * 1
        else:
            weights2['prediction_dnn_v'] = tf.Variable(
                np.ones((self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1

        num_fenlayer = len(self.atten_layers)
        if num_fenlayer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor * self.num_field + self.atten_layers[0]))

            weights2['fenlayer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot,
                                 size=(self.hidden_factor * self.num_field, self.atten_layers[0])),
                dtype=np.float32)
            weights2['fenbias_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.atten_layers[0])),
                dtype=np.float32)  # 1 * layers[0]

            for i in range(1, num_fenlayer):
                glorot = np.sqrt(2.0 / (self.atten_layers[i - 1] + self.atten_layers[i]))
                weights2['fenlayer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.atten_layers[i - 1], self.atten_layers[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights2['fenbias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.atten_layers[i])),
                    dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.atten_layers[-1] + 1))

            weights2['prediction_attention'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.atten_layers[-1], self.num_field)),
                dtype=np.float32)  # layers[-1] * 1

        num_layer = len(self.deep_layers)
        if num_layer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.deep_layers[0]))
            weights2['layer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor, self.deep_layers[0])), dtype=np.float32)
            weights2['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                             dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                weights2['layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights2['bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])), dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.deep_layers[-1] + 1))
            weights2['prediction_dnn'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[-1], 1)),
                                                 dtype=np.float32)  # layers[-1] * 1
        else:
            weights2['prediction_dnn'] = tf.Variable(
                np.ones((self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1

        return all_weights, weights1, weights2

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.user_features: data['X_user'], self.train_labels: data['Y'],
                     self.wins_features: data['X_wins'], self.wins_nums: data['X_wins_nums'],
                     self.fails_features: data['X_fails'], self.fails_nums: data['X_fails_nums'],
                     self.skill_nums: data['X_skill_nums'], self.item_features: data['X_item'],
                     self.skill_features: data['X_skill'], self.dropout_keep: self.keep_prob,
                     self.difficulty_labels: data['Y_diff'], self.train_phase: True,
                     self.dropout_keep_attention: self.keep_attention, self.dropout_keep_deep: self.keep_deep,
                     self.dropout_keep_diff: self.keep_diff, self.dropout_keep_attention_v: self.keep_attention_v,
                     self.dropout_keep_deep_v: self.keep_deep_v, self.last_features: data['X_last'],
                     self.last_nums: data['X_last_nums']}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X_user, X_item, X_skill, Y, Y_diff, X_skill_nums, X_wins, X_wins_nums, X_fails, X_fails_nums = [], [], [], [], [], [], [], [], [], []
        X_last, X_last_nums = [], []
        # forward get sample
        i = start_index
        while len(X_user) < batch_size and i < len(data['X_user']):
            if len(data['X_user'][i]) == len(data['X_user'][start_index]):
                Y.append([data['Y'][i]])
                Y_diff.append([data['Y_diff'][i]])
                X_user.append(data['X_user'][i])
                X_item.append(data['X_item'][i])
                X_skill.append(data['X_skill'][i])
                X_wins.append(data['X_wins'][i])
                X_fails.append(data['X_fails'][i])
                X_skill_nums.append(data['X_skill_nums'][i])
                X_wins_nums.append(data['X_wins_nums'][i])
                X_fails_nums.append(data['X_fails_nums'][i])
                X_last.append(data['X_last'][i])
                X_last_nums.append(data['X_last_nums'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X_user) < batch_size and i >= 0:
            if len(data['X_user'][i]) == len(data['X_user'][start_index]):
                Y.append([data['Y'][i]])
                Y_diff.append([data['Y_diff'][i]])
                X_user.append(data['X_user'][i])
                X_item.append(data['X_item'][i])
                X_skill.append(data['X_skill'][i])
                X_wins.append(data['X_wins'][i])
                X_fails.append(data['X_fails'][i])
                X_skill_nums.append(data['X_skill_nums'][i])
                X_wins_nums.append(data['X_wins_nums'][i])
                X_fails_nums.append(data['X_fails_nums'][i])
                X_last.append(data['X_last'][i])
                X_last_nums.append(data['X_last_nums'][i])
                i = i - 1
            else:
                break
        return {'X_user': X_user, 'X_item': X_item, 'X_skill': X_skill, 'Y': Y, 'X_skill_nums': X_skill_nums,
                'X_wins': X_wins, 'X_wins_nums': X_wins_nums,
                'X_fails': X_fails, 'X_fails_nums': X_fails_nums, 'Y_diff': Y_diff, 'X_last': X_last, 'X_last_nums': X_last_nums}

    def shuffle_in_unison_scary(self, a, b, c, d, e, f, g, h, i, j, k, l):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
        np.random.set_state(rng_state)
        np.random.shuffle(e)
        np.random.set_state(rng_state)
        np.random.shuffle(f)
        np.random.set_state(rng_state)
        np.random.shuffle(g)
        np.random.set_state(rng_state)
        np.random.shuffle(h)
        np.random.set_state(rng_state)
        np.random.shuffle(i)
        np.random.set_state(rng_state)
        np.random.shuffle(j)
        np.random.set_state(rng_state)
        np.random.shuffle(k)
        np.random.set_state(rng_state)
        np.random.shuffle(l)

    def train(self, Train_data, Test_data):  # fit a dataset
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X_user'], Train_data['X_item'], Train_data['X_skill'],
                                         Train_data['Y'], Train_data['X_skill_nums'],
                                         Train_data['X_wins'], Train_data['Y_diff'],
                                         Train_data['X_wins_nums'], Train_data['X_fails'], Train_data['X_fails_nums'],Train_data['X_last'], Train_data['X_last_nums'])
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()

            # output validation
            train_acc, train_auc = self.evaluate(Train_data)
            test_acc, test_auc = self.evaluate(Test_data)

            self.train_acc.append(train_acc)
            self.test_acc.append(test_acc)
            self.train_auc.append(train_auc)
            self.test_auc.append(test_auc)

            if self.verbose > 0 and epoch % self.verbose == 0:

                logger.info("Epoch %d [%.1f s]\ttrain_acc=%.4f, test_acc=%.4f [%.1f s]"
                            % (epoch + 1, t2 - t1, train_acc, test_acc, time() - t2))
                logger.info("Epoch %d [%.1f s]\ttrain_auc=%.4f, test_auc=%.4f [%.1f s]"
                            % (epoch + 1, t2 - t1, train_auc, test_auc, time() - t2))

            if self.early_stop > 0 and self.eva_termination(self.test_auc):
                # print "Early stop at %d based on validation result." %(epoch+1)
                break

    def eva_termination(self, valid):
        if self.loss_type == 'square_loss':
            if len(valid) > 5:
                if valid[-1] < valid[-2] and valid[-2] < valid[-3] and valid[-3] < valid[-4] and valid[-4] < valid[-5]:
                    return True
        else:
            if len(valid) > 10:
                if valid[-1] < valid[-2] < valid[-3] < valid[-4] < valid[-5] < valid[-6] < valid[-7] < valid[-8] < valid[-9] < valid[-10]:
                    return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        feed_dict = {self.user_features: [item for item in data['X_user']], self.train_labels: [[y] for y in data['Y']],
                     self.item_features: [item for item in data['X_item']],
                     self.skill_features: [item for item in data['X_skill']],
                     self.skill_nums: [item for item in data['X_skill_nums']],
                     self.difficulty_labels: [[y] for y in data['Y_diff']],
                     self.wins_nums: [item for item in data['X_wins_nums']],
                     self.wins_features: [item for item in data['X_wins']],
                     self.fails_nums: [item for item in data['X_fails_nums']],
                     self.fails_features: [item for item in data['X_fails']],
                     self.dropout_keep: self.no_dropout, self.dropout_keep_attention: self.no_atten_dropout,
                     self.dropout_keep_deep: self.no_deep_dropout, self.train_phase: False, self.dropout_keep_diff: self.no_dropout_diff,
                     self.dropout_keep_attention_v: self.no_atten_dropout_v, self.dropout_keep_deep_v: self.no_deep_dropout_v,
                     self.last_features: [item for item in data['X_last']],
                     self.last_nums: [item for item in data['X_last_nums']]}
        predictions, loss = self.sess.run((self.out, self.loss), feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(data['Y'], (num_example,))
        if self.loss_type == 'square_loss':
            predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
            predictions_bounded = np.minimum(predictions_bounded,
                                             np.ones(num_example) * max(y_true))  # bound the higher values
            RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
            return RMSE
        elif self.loss_type == 'log_loss':

            auc = roc_auc_score(y_true, y_pred)

            acc = np.mean(y_true == np.round(y_pred))

            return acc, auc


if __name__ == '__main__':
    # Data loading
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('MF-DAKT.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    data = Data_loader()
    X_user, X_item, X_skill, X_skill_nums, X_wins, X_fails, X_wins_nums, X_fails_nums, y, y_diff, num_users, num_items, num_skills, X_last, X_last_nums = data.data_load()
    logger.info("user_nums=%d, item_nums=%d, skill_max_nums=%d" % (num_users, num_items, num_skills))
    logger.info("Data loaded successfully-----")
    print("Data loaded successfully-----")
    kf = KFold(n_splits=5, shuffle=True)
    list_acc = []
    list_auc = []
    list_acc_best, list_auc_best = [], []
    total_time = []
    for train_index, test_index in kf.split(y):
        X_train_user, X_test_user = np.array(X_user)[train_index], np.array(X_user)[test_index]
        X_train_item, X_test_item = np.array(X_item)[train_index], np.array(X_item)[test_index]
        X_train_skill, X_test_skill = np.array(X_skill)[train_index], np.array(X_skill)[test_index]
        Y_train, Y_test = np.array(y)[train_index], np.array(y)[test_index]
        Y_train_diff, Y_test_diff = np.array(y_diff)[train_index], np.array(y_diff)[test_index]
        X_train_skill_nums, X_test_skill_nums = np.array(X_skill_nums)[train_index], np.array(X_skill_nums)[test_index]
        X_train_wins, X_test_wins = np.array(X_wins)[train_index], np.array(X_wins)[test_index]
        X_train_fails, X_test_fails = np.array(X_fails)[train_index], np.array(X_fails)[test_index]
        X_train_wins_nums, X_test_wins_nums = np.array(X_wins_nums)[train_index], np.array(X_wins_nums)[test_index]
        X_train_fails_nums, X_test_fails_nums = np.array(X_fails_nums)[train_index], np.array(X_fails_nums)[test_index]
        X_train_last, X_test_last = np.array(X_last)[train_index], np.array(X_last)[test_index]
        X_train_last_nums, X_test_last_nums = np.array(X_last_nums)[train_index], np.array(X_last_nums)[test_index]
        Train_data = {'X_user': X_train_user, 'X_item': X_train_item, 'X_skill': X_train_skill, 'Y': Y_train,
                      'X_skill_nums': X_train_skill_nums, 'X_wins': X_train_wins, 'X_wins_nums': X_train_wins_nums,
                      'X_fails': X_train_fails, 'X_fails_nums': X_train_fails_nums, 'Y_diff': Y_train_diff,
                      'X_last': X_train_last, 'X_last_nums': X_train_last_nums}
        Test_data = {'X_user': X_test_user, 'X_item': X_test_item, 'X_skill': X_test_skill, 'Y': Y_test,
                     'X_skill_nums': X_test_skill_nums, 'X_wins': X_test_wins,
                     'X_wins_nums': X_test_wins_nums, 'X_fails': X_test_fails, 'X_fails_nums': X_test_fails_nums,
                     'Y_diff': Y_test_diff, 'X_last': X_test_last, 'X_last_nums': X_test_last_nums}
        print("Data is splitted successfully-----")
        args = parse_args()
        if args.verbose > 0:
            logger.info(
                "hidden_factor=%d, layers=%s, dropout_keep=%s, diff_layers=%s, keep_diff=%s, lr=%.4f, lambda=%.4f, optimizer=%s, activation=%s, "
                "fenlayers=%s, drop_atten=%s, deeplayer=%s, drop_deep=%s "
                % (args.hidden_factor, args.layers, args.keep_prob, args.diff_layers, args.keep_diff, args.lr, args.lamda, args.optimizer, args.activation, args.fenlayers, args.keep_attention, args.deep_layers, args.keep_deep))
        activation_function = tf.nn.relu
        if args.activation == 'sigmoid':
            activation_function = tf.sigmoid
        elif args.activation == 'tanh':
            activation_function == tf.tanh
        elif args.activation == 'identity':
            activation_function = tf.identity

        # Training
        t1 = time()
        model = DAKT(num_users, num_items, num_skills, args.hidden_factor, args.loss_type, args.pretrain,
                      args.epoch, args.batch_size, args.lr, args.lamda, eval(args.keep_prob), args.optimizer,
                      args.batch_norm,
                      activation_function, args.verbose, args.early_stop, eval(args.layers))
        model.train(Train_data, Test_data)

        best_auc_score = max(model.test_auc)
        best_acc_score = max(model.test_acc)

        best_epoch = model.test_acc.index(best_acc_score)
        best_epoch_auc = model.test_auc.index(best_auc_score)

        logger.info("Best Iter(test_acc)= %d\t train(acc) = %.4f, test(acc) = %.4f [%.1f s]"
                    % (best_epoch + 1, model.train_acc[best_epoch], model.test_acc[best_epoch], time() - t1))
        logger.info("Best Iter(test_auc)= %d\t train(auc) = %.4f, test(auc) = %.4f [%.1f s]"
                    % (
                        best_epoch_auc + 1, model.train_auc[best_epoch_auc], model.test_auc[best_epoch_auc],
                        time() - t1))

        list_acc.append(model.test_acc[-1])
        list_auc.append(model.test_auc[-1])

        list_acc_best.append(best_acc_score)
        list_auc_best.append(best_auc_score)

        total_time.append(time() - t1)

    logger.info("Average(all)\t test(acc) = %.4f+/-%.4f, test(auc) = %.4f+/-%.4f" % (
        np.mean(list_acc), np.std(list_acc), np.mean(list_auc), np.std(list_auc)))
    logger.info("Average(Best)\t test(acc) = %.4f+/-%.4f, test(auc) = %.4f+/-%.4f" % (
        np.mean(list_acc_best), np.std(list_acc_best), np.mean(list_auc_best), np.std(list_auc_best)))

    logger.info("total time:%.1f s" % (np.sum(total_time)))
