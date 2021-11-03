import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error


class MF_DAKT(BaseEstimator, TransformerMixin):
    def __init__(self, features_student, features_question, features_concept, args, length, logger, random_seed=2016):
        # bind params to class
        self.batch_size = args.batch_size
        self.hidden_factor = args.hidden_factor
        self.loss_type = args.loss_type
        self.pretrain_flag = args.pretrain
        self.features_student = features_student
        self.features_question = features_question
        self.features_concept = features_concept
        self.lamda_bilinear = args.lamda
        self.epoch = args.epoch
        self.random_seed = random_seed
        self.logger = logger

        self.length = length

        # regularization term about question difficulty parameters in factor interaction subspace
        self.diff_layers_interaction = eval(args.diff_layers_interaction)
        self.keep_diff_interaction = np.array(eval(args.keep_diff_interaction))
        self.no_diff_interaction_dropout = np.array([1 for i in range(len(eval(args.keep_diff_interaction)))])

        # regularization term about question difficulty parameters in factor subspace
        self.diff_layers_sum = np.array(eval(args.diff_layers_sum))
        self.keep_diff_sum = np.array(eval(args.keep_diff_sum))
        self.no_diff_sum_dropout = np.array([1 for i in range(len(eval(args.keep_diff_sum)))])

        # attention-based pooling parameters of vectors in factor subspace
        self.atten_layers_sum_pooling = np.array(eval(args.atten_layers_sum_pooling))
        self.keep_atten_sum_pooling = np.array(eval(args.keep_atten_sum_pooling))
        self.no_atten_sum_pooling_dropout = np.array([1 for i in range(len(eval(args.keep_atten_sum_pooling)))])

        # attention parameters in ACNN of factor subspace
        self.atten_layers_acnn_sum = np.array(eval(args.atten_layers_acnn_sum))
        self.keep_atten_acnn_sum = np.array(eval(args.keep_atten_acnn_sum))
        self.no_atten_acnn_sum_dropout = np.array([1 for i in range(len(eval(args.keep_atten_acnn_sum)))])

        # DNN parameters of sum pooling vector in factor subspace
        self.deep_layers_sum_pooling = np.array(eval(args.deep_layers_sum_pooling))
        self.keep_deep_sum_pooling = np.array(eval(args.keep_deep_sum_pooling))
        self.no_deep_sum_pooling_dropout = np.array([1 for i in range(len(eval(args.keep_deep_sum_pooling)))])

        # attention-based pooling parameters of vectors in factor interaction subspace
        self.atten_layers_interaction_pooling = np.array(eval(args.atten_layers_interaction_pooling))
        self.keep_atten_interaction_pooling = np.array(eval(args.keep_atten_interaction_pooling))
        self.no_atten_interaction_pooling_dropout = np.array([1 for i in range(len(eval(args.keep_atten_interaction_pooling)))])

        # attention parameters in ACNN of factor interaction subspace
        self.atten_layers_acnn_interaction = np.array(eval(args.atten_layers_acnn_interaction))
        self.keep_atten_acnn_interaction = np.array(eval(args.keep_atten_acnn_interaction))
        self.no_atten_acnn_interaction_dropout = np.array([1 for i in range(len(eval(args.keep_atten_acnn_interaction)))])

        # DNN parameters of sum pooling vector in factor interaction subspace
        self.deep_layers_interaction_pooling = np.array(eval(args.deep_layers_interaction_pooling))
        self.keep_deep_interaction_pooling = np.array(eval(args.keep_deep_interaction_pooling))
        self.no_deep_interaction_pooling_dropout = np.array([1 for i in range(len(eval(args.keep_deep_interaction_pooling)))])

        self.optimizer_type = args.optimizer
        self.learning_rate = args.lr
        self.batch_norm = batch_norm
        self.verbose = args.verbose
        self.early_stop = args.early_stop
        self.num_field = 4

        # record performance of each epoch
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

            # _________ Input Data _________
            self.student_features = tf.placeholder(tf.int32, shape=[None, None])  # None * 1
            self.question_features = tf.placeholder(tf.int32, shape=[None, None])  # None * 1
            self.concept_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.concept_nums = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M
            self.success_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.success_nums = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M
            self.fails_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.fails_nums = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M
            self.recent_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.recent_nums = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M
            self.recent_interval = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M

            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.difficulty_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1

            self.dropout_diff_sum = tf.placeholder(tf.float32, shape=[None])
            self.dropout_diff_interaction = tf.placeholder(tf.float32, shape=[None])

            self.dropout_atten_acnn_sum = tf.placeholder(tf.float32, shape=[None])
            self.dropout_atten_acnn_interaction = tf.placeholder(tf.float32, shape=[None])

            self.dropout_atten_sum = tf.placeholder(tf.float32, shape=[None])
            self.dropout_atten_interaction = tf.placeholder(tf.float32, shape=[None])

            self.dropout_dnn_sum = tf.placeholder(tf.float32, shape=[None])
            self.dropout_dnn_interaction = tf.placeholder(tf.float32, shape=[None])

            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights, self.weights2, self.weights3 = self._initialize_weights()

            self.factors = tf.reshape(tf.nn.embedding_lookup(self.weights['concept_forget_factors'], self.concept_features),
                                      [-1, self.length])
            self.last_nums_forget = tf.multiply(self.recent_nums,
                                                tf.divide(1, tf.exp(tf.multiply(self.recent_interval, self.factors))))  # forgetting function of recent factor

            # ********************* Embedding Component of Factor interaction subspace ***********************

            # _________ get embeddings of factors in factor interaction subspace _________
            interaction_embeddings_student = tf.nn.embedding_lookup(self.weights['student_embeddings_interaction'], self.student_features)  # None * 1 * d
            interaction_embeddings_question = tf.nn.embedding_lookup(self.weights['question_embeddings_interaction'], self.question_features)   # None * 1 * d
            interaction_embeddings_concept = tf.multiply(
                tf.nn.embedding_lookup(self.weights['concept_embeddings_interaction'], self.concept_features),
                tf.expand_dims(self.concept_nums, 2))   # None * features_M * d
            interaction_embeddings_success = tf.multiply(
                tf.nn.embedding_lookup(self.weights['success_embeddings_interaction'], self.success_features),
                tf.expand_dims(self.success_nums, 2))  # None * features_M * d
            interaction_embeddings_fails = tf.multiply(
                tf.nn.embedding_lookup(self.weights['fails_embeddings_interaction'], self.fails_features),
                tf.expand_dims(self.fails_nums, 2))  # None * features_M * d
            interaction_embeddings_recent = tf.multiply(
                tf.nn.embedding_lookup(self.weights['recent_embeddings_interaction'], self.recent_features),
                tf.expand_dims(self.last_nums_forget, 2))  # None * features_M * d

            self.concapt_nums_count = tf.expand_dims(tf.reduce_sum(self.concept_nums, 1, keepdims=True), 2)

            interaction_embeddings_concept = tf.reduce_sum(interaction_embeddings_concept, 1, keepdims=True)
            interaction_embeddings_concept_mean = tf.div(interaction_embeddings_concept, self.concapt_nums_count)  # average concept embedding

            interaction_embeddings_success = tf.reduce_sum(interaction_embeddings_success, 1, keepdims=True)  # sum success embedding
            interaction_embeddings_fails = tf.reduce_sum(interaction_embeddings_fails, 1, keepdims=True)  # sum fail embedding

            self.skill_nums_count_recent = tf.expand_dims(tf.reduce_sum(self.recent_nums, 1, keepdims=True), 2)

            interaction_embeddings_recent_mean = tf.reduce_sum(interaction_embeddings_recent, 1, keepdims=True)

            # ________ Question difficulty regularization term in factor interaction subspace __________
            self.diff_interaction = tf.reshape(interaction_embeddings_question, [-1, self.hidden_factor])
            for i in range(0, len(self.diff_layers_interaction)):
                self.diff_interaction = tf.add(tf.matmul(self.diff_interaction, self.weights2['diff_layer_interaction_%d' % i]), self.weights2['diff_bias_interaction_%d' % i])  # None * layer[i]
                if self.batch_norm:
                    self.diff_interaction = self.batch_norm_layer(self.diff_interaction, train_phase=self.train_phase,
                                                                  scope_bn='diff_bn_interaction_%d' % i)  # batch normalization

                self.diff_interaction = tf.nn.relu(self.diff_interaction)
                self.diff_interaction = tf.nn.dropout(self.diff_interaction, self.dropout_diff_interaction[i])  # dropout at each Deep layer
            self.difficulty_interaction = tf.matmul(self.diff_interaction, self.weights2['diff_pred_interaction'])  # None * 1
            self.mse_interaction = tf.reduce_mean(tf.square(self.difficulty_interaction - self.difficulty_labels))  # regularization loss of factor interaction subspace

            # ________ ACNN component in factor interaction subspace __________
            interaction_embeddings_attempt = tf.concat([interaction_embeddings_success, interaction_embeddings_fails], 1)
            interaction_embeddings_attempt = tf.concat([interaction_embeddings_attempt, interaction_embeddings_recent_mean], 1)
            # ------------ attention component of ACNN --------------------
            interaction_embeddings_attempt_dnn = tf.reshape(interaction_embeddings_attempt, shape=[-1, 3 * self.hidden_factor])
            self.dnn_recent_interaction = tf.add(tf.matmul(interaction_embeddings_attempt_dnn, self.weights3['attenweight_acnn_interaction_0']),
                                                 self.weights3['attenbias_acnn_interaction_0'])  # None * layer[i]
            if self.batch_norm:
                self.dnn_recent_interaction = self.batch_norm_layer(self.dnn_recent_interaction, train_phase=self.train_phase,
                                                                    scope_bn='attenbn_acnn_interaction_0')  # batch normalization
            self.dnn_recent_interaction = tf.nn.relu(self.dnn_recent_interaction)
            self.dnn_recent_interaction = tf.nn.dropout(self.dnn_recent_interaction,
                                                        self.dropout_atten_acnn_interaction[0])

            for i in range(1, len(self.atten_layers_acnn_interaction)):
                self.dnn_recent_interaction = tf.add(tf.matmul(self.dnn_recent_interaction, self.weights3['attenweight_acnn_interaction_%d' % i]),
                                                     self.weights3['attenbias_acnn_interaction_%d' % i])
                if self.batch_norm:
                    self.dnn_recent_interaction = self.batch_norm_layer(self.dnn_recent_interaction, train_phase=self.train_phase,
                                                                        scope_bn='attenbn_acnn_interaction_%d' % i)
                self.dnn_recent_interaction = tf.nn.relu(self.dnn_recent_interaction)
                self.dnn_recent_interaction = tf.nn.dropout(self.dnn_recent_interaction,
                                                            self.dropout_atten_acnn_interaction[i])
            self.dnn_out_recent_interaction = tf.matmul(self.dnn_recent_interaction, self.weights3['attenpred_acnn_interaction'])  # None * 3
            self.out_recent_interaction = tf.nn.softmax(self.dnn_out_recent_interaction)
            self.attempt_embeddings_interaction = tf.multiply(interaction_embeddings_attempt, tf.expand_dims(self.out_recent_interaction, 2))
            # ------------ one-dimensional convolution component of ACNN --------------------
            conv_embeddings_interaction = tf.layers.conv1d(self.attempt_embeddings_interaction, filters=128, kernel_size=2, strides=1,
                                               padding='VALID')
            conv_embeddings_interaction = tf.nn.relu(conv_embeddings_interaction)
            conv_embeddings_interaction = tf.layers.conv1d(conv_embeddings_interaction, filters=128, kernel_size=2, strides=1, padding='VALID')
            conv_embeddings_interaction = tf.nn.relu(conv_embeddings_interaction)

            # ________ Attention-based Interaction Pooling in factor interaction subspace __________
            interaction_embeddings = tf.concat([interaction_embeddings_student, interaction_embeddings_question], 1)
            interaction_embeddings = tf.concat([interaction_embeddings, interaction_embeddings_concept_mean], 1)
            interaction_embeddings = tf.concat([interaction_embeddings, conv_embeddings_interaction], 1)  # None * 4
            # ------------ attention component of interaction pooling --------------------
            dnn_interaction_embeddings = tf.reshape(interaction_embeddings, shape=[-1, self.num_field * self.hidden_factor])
            self.dnn_interaction = tf.add(tf.matmul(dnn_interaction_embeddings, self.weights3['attenweight_pooling_interaction_0']),
                                          self.weights3['attenbias_pooling_interaction_0'])
            if self.batch_norm:
                self.dnn_interaction = self.batch_norm_layer(self.dnn_interaction, train_phase=self.train_phase,
                                                             scope_bn='attenbn_pooling_interaction_0')
            self.dnn_interaction = tf.nn.relu(self.dnn_interaction)
            self.dnn_interaction = tf.nn.dropout(self.dnn_interaction, self.dropout_atten_interaction[0])

            for i in range(1, len(self.atten_layers_interaction_pooling)):
                self.dnn_interaction = tf.add(tf.matmul(self.dnn_interaction, self.weights3['attenweight_pooling_interaction_%d' % i]),
                                              self.weights3['attenbias_pooling_interaction_%d' % i])
                if self.batch_norm:
                    self.dnn_interaction = self.batch_norm_layer(self.dnn_interaction, train_phase=self.train_phase,
                                                                 scope_bn='attenbn_pooling_interaction_%d' % i)
                self.dnn_interaction = tf.nn.relu(self.dnn_interaction)
                self.dnn_interaction = tf.nn.dropout(self.dnn_interaction, self.dropout_atten_interaction[i])
            self.dnn_out_interaction = tf.matmul(self.dnn_interaction, self.weights3['attenpred_pooling_interaction'])  # None * 4
            self.out_interaction = tf.constant(float(4)) * tf.nn.softmax(self.dnn_out_interaction)
            self.interaction_embeddings = tf.multiply(interaction_embeddings, tf.expand_dims(self.out_interaction, 2))

            # ------------ pooling component of interaction pooling --------------------
            self.summed_features_emb = tf.reduce_sum(self.interaction_embeddings, 1)
            self.summed_features_emb_square = tf.square(self.summed_features_emb)

            self.squared_features_emb = tf.square(self.interaction_embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)

            self.interaction_pooling = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)
            if self.batch_norm:
                self.interaction_pooling = self.batch_norm_layer(self.interaction_pooling, train_phase=self.train_phase,
                                                                 scope_bn='bn_interaction')
            self.interaction_pooling = tf.nn.dropout(self.interaction_pooling, self.dropout_diff_interaction[-1])

            # ------------ DNN component of interaction pooling --------------------
            for i in range(0, len(self.deep_layers_interaction_pooling)):
                self.interaction_pooling = tf.add(tf.matmul(self.interaction_pooling, self.weights3['layer_interaction_%d' % i]),
                                                  self.weights3['bias_interaction_%d' % i])

                if self.batch_norm:
                    self.interaction_pooling = self.batch_norm_layer(self.interaction_pooling, train_phase=self.train_phase,
                                                                     scope_bn='bn_interaction_deep%d' % i)

                self.interaction_pooling = tf.nn.relu(self.interaction_pooling)
                self.interaction_pooling = tf.nn.dropout(self.interaction_pooling, self.dropout_dnn_interaction[i])
            self.interaction_pooling = tf.matmul(self.interaction_pooling, self.weights3['prediction_dnn_interaction']) # None * 1

            interaction_pooling_value = tf.reduce_sum(self.interaction_pooling, 1, keep_dims=True)  # turn interaction pooling vector as a value

            # ********************* Embedding Component of Factor subspace ***********************

            # _________ get embeddings of factors in factor subspace _________
            self.sum_embeddings_student = tf.nn.embedding_lookup(self.weights['student_embeddings_sum'], self.student_features)
            self.sum_embeddings_question = tf.nn.embedding_lookup(self.weights['question_embeddings_sum'], self.question_features)
            self.sum_embeddings_concept = tf.reduce_sum(
                tf.multiply(tf.nn.embedding_lookup(self.weights['concept_embeddings_sum'], self.concept_features),
                            tf.expand_dims(self.concept_nums, 2)), 1, keepdims=True)
            self.sum_embeddings_success = tf.reduce_sum(
                tf.multiply(tf.nn.embedding_lookup(self.weights['success_embeddings_sum'], self.success_features),
                            tf.expand_dims(self.success_nums, 2)), 1, keepdims=True)
            self.sum_embeddings_fails = tf.reduce_sum(
                tf.multiply(tf.nn.embedding_lookup(self.weights['fails_embeddings_sum'], self.fails_features),
                            tf.expand_dims(self.fails_nums, 2)), 1, keepdims=True)
            self.sum_embeddings_recent = tf.reduce_sum(
                tf.multiply(tf.nn.embedding_lookup(self.weights['recent_embeddings_sum'], self.recent_features),
                            tf.expand_dims(self.last_nums_forget, 2)), 1, keepdims=True)

            self.sum_embeddings_concept_mean = tf.div(self.sum_embeddings_concept, self.concapt_nums_count)

            # ________ Question difficulty regularization term in factor subspace __________
            self.diff_sum = tf.reshape(self.sum_embeddings_question, [-1, self.hidden_factor])
            # ________ Deep Layers __________
            for i in range(0, len(self.diff_layers_sum)):
                self.diff_sum = tf.add(tf.matmul(self.diff_sum, self.weights2['diff_layer_sum_%d' % i]),
                                       self.weights2['diff_bias_sum_%d' % i])
                if self.batch_norm:
                    self.diff_sum = self.batch_norm_layer(self.diff_sum, train_phase=self.train_phase,
                                                          scope_bn='diff_bn_sum_%d' % i)
                self.diff_sum = tf.nn.relu(self.diff_sum)
                self.diff_sum = tf.nn.dropout(self.diff_sum, self.dropout_diff_sum[i])
            self.difficulty_sum = tf.matmul(self.diff_sum, self.weights2['diff_pred_sum'])
            self.mse_sum = tf.reduce_mean(tf.square(self.difficulty_sum - self.difficulty_labels))

            # ________ ACNN component in factor subspace __________
            sum_embeddings_attempt = tf.concat([self.sum_embeddings_success, self.sum_embeddings_fails], 1)
            sum_embeddings_attempt = tf.concat([sum_embeddings_attempt, self.sum_embeddings_concept_mean], 1)
            # ------------ attention component of ACNN --------------------
            sum_embeddings_attempt_dnn = tf.reshape(sum_embeddings_attempt, shape=[-1, 3 * self.hidden_factor])
            self.dnn_recent_sum = tf.add(tf.matmul(sum_embeddings_attempt_dnn, self.weights3['attenweight_acnn_sum_0']),
                                         self.weights3['attenbias_acnn_sum_0'])
            if self.batch_norm:
                self.dnn_recent_sum = self.batch_norm_layer(self.dnn_recent_sum, train_phase=self.train_phase,
                                                            scope_bn='attenbn_acnn_sum_0')
            self.dnn_recent_sum = tf.nn.relu(self.dnn_recent_sum)
            self.dnn_recent_sum = tf.nn.dropout(self.dnn_recent_sum,
                                                self.dropout_atten_acnn_sum[0])
            for i in range(1, len(self.atten_layers_acnn_sum)):
                self.dnn_recent_sum = tf.add(tf.matmul(self.dnn_recent_sum, self.weights3['attenweight_acnn_sum_%d' % i]),
                                             self.weights3['attenbias_acnn_sum_%d' % i])
                if self.batch_norm:
                    self.dnn_recent_sum = self.batch_norm_layer(self.dnn_recent_sum, train_phase=self.train_phase,
                                                                scope_bn='attenbn_acnn_sum_%d' % i)
                self.dnn_recent_sum = tf.nn.relu(self.dnn_recent_sum)
                self.dnn_recent_sum = tf.nn.dropout(self.dnn_recent_sum,
                                                    self.dropout_atten_acnn_sum[i])
            self.dnn_out_recent_sum = tf.matmul(self.dnn_recent_sum, self.weights3['attenpred_acnn_sum'])  # None * 3
            self.out_recent_sum = tf.nn.softmax(self.dnn_out_recent_sum)
            self.attempt_embeddings_sum = tf.multiply(sum_embeddings_attempt, tf.expand_dims(self.out_recent_sum, 2))
            # ------------ one-dimensional convolution component of ACNN --------------------
            conv_embeddings_sum = tf.layers.conv1d(self.attempt_embeddings_sum, filters=128, kernel_size=2, strides=1, padding='VALID')
            conv_embeddings_sum = tf.nn.relu(conv_embeddings_sum)
            conv_embeddings_sum = tf.layers.conv1d(conv_embeddings_sum, filters=128, kernel_size=2, strides=1, padding='VALID')
            conv_embeddings_sum = tf.nn.relu(conv_embeddings_sum)

            # ________ Attention-based Sum Pooling in factor subspace __________
            self.sum_embeddings = tf.concat([self.sum_embeddings_student, self.sum_embeddings_question], 1)
            self.sum_embeddings = tf.concat([self.sum_embeddings, self.sum_embeddings_concept_mean], 1)
            self.sum_embeddings = tf.concat([self.sum_embeddings, conv_embeddings_sum], 1)
            # ------------ attention component of sum pooling --------------------
            dnn_sum_embeddings = tf.reshape(self.sum_embeddings,
                                                shape=[-1, self.num_field * self.hidden_factor])
            self.dnn_sum = tf.add(tf.matmul(dnn_sum_embeddings, self.weights3['attenweight_pooling_sum_0']),
                                  self.weights3['attenbias_pooling_sum_0'])
            if self.batch_norm:
                self.dnn_sum = self.batch_norm_layer(self.dnn_sum, train_phase=self.train_phase,
                                                     scope_bn='attenbn_pooling_sum_0')
            self.dnn_sum = tf.nn.relu(self.dnn_sum)
            self.dnn_sum = tf.nn.dropout(self.dnn_sum, self.dropout_atten_sum[0])

            for i in range(1, len(self.atten_layers_sum_pooling)):
                self.dnn_sum = tf.add(tf.matmul(self.dnn_sum, self.weights3['attenweight_pooling_sum_%d' % i]),
                                      self.weights3['attenbias_pooling_sum_%d' % i])
                if self.batch_norm:
                    self.dnn_sum = self.batch_norm_layer(self.dnn_sum, train_phase=self.train_phase,
                                                         scope_bn='attenbn_pooling_sum_%d' % i)
                self.dnn_sum = tf.nn.relu(self.dnn_sum)
                self.dnn_sum = tf.nn.dropout(self.dnn_sum, self.dropout_atten_sum[i])  # dropout at each Deep layer
            self.dnn_out_sum = tf.matmul(self.dnn_sum, self.weights3['attenpred_pooling_sum'])
            self.out_sum = tf.constant(float(4)) * tf.nn.softmax(self.dnn_out_sum)
            self.sum_embeddings = tf.multiply(self.sum_embeddings, tf.expand_dims(self.out_sum, 2))
            # ------------ pooling component of sum pooling --------------------
            self.sum_embeddings = tf.reduce_sum(self.sum_embeddings, 1)
            if self.batch_norm:
                self.sum_pooling = self.batch_norm_layer(self.sum_embeddings, train_phase=self.train_phase, scope_bn='bn_bias')

            self.sum_pooling = tf.nn.dropout(self.sum_pooling, self.dropout_dnn_sum[-1])
            # ------------ DNN component of sum pooling --------------------
            for i in range(0, len(self.deep_layers_sum_pooling)):
                self.sum_pooling = tf.add(tf.matmul(self.sum_pooling, self.weights3['layer_sum_%d' % i]),
                                                  self.weights3['bias_sum_%d' % i])

                if self.batch_norm:
                    self.sum_pooling = self.batch_norm_layer(self.sum_pooling, train_phase=self.train_phase,
                                                                     scope_bn='bn_sum_%d' % i)

                self.sum_pooling = tf.nn.relu(self.sum_pooling)
                self.sum_pooling = tf.nn.dropout(self.sum_pooling, self.dropout_dnn_sum[i])
            self.sum_pooling = tf.matmul(self.sum_pooling, self.weights3['prediction_dnn_sum'])
            sum_pooling_value = tf.reduce_sum(self.sum_pooling, 1, keep_dims=True)

            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out = tf.add_n([interaction_pooling_value, sum_pooling_value, Bias])  # None * 1

            # Compute the loss.
            if self.loss_type == 'square_loss':
                if self.lamda_bilinear > 0:
                    self.loss = tf.nn.l2_loss(
                        tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(
                        self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                else:
                    self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
            elif self.loss_type == 'log_loss':
                if self.lamda_bilinear > 0:
                    self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out,
                                                                        labels=self.train_labels) + tf.add_n(
                        [tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights3[v]) for v in
                         self.weights3])
                else:
                    self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.train_labels)
            self.out = tf.nn.sigmoid(self.out)
            self.total_loss = tf.reduce_mean(self.loss) + self.mse_interaction + self.mse_sum
            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.total_loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.total_loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.total_loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.total_loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        weights1 = dict()
        weights2 = dict()
        if self.pretrain_flag > 0:  # with pretrain
            pretrain_file = 'ednet/pre_train/problem_embedding_inner_diff_128dims'
            weight_saver = tf.train.import_meta_graph(pretrain_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            item_bias = pretrain_graph.get_tensor_by_name('item_bias:0')
            item_embeddings = pretrain_graph.get_tensor_by_name('item_embeddings:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, pretrain_file)
                ie, ib = sess.run([item_embeddings, item_bias])
            all_weights['question_embeddings_interaction'] = tf.Variable(ie, dtype=tf.float32)
            all_weights['question_embeddings_sum'] = tf.Variable(ib, dtype=tf.float32)
        else:  # without pretrain
            all_weights['question_embeddings_interaction'] = tf.Variable(
                tf.random_normal([self.features_question, self.hidden_factor], 0.0, 0.01),
                name='question_embeddings_interaction')
            all_weights['question_embeddings_sum'] = tf.Variable(
                tf.random_normal([self.features_question, self.hidden_factor], 0.0, 0.01),
                name='question_embeddings_sum')

        mn_ = tf.Variable(tf.zeros([1, 1]))
        all_weights['concept_forget_factors'] = tf.Variable(tf.random_normal([self.features_concept, 1], 0.0, 0.01))
        all_weights['concept_forget_factors'] = tf.concat([all_weights['concept_forget_factors'], mn_], 0)

        # global bias initialization
        all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1

        # ********************* Parameters of Factor interaction subspace ***********************
        # parameters initialization of embeddings in factor interaction subspace
        all_weights['student_embeddings_interaction'] = tf.Variable(
            tf.random_normal([self.features_student, self.hidden_factor], 0.0, 0.01),
            name='student_embeddings_interaction')  # features_M * K
        all_weights['concept_embeddings_interaction'] = tf.Variable(
            tf.random_normal([self.features_concept, self.hidden_factor], 0.0, 0.01),
            name='concept_embeddings_interaction')  # features_M * K
        mn = tf.Variable(tf.zeros([1, self.hidden_factor]))
        all_weights['concept_embeddings_interaction'] = tf.concat([all_weights['concept_embeddings_interaction'], mn], 0)
        all_weights['success_embeddings_interaction'] = tf.Variable(
            tf.random_normal([self.features_concept, self.hidden_factor], 0.0, 0.01),
            name='success_embeddings_interaction')  # features_M * K
        all_weights['success_embeddings_interaction'] = tf.concat([all_weights['success_embeddings_interaction'], mn], 0)
        all_weights['fails_embeddings_interaction'] = tf.Variable(
            tf.random_normal([self.features_concept, self.hidden_factor], 0.0, 0.01),
            name='fails_embeddings_interaction')  # features_M * K
        all_weights['fails_embeddings_interaction'] = tf.concat([all_weights['fails_embeddings_interaction'], mn], 0)
        all_weights['recent_embeddings_interaction'] = tf.Variable(
            tf.random_normal([3 * self.features_concept, self.hidden_factor], 0.0, 0.01),
            name='recent_embeddings_interaction')  # features_M * K
        all_weights['recent_embeddings_interaction'] = tf.concat([all_weights['recent_embeddings_interaction'], mn], 0)

        # parameters initialization of difficulty regularization terms in factor interaction subspace
        num_layer = len(self.diff_layers_interaction)
        if num_layer > 0:
            glorot = np.sqrt(2.0 / (1 * self.hidden_factor + self.diff_layers_interaction[0]))
            weights1['diff_layer_interaction_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1 * self.hidden_factor, self.diff_layers_interaction[0])), dtype=np.float32)
            weights1['diff_bias_interaction_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.diff_layers_interaction[0])),
                                             dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.diff_layers_interaction[i - 1] + self.diff_layers_interaction[i]))
                weights1['diff_layer_interaction_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.diff_layers_interaction[i - 1], self.diff_layers_interaction[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights1['diff_bias_interaction_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.diff_layers_interaction[i])), dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.diff_layers_interaction[-1] + 1))
            weights1['diff_pred_interaction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.diff_layers_interaction[-1], 1)),
                                                      dtype=np.float32)  # layers[-1] * 1
        else:
            weights1['diff_pred_interaction'] = tf.Variable(
                np.ones((2 * self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1

        # parameters of attention component of ACNN in factor interaction subspace
        num_fenlayer_last = len(self.atten_layers_acnn_interaction)
        if num_fenlayer_last > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor * 3 + self.atten_layers_acnn_interaction[0]))
            weights2['attenweight_acnn_interaction_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor * 3, self.atten_layers_acnn_interaction[0])),
                dtype=np.float32)
            weights2['attenbias_acnn_interaction_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.atten_layers_acnn_interaction[0])),
                dtype=np.float32)  # 1 * layers[0]

            for i in range(1, num_fenlayer_last):
                glorot = np.sqrt(2.0 / (self.atten_layers_acnn_interaction[i - 1] + self.atten_layers_acnn_interaction[i]))
                weights2['attenweight_acnn_interaction_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot,size=(self.atten_layers_acnn_interaction[i - 1],
                                                               self.atten_layers_acnn_interaction[i])), dtype=np.float32)  # layers[i-1]*layers[i]
                weights2['attenbias_acnn_interaction_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.atten_layers_acnn_interaction[i])),
                    dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.atten_layers_acnn_interaction[-1] + 1))
            weights2['attenpred_acnn_interaction'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.atten_layers_acnn_interaction[-1], 3)),
                dtype=np.float32)  # layers[-1] * 1

        # parameters of attention component of interaction_pooling in factor interaction subspace
        num_fenlayer_v = len(self.atten_layers_interaction_pooling)
        if num_fenlayer_v > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor * self.num_field + self.atten_layers_interaction_pooling[0]))
            weights2['attenweight_pooling_interaction_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot,
                                 size=(self.hidden_factor * self.num_field, self.atten_layers_interaction_pooling[0])),
                dtype=np.float32)
            weights2['attenbias_pooling_interaction_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.atten_layers_interaction_pooling[0])),
                dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_fenlayer_v):
                glorot = np.sqrt(
                    2.0 / (self.atten_layers_interaction_pooling[i - 1] + self.atten_layers_interaction_pooling[i]))
                weights2['attenweight_pooling_interaction_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(
                    self.atten_layers_interaction_pooling[i - 1], self.atten_layers_interaction_pooling[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights2['attenbias_pooling_interaction_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.atten_layers_interaction_pooling[i])),
                    dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.atten_layers_interaction_pooling[-1] + 1))
            weights2['attenpred_pooling_interaction'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.atten_layers_interaction_pooling[-1], self.num_field)),
                dtype=np.float32)  # layers[-1] * 1

        # parameters of DNN component of interaction_pooling in factor interaction subspace
        num_layer = len(self.deep_layers_interaction_pooling)
        if num_layer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.deep_layers_interaction_pooling[0]))
            weights2['layer_interaction_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot,
                                 size=(self.hidden_factor, self.deep_layers_interaction_pooling[0])), dtype=np.float32)
            weights2['bias_interaction_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers_interaction_pooling[0])),
                dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(
                    2.0 / (self.deep_layers_interaction_pooling[i - 1] + self.deep_layers_interaction_pooling[i]))
                weights2['layer_interaction_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(
                        self.deep_layers_interaction_pooling[i - 1], self.deep_layers_interaction_pooling[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights2['bias_interaction_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers_interaction_pooling[i])),
                    dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.deep_layers_interaction_pooling[-1] + 1))
            weights2['prediction_dnn_interaction'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers_interaction_pooling[-1], 1)),
                dtype=np.float32)  # layers[-1] * 1
        else:
            weights2['prediction_dnn_interaction'] = tf.Variable(
                np.ones((self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1
        # ************************************************************************

        # ********************* Parameters of Factor subspace ***********************
        # parameters initialization of embeddings in factor subspace
        all_weights['student_embeddings_sum'] = tf.Variable(
            tf.random_normal([self.features_student, self.hidden_factor], 0.0, 0.01),
            name='student_embeddings_sum')  # features_M * K
        all_weights['concept_embeddings_sum'] = tf.Variable(
            tf.random_normal([self.features_concept, self.hidden_factor], 0.0, 0.01),
            name='concept_embeddings_sum')  # features_M * K
        mn1 = tf.Variable(tf.zeros([1, self.hidden_factor]))
        all_weights['concept_embeddings_sum'] = tf.concat([all_weights['concept_embeddings_sum'], mn1], 0)
        all_weights['success_embeddings_sum'] = tf.Variable(
            tf.random_normal([self.features_concept, self.hidden_factor], 0.0, 0.01),
            name='success_embeddings_sum')  # features_M * K
        all_weights['success_embeddings_sum'] = tf.concat([all_weights['success_embeddings_sum'], mn1], 0)
        all_weights['fails_embeddings_sum'] = tf.Variable(
            tf.random_normal([self.features_concept, self.hidden_factor], 0.0, 0.01),
            name='fails_embeddings_sum')  # features_M * K
        all_weights['fails_embeddings_sum'] = tf.concat([all_weights['fails_embeddings_sum'], mn1], 0)
        all_weights['recent_embeddings_sum'] = tf.Variable(
            tf.random_normal([3 * self.features_concept, self.hidden_factor], 0.0, 0.01),
            name='recent_embeddings_sum')  # features_M * K
        all_weights['recent_embeddings_sum'] = tf.concat([all_weights['recent_embeddings_sum'], mn1], 0)

        # parameters initialization of difficulty regularization terms in factor subspace
        num_diff_layer = len(self.diff_layers_sum)
        if num_diff_layer > 0:
            glorot = np.sqrt(2.0 / (1 * self.hidden_factor + self.diff_layers_sum[0]))
            weights1['diff_layer_sum_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1 * self.hidden_factor, self.diff_layers_sum[0])),
                dtype=np.float32)
            weights1['diff_bias_sum_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.diff_layers_sum[0])),
                                                  dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.diff_layers_sum[i - 1] + self.diff_layers_sum[i]))
                weights1['diff_layer_sum_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.diff_layers_sum[i - 1], self.diff_layers_sum[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights1['diff_bias_sum_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.diff_layers_sum[i])),
                    dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.diff_layers_sum[-1] + 1))
            weights1['diff_pred_sum'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.diff_layers_sum[-1], 1)),
                dtype=np.float32)  # layers[-1] * 1
        else:
            weights1['diff_pred_sum'] = tf.Variable(
                np.ones((2 * self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1

        # parameters initialization of attention component of ACNN in factor subspace
        num_fenlayer = len(self.atten_layers_acnn_sum)
        if num_fenlayer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor * 3 + self.atten_layers_acnn_sum[0]))

            weights2['attenweight_acnn_sum_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot,
                                 size=(self.hidden_factor * 3, self.atten_layers_acnn_sum[0])),
                dtype=np.float32)
            weights2['attenbias_acnn_sum_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.atten_layers_acnn_sum[0])),
                dtype=np.float32)  # 1 * layers[0]

            for i in range(1, num_fenlayer):
                glorot = np.sqrt(2.0 / (self.atten_layers_acnn_sum[i - 1] + self.atten_layers_acnn_sum[i]))
                weights2['attenweight_acnn_sum_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot,
                                     size=(self.atten_layers_acnn_sum[i - 1], self.atten_layers_acnn_sum[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights2['attenbias_acnn_sum_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.atten_layers_acnn_sum[i])),
                    dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.atten_layers_acnn_sum[-1] + 1))

            weights2['attenpred_acnn_sum'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.atten_layers_acnn_sum[-1], 3)),
                dtype=np.float32)  # layers[-1] * 1

        # parameters initialization of attention component of sum pooling in factor subspace
        num_fenlayer_sum_pooling = len(self.atten_layers_sum_pooling)
        if num_fenlayer_sum_pooling > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor * self.num_field + self.atten_layers_sum_pooling[0]))

            weights2['attenweight_pooling_sum_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot,
                                 size=(self.hidden_factor * self.num_field, self.atten_layers_sum_pooling[0])),
                dtype=np.float32)
            weights2['attenbias_pooling_sum_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.atten_layers_sum_pooling[0])),
                dtype=np.float32)  # 1 * layers[0]

            for i in range(1, num_fenlayer_sum_pooling):
                glorot = np.sqrt(2.0 / (self.atten_layers_sum_pooling[i - 1] + self.atten_layers_sum_pooling[i]))
                weights2['attenweight_pooling_sum_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot,
                                     size=(self.atten_layers_sum_pooling[i - 1], self.atten_layers_sum_pooling[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights2['attenbias_pooling_sum_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.atten_layers_sum_pooling[i])),
                    dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.atten_layers_sum_pooling[-1] + 1))

            weights2['attenpred_pooling_sum'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.atten_layers_sum_pooling[-1], self.num_field)),
                dtype=np.float32)  # layers[-1] * 1

        # parameters initialization of DNN component of sum pooling in factor subspace
        num_layer_sum_pooling = len(self.deep_layers_sum_pooling)
        if num_layer_sum_pooling > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.deep_layers_sum_pooling[0]))
            weights2['layer_sum_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor, self.deep_layers_sum_pooling[0])),
                dtype=np.float32)
            weights2['bias_sum_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers_sum_pooling[0])),
                                               dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer_sum_pooling):
                glorot = np.sqrt(2.0 / (self.deep_layers_sum_pooling[i - 1] + self.deep_layers_sum_pooling[i]))
                weights2['layer_sum_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.deep_layers_sum_pooling[i - 1], self.deep_layers_sum_pooling[i])),
                    dtype=np.float32)  # layers[i-1]*layers[i]
                weights2['bias_sum_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers_sum_pooling[i])),
                    dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.deep_layers_sum_pooling[-1] + 1))
            weights2['prediction_dnn_sum'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers_sum_pooling[-1], 1)),
                dtype=np.float32)  # layers[-1] * 1
        else:
            weights2['prediction_dnn_sum'] = tf.Variable(
                np.ones((self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1
        # ************************************************************************

        return all_weights, weights1, weights2

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.student_features: data['X_student'], self.train_labels: data['Y'],
                     self.success_features: data['X_success'], self.success_nums: data['X_success_nums'],
                     self.fails_features: data['X_fails'], self.fails_nums: data['X_fails_nums'],
                     self.concept_nums: data['X_concept_nums'], self.question_features: data['X_question'],
                     self.concept_features: data['X_concept'], self.dropout_diff_interaction: self.keep_diff_interaction,
                     self.difficulty_labels: data['Y_diff'], self.train_phase: True,
                     self.dropout_atten_sum: self.keep_atten_sum_pooling, self.dropout_dnn_sum: self.keep_deep_sum_pooling,
                     self.dropout_diff_sum: self.keep_diff_sum, self.dropout_atten_interaction: self.keep_atten_interaction_pooling,
                     self.dropout_dnn_interaction: self.keep_deep_interaction_pooling, self.recent_features: data['X_recent'],
                     self.recent_nums: data['X_recent_nums'], self.dropout_atten_acnn_sum: self.keep_atten_acnn_sum,
                     self.dropout_atten_acnn_interaction: self.keep_atten_acnn_interaction,
                     self.recent_interval: data['X_recent_interval']}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X_student, X_question, X_concept, Y, Y_diff, X_concept_nums, X_success, X_success_nums, X_fails, X_fails_nums = [], [], [], [], [], [], [], [], [], []
        X_recent, X_recent_nums = [], []
        X_recent_interval = []
        # forward get sample
        i = start_index
        while len(X_student) < batch_size and i < len(data['X_student']):
            if len(data['X_student'][i]) == len(data['X_student'][start_index]):
                Y.append([data['Y'][i]])
                Y_diff.append([data['Y_diff'][i]])
                X_student.append(data['X_student'][i])
                X_question.append(data['X_question'][i])
                X_concept.append(data['X_concept'][i])
                X_success.append(data['X_success'][i])
                X_fails.append(data['X_fails'][i])
                X_concept_nums.append(data['X_concept_nums'][i])
                X_success_nums.append(data['X_success_nums'][i])
                X_fails_nums.append(data['X_fails_nums'][i])
                X_recent.append(data['X_recent'][i])
                X_recent_nums.append(data['X_recent_nums'][i])
                X_recent_interval.append(data['X_recent_interval'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X_student) < batch_size and i >= 0:
            if len(data['X_student'][i]) == len(data['X_student'][start_index]):
                Y.append([data['Y'][i]])
                Y_diff.append([data['Y_diff'][i]])
                X_student.append(data['X_student'][i])
                X_question.append(data['X_question'][i])
                X_concept.append(data['X_concept'][i])
                X_success.append(data['X_success'][i])
                X_fails.append(data['X_fails'][i])
                X_concept_nums.append(data['X_concept_nums'][i])
                X_success_nums.append(data['X_success_nums'][i])
                X_fails_nums.append(data['X_fails_nums'][i])
                X_recent.append(data['X_recent'][i])
                X_recent_nums.append(data['X_recent_nums'][i])
                X_recent_interval.append(data['X_recent_interval'][i])
                i = i - 1
            else:
                break
        return {'X_student': X_student, 'X_question': X_question, 'X_concept': X_concept, 'Y': Y, 'X_concept_nums': X_concept_nums,
                'X_success': X_success, 'X_success_nums': X_success_nums,
                'X_fails': X_fails, 'X_fails_nums': X_fails_nums, 'Y_diff': Y_diff, 'X_recent': X_recent,
                'X_recent_nums': X_recent_nums, 'X_recent_interval': X_recent_interval}

    def shuffle_in_unison_scary(self, a, b, c, d, e, f, g, h, i, j, k, l, m):
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
        np.random.set_state(rng_state)
        np.random.shuffle(m)

    def train(self, Train_data, Test_data):  # fit a dataset
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X_student'], Train_data['X_question'], Train_data['X_concept'],
                                         Train_data['Y'], Train_data['X_concept_nums'],
                                         Train_data['X_success'], Train_data['Y_diff'],
                                         Train_data['X_success_nums'], Train_data['X_fails'], Train_data['X_fails_nums'],
                                         Train_data['X_recent'], Train_data['X_recent_nums'], Train_data['X_recent_interval'])
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
                print("Epoch %d [%.1f s]\ttrain_acc=%.4f, test_acc=%.4f [%.1f s]"
                            % (epoch + 1, t2 - t1, train_acc, test_acc, time() - t2))
                print("Epoch %d [%.1f s]\ttrain_auc=%.4f, test_auc=%.4f [%.1f s]"
                            % (epoch + 1, t2 - t1, train_auc, test_auc, time() - t2))
                self.logger.info("Epoch %d [%.1f s]\ttrain_acc=%.4f, test_acc=%.4f [%.1f s]"
                            % (epoch + 1, t2 - t1, train_acc, test_acc, time() - t2))
                self.logger.info("Epoch %d [%.1f s]\ttrain_auc=%.4f, test_auc=%.4f [%.1f s]"
                            % (epoch + 1, t2 - t1, train_auc, test_auc, time() - t2))

    def eva_termination(self, valid):
        if self.loss_type == 'square_loss':
            if len(valid) > 5:
                if valid[-1] < valid[-2] and valid[-2] < valid[-3] and valid[-3] < valid[-4] and valid[-4] < valid[-5]:
                    return True
        else:
            if len(valid) > 10:
                if valid[-1] < valid[-2] < valid[-3] < valid[-4] < valid[-5] < valid[-6] < valid[-7] < valid[-8] < \
                        valid[-9] < valid[-10]:
                    return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        feed_dict = {self.student_features: [item for item in data['X_student']], self.train_labels: [[y] for y in data['Y']],
                     self.question_features: [item for item in data['X_question']],
                     self.concept_features: [item for item in data['X_concept']],
                     self.concept_nums: [item for item in data['X_concept_nums']],
                     self.difficulty_labels: [[y] for y in data['Y_diff']],
                     self.success_nums: [item for item in data['X_success_nums']],
                     self.success_features: [item for item in data['X_success']],
                     self.fails_nums: [item for item in data['X_fails_nums']],
                     self.fails_features: [item for item in data['X_fails']],
                     self.dropout_diff_interaction: self.no_diff_interaction_dropout, self.dropout_atten_sum: self.no_atten_acnn_sum_dropout,
                     self.dropout_dnn_sum: self.no_deep_sum_pooling_dropout, self.train_phase: False,
                     self.dropout_diff_sum: self.no_diff_sum_dropout,
                     self.dropout_atten_interaction: self.no_atten_interaction_pooling_dropout,
                     self.dropout_dnn_interaction: self.no_deep_interaction_pooling_dropout,
                     self.recent_features: [item for item in data['X_recent']],
                     self.recent_nums: [item for item in data['X_recent_nums']],
                     self.dropout_atten_acnn_sum: self.no_atten_acnn_sum_dropout,
                     self.dropout_atten_acnn_interaction: self.no_atten_acnn_interaction_dropout,
                     self.recent_interval: [item for item in data['X_recent_interval']]}
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
