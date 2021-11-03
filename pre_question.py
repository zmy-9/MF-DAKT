import tensorflow as tf
import numpy as np
import math
from scipy import sparse
import pickle
from time import time
import logging

# ------------------------------------
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('2021-11-pre-training-dims128.log')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.addHandler(ch)
# ------------------------------------


def load_pkl(pkl_name):
    with open(pkl_name + '.pkl', 'rb') as f:
        return pickle.load(f)


# load question graph information
# load question-relations
question_question_relation = sparse.load_npz('ednet/pro_pro.npz')  # read question relation matrix
print('The edges of questions are %d' % question_question_relation.nnz)
logger.info('The edges of questions are %d' % question_question_relation.nnz)
question_num = question_question_relation.shape[0]
print('question-nums %d' % question_num)
logger.info('question-nums %d' % question_num)
question_question_dense = question_question_relation.toarray()  # num_questions * num_questions

# load question difficulty-levles
diff = load_pkl('ednet/problem_difficulty')
print('question-nums-diff %d' % len(diff))  # num_questions * 1
logger.info('question-nums-diff %d' % len(diff))

# set the parameters
keep_prob = 0.5
lr = 0.001
bs = 256
epochs = 20
embed_dim = 128

# set placeholder
tf_question = tf.placeholder(tf.int32, [None])  # input question id
tf_question_question_targets = tf.placeholder(tf.float32, [None, question_num], name='tf_question_question')  # input question similarity
tf_keep_prob = tf.placeholder(tf.float32, [1], name='tf_keep_prob')
tf_diff = tf.placeholder(tf.float32, [None], name='tf_diff')  # input question difficulty

# initialize question representations of the factor interaction subspace
question_embedding_matrix = tf.Variable(tf.random_normal([question_num, embed_dim], 0.0, 0.01), name='item_embeddings')
# initialize question representations of the factor subspace
question_bias_matrix = tf.Variable(tf.random_normal([question_num, embed_dim], 0.0, 0.01), name='item_bias')

# embeded question
question_embed = tf.nn.embedding_lookup(question_embedding_matrix, tf_question)  # [bs, embed_dim]
question_bias = tf.nn.embedding_lookup(question_bias_matrix, tf_question)  # [bs, embed_dim]

# calculate the difficulty of question embeddings ------------------------------
question_diff_embeddings = tf.reshape(tf.layers.dense(question_embed, units=1), [-1])
question_diff_bias = tf.reshape(tf.layers.dense(question_bias, units=1), [-1])

# squared loss of question difficulty-level
mse_emb_diff = tf.reduce_mean(tf.square(question_diff_embeddings - tf_diff))
mse_bias_diff = tf.reduce_mean(tf.square(question_diff_bias - tf_diff))

# calculate the similarity based on the inner product or cosine
question_question_logits_embeddings = tf.reshape(tf.matmul(question_embed, tf.transpose(question_embedding_matrix)), [-1])
question_question_logits_bias = tf.reshape(tf.matmul(question_bias, tf.transpose(question_bias_matrix)), [-1])

# question-relation labels
tf_question_question_targets_reshape = tf.reshape(tf_question_question_targets, [-1])

# squared loss of question relations
loss_question_question_embeddings = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_question_question_targets_reshape, logits=question_question_logits_embeddings))
loss_question_question_bias = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_question_question_targets_reshape, logits=question_question_logits_bias))

# the total loss, and we can set different weight on them
loss = loss_question_question_embeddings + loss_question_question_bias + mse_emb_diff + mse_bias_diff

# begin training
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)
print('finish building graph')
logger.info('finish building graph')


saver = tf.train.Saver()
train_steps = int(math.ceil(question_num / float(bs)))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        t_initial = time()
        train_loss = 0
        for m in range(train_steps):
            b, e = m * bs, min((m + 1) * bs, question_num)
            batch_question = np.arange(b, e).astype(np.int32)  # read batch question id
            batch_question_question_targets = question_question_dense[b:e, :]  # read batch question-relation labels
            batch_diff = [diff[i] for i in range(b, e)]
            feed_dict = {tf_question: batch_question,
                         tf_question_question_targets: batch_question_question_targets,
                         tf_keep_prob: [keep_prob],
                         tf_diff: batch_diff}

            _, loss_ = sess.run([train_op, loss], feed_dict=feed_dict)
            train_loss += loss_

        train_loss /= train_steps
        print("epoch %d, loss %.8f [%.1f s]" % (i, train_loss, time() - t_initial))
        logger.info("epoch %d, loss %.8f [%.1f s]" % (i, train_loss, time() - t_initial))
    saver.save(sess, 'ednet/pre_train/problem_embedding_pp_diff_128dims')

    print('finish training')
    logger.info("finish training")
