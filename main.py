import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
import argparse
from data_loader import Data_loader
import MF_DAKT
import logging

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MF-DAKT.")
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='Pre-train flag. 0: train from scratch; 1: load from pretrain file')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=128,
                        help='Number of hidden factors.')
    parser.add_argument('--diff_layers_interaction', nargs='?', default='[128]',
                        help="Size of each layer in the diffculty regularization in factor interaction subspace.")
    parser.add_argument('--keep_diff_interaction', nargs='?', default='[0.3,0.3]',
                        help='Keep probability for the diffculty regularization in factor interaction subspace.')
    parser.add_argument('--diff_layers_sum', nargs='?', default='[256]',
                        help="Size of each layer in the diffculty regularization in factor subspace.")
    parser.add_argument('--keep_diff_sum', nargs='?', default='[0.3]',
                        help='Keep probability for the diffculty regularization in factor subspace.')
    parser.add_argument('--atten_layers_sum_pooling', nargs='?', default='[256]',
                        help="Size of attention layer in the sum pooling of factor subspace.")
    parser.add_argument('--keep_atten_sum_pooling', nargs='?', default='[0.4,0.3]',
                        help='Keep probability for the attention layer in the sum pooling of factor subspace.')
    parser.add_argument('--atten_layers_interaction_pooling', nargs='?', default='[256]',
                        help="Size of attention layer in the interaction pooling of factor interaction subspace.")
    parser.add_argument('--keep_atten_interaction_pooling', nargs='?', default='[0.4,0.3]',
                        help='Keep probability for the attention layer in the interaction pooling of factor interaction subspace.')
    parser.add_argument('--atten_layers_acnn_sum', nargs='?', default='[256]',
                        help="Size of attention layer in the ACNN of factor subspace.")
    parser.add_argument('--keep_atten_acnn_sum', nargs='?', default='[0.4,0.3]',
                        help='Keep probability of attention layer in the ACNN of factor subspace.')
    parser.add_argument('--atten_layers_acnn_interaction', nargs='?', default='[256]',
                        help="Size of attention layer in the ACNN of factor interaction subspace.")
    parser.add_argument('--keep_atten_acnn_interaction', nargs='?', default='[0.4,0.3]',
                        help='Keep probability of attention layer in the ACNN of factor interaction subspace.')
    parser.add_argument('--deep_layers_sum_pooling', nargs='?', default='[32]',
                        help="Size of deep layer in the sum pooling of factor subspace.")
    parser.add_argument('--keep_deep_sum_pooling', nargs='?', default='[0.4,0.2]',
                        help='Keep probability of deep layer in the sum pooling of factor subspace.')
    parser.add_argument('--deep_layers_interaction_pooling', nargs='?', default='[32]',
                        help="Size of deep layer in the interaction pooling of factor interaction subspace.")
    parser.add_argument('--keep_deep_interaction_pooling', nargs='?', default='[0.4]',
                        help='Keep probability of deep layer in the interaction pooling of factor interaction subspace.')
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

    return parser.parse_args()


if __name__ == '__main__':

    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('2021-11-2-github-np-pretrain.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Data loading
    data = Data_loader()
    list_acc, list_auc, list_acc_best, list_auc_best = [], [], [], []

    dic, num_student, num_question, num_concept, length = data.data_load()
    print("student_nums=%d, question_nums=%d, concept_max_nums=%d" % (num_student, num_question, num_concept))
    logger.info("student_nums=%d, question_nums=%d, concept_max_nums=%d" % (num_student, num_question, num_concept))
    print("Data loaded successfully-----")
    logger.info("Data loaded successfully-----")

    X_train_student, X_train_question, X_train_concept, X_train_concept_nums, X_train_success, X_train_fails, X_train_success_nums, X_train_fails_nums, \
    X_train_recent, X_train_recent_nums, X_train_recent_interval, Y_train, Y_train_diff = [], [], [], [], [], [], [], [], [], [], [], [], []
    X_test_student, X_test_question, X_test_concept, X_test_concept_nums, X_test_success, X_test_fails, X_test_success_nums, X_test_fails_nums, \
    X_test_recent, X_test_recent_nums, X_test_recent_interval, Y_test, Y_test_diff = [], [], [], [], [], [], [], [], [], [], [], [], []

    train_rate = 0.8
    test_rate = 0.2

    import random
    for user in dic:
        random.shuffle(dic[user])
        length_tmp = len(dic[user])
        for index in range(int(train_rate * length_tmp)):
            X_train_student.append(dic[user][index][0])
            X_train_question.append(dic[user][index][1])
            X_train_concept.append(dic[user][index][2])
            X_train_concept_nums.append(dic[user][index][3])
            X_train_success.append(dic[user][index][4])
            X_train_fails.append(dic[user][index][5])
            X_train_success_nums.append(dic[user][index][6])
            X_train_fails_nums.append(dic[user][index][7])
            X_train_recent.append(dic[user][index][8])
            X_train_recent_nums.append(dic[user][index][9])
            X_train_recent_interval.append(dic[user][index][10])
            Y_train.append(dic[user][index][11])
            Y_train_diff.append(dic[user][index][12])
        for index in range(length_tmp - int(test_rate * length_tmp), length_tmp):
            X_test_student.append(dic[user][index][0])
            X_test_question.append(dic[user][index][1])
            X_test_concept.append(dic[user][index][2])
            X_test_concept_nums.append(dic[user][index][3])
            X_test_success.append(dic[user][index][4])
            X_test_fails.append(dic[user][index][5])
            X_test_success_nums.append(dic[user][index][6])
            X_test_fails_nums.append(dic[user][index][7])
            X_test_recent.append(dic[user][index][8])
            X_test_recent_nums.append(dic[user][index][9])
            X_test_recent_interval.append(dic[user][index][10])
            Y_test.append(dic[user][index][11])
            Y_test_diff.append(dic[user][index][12])
    Train_data = {'X_student': X_train_student, 'X_question': X_train_question, 'X_concept': X_train_concept, 'Y': Y_train,
                  'X_concept_nums': X_train_concept_nums, 'X_success': X_train_success, 'X_success_nums': X_train_success_nums,
                  'X_fails': X_train_fails, 'X_fails_nums': X_train_fails_nums, 'Y_diff': Y_train_diff,
                  'X_recent': X_train_recent, 'X_recent_nums': X_train_recent_nums,
                  'X_recent_interval': X_train_recent_interval}
    Test_data = {'X_student': X_test_student, 'X_question': X_test_question, 'X_concept': X_test_concept, 'Y': Y_test,
                 'X_concept_nums': X_test_concept_nums, 'X_success': X_test_success,
                 'X_success_nums': X_test_success_nums, 'X_fails': X_test_fails, 'X_fails_nums': X_test_fails_nums,
                 'Y_diff': Y_test_diff, 'X_recent': X_test_recent, 'X_recent_nums': X_test_recent_nums,
                 'X_recent_interval': X_test_recent_interval}
    print("Data is splitted successfully-----")
    logger.info("Data is splitted successfully-----")

    args = parse_args()

    # Training and evaluate
    t1 = time()
    model = MF_DAKT.MF_DAKT(num_student, num_question, num_concept, args, length, logger)
    model.train(Train_data, Test_data)
    best_auc_score = max(model.test_auc)
    best_epoch_auc = model.test_auc.index(best_auc_score)
    print("Best Iter(test_acc)= %d\t train(acc) = %.4f, test(acc) = %.4f [%.1f s]"
                % (best_epoch_auc + 1, model.train_acc[best_epoch_auc], model.test_acc[best_epoch_auc], time() - t1))
    logger.info("Best Iter(test_acc)= %d\t train(acc) = %.4f, test(acc) = %.4f [%.1f s]"
                % (best_epoch_auc + 1, model.train_acc[best_epoch_auc], model.test_acc[best_epoch_auc], time() - t1))
    print("Best Iter(test_auc)= %d\t train(auc) = %.4f, test(auc) = %.4f [%.1f s]"
                % (
                    best_epoch_auc + 1, model.train_auc[best_epoch_auc], model.test_auc[best_epoch_auc],
                    time() - t1))
    logger.info("Best Iter(test_auc)= %d\t train(auc) = %.4f, test(auc) = %.4f [%.1f s]"
                % (
                    best_epoch_auc + 1, model.train_auc[best_epoch_auc], model.test_auc[best_epoch_auc],
                    time() - t1))
    list_auc.append(model.test_auc[-1])
    list_auc_best.append(best_auc_score)
print("Average(Best)\t test(acc) = %.4f+/-%.4f, test(auc) = %.4f+/-%.4f" % (
    np.mean(list_acc_best), np.std(list_acc_best), np.mean(list_auc_best), np.std(list_auc_best)))
logger.info("Average(Best)\t test(acc) = %.4f+/-%.4f, test(auc) = %.4f+/-%.4f" % (
    np.mean(list_acc_best), np.std(list_acc_best), np.mean(list_auc_best), np.std(list_auc_best)))
