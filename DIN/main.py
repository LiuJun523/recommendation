import os
import sys
import time
import argparse
import pickle
import tensorflow as tf
import random

from DIN import DIN, data


def calc_auc(raw_arr):
    arr = sorted(raw_arr, key=lambda d: d[2])
    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0

    for record in arr:
        fp2 += record[0]
        tp2 += record[1]

        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None


def _auc_arr(score):
    score_p = score[:, 0]
    score_n = score[:, 1]
    score_arr = []
    for s in score_p.tolist():
        score_arr.append([0, 1, s])
    for s in score_n.tolist():
        score_arr.append([1, 0, s])
    return score_arr


def _eval(sess, model, args):
    auc_sum = 0.0
    score_arr = []
    for _, uij in data.DataInputTest(test_set, args.test_batch_size):
        auc_, score_ = model.eval(sess, uij)
        score_arr += _auc_arr(score_)
        auc_sum += auc_ * len(uij[0])

    test_gauc = auc_sum / len(test_set)

    AUC = calc_auc(score_arr)

    global best_auc
    if best_auc < test_gauc:
        best_auc = test_gauc
        model.save(sess, args.model_name)
    return test_gauc, AUC


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='')
    parser.add_argument('--model_name', help='The directory of model', type=str, default='DIN.ckpt')
    parser.add_argument('--train_batch_size', help='size of train batch', type=int, default=32)
    parser.add_argument('--test_batch_size', help='size for test batch', type=int, default=512)
    parser.add_argument('--hidden_units', help='hidden units', type=int, default=32)
    parser.add_argument('--epoch', help='number of epochs', type=int, default=10)
    parser.add_argument('--learning_rate', help='learning rate', type=float, default=0.1)
    args = parser.parse_args(args=[])

    with open(args.input_dir, 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)

    best_auc = 0.0

    tf.reset_default_graph()
    with tf.Session() as sess:
        model = DIN.DIN(args, user_count, item_count, cate_count, cate_list)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        start_time = time.time()

        for _ in range(args.epoch):
            random.shuffle(train_set)
            epoch_size = round(len(train_set) / args.train_batch_size)
            loss_sum = 0.0

            for _, uij in data.DataInput(train_set, args.train_batch_size):
                loss = model.train(sess, uij, args.learning_rate)
                loss_sum += loss

                if model.global_step.eval() % 10 == 0:
                    test_gauc, AUC = _eval(sess, model, args)

                    if model.global_step.eval() % 1000 == 0:
                        test_gauc, AUC = _eval(sess, model)
                        print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                              (model.global_epoch_step.eval(), model.global_step.eval(),
                               loss_sum / 1000, test_gauc, AUC))
                        sys.stdout.flush()
                        loss_sum = 0.0

                print('Epoch %d DONE\tCost time: %.2f' %
                      (model.global_epoch_step.eval(), time.time() - start_time))
                sys.stdout.flush()
                model.global_epoch_step_op.eval()

            print('best test_gauc:', best_auc)
            sys.stdout.flush()
