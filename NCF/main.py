import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

import NCF


def reindex(data):
    data_map = {}
    for i in range(len(data)):
        data_map[data[i]] = i
    return data_map


def load_dataset(file_path):
    # read data
    cols = ['userid', 'itemid', 'label']
    train_data = pd.read_csv(os.path.join(file_path, 'train.csv'))[cols]
    test_data = pd.read_csv(os.path.join(file_path, 'test.csv'))[cols]
    all_data = pd.concat([train_data, test_data], axis=0)

    # reindex
    user_map = reindex(all_data['userid'].unique())
    item_map = reindex(all_data['itemid'].unique())
    all_data['userid'] = all_data['userid'].map(lambda x: user_map[x])
    all_data['itemid'] = all_data['itemid'].map(lambda x: item_map[x])

    user_set = set(all_data['userid'].unique())
    item_set = set(all_data['itemid'].unique())
    user_size = len(user_set)
    item_size = len(item_set)

    train_data['userid'] = train_data['userid'].map(lambda x: user_map[x])
    train_data['itemid'] = train_data['itemid'].map(lambda x: item_map[x])
    test_data['userid'] = test_data['userid'].map(lambda x: user_map[x])
    test_data['itemid'] = test_data['itemid'].map(lambda x: item_map[x])

    X_train = train_data[['userid', 'itemid']].values
    y_train = train_data['label'].values
    X_test = test_data[['userid', 'itemid']].values
    y_test = test_data['label'].values

    return X_train, y_train, X_test, y_test, user_size, item_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='')
    parser.add_argument('--model_name', help='The directory of model', type=str, default='NCF.ckpt')
    parser.add_argument('--embed_size', help='the size for embedding user and item', type=int, default=16)
    parser.add_argument('--batch_size', help='size of mini-batch', type=int, default=128)
    parser.add_argument('--epoch', help='number of epochs', type=int, default=10)
    parser.add_argument('--regular_rate', help='regular rate', type=float, default=0.1)
    parser.add_argument('--learning_rate', help='learning rate', type=float, default=0.001)
    parser.add_argument('--dropout', help='dropout rate', type=float, default=0.2)
    parser.add_argument('--decay_steps', help='decay steps', type=int, default=10000)
    parser.add_argument('--decay_rate', help='decay rate', type=float, default=0.99)
    args = parser.parse_args(args=[])

    # load data set
    X_train, y_train, X_test, y_test, user_size, item_size = load_dataset(args.input_dir)

    tf.reset_default_graph()
    with tf.Session() as sess:
        # define model
        model = NCF.NCF(args, user_size, item_size)
        model.build()

        ckpt = tf.train.get_checkpoint_state(os.path.join(args.input_dir, args.model_name))
        if ckpt:
            print('Loading model parameters from %s' % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Creating model with inital parameters')
            sess.run(tf.global_variables_initializer())

        step = 0
        for epoch in range(args.epoch):
            start_time = time.time()
            # train
            losses = []
            start, end = 0, args.batch_size
            batch, total = 0, X_train.shape[0]
            while end < total:
                start = batch * args.batch_size
                end = start + args.batch_size if start + args.batch_size < total else total
                loss = model.run(sess,
                                 X_train[start:end],
                                 y_train[start:end],
                                 True,
                                 step)
                losses.append(loss)
                step += 1
                batch += 1
            end_time = time.time()
            print("Epoch %d training: loss = %.4f, took: %s" %
                  (epoch + 1, np.mean(losses), time.strftime("%H: %M: %S", time.gmtime(end_time - start_time))))

            # test
            actuals, predicts = [], []
            start, end = 0, args.batch_size
            batch, total = 0, X_test.shape[0]
            while end < total:
                start = batch * args.batch_size
                end = start + args.batch_size if start + args.batch_size < total else total
                actual, predict = model.run(sess,
                                            X_test[start:end],
                                            y_test[start:end],
                                            False)
                actuals.append(actual)
                predicts.append(predict)
                losses.append(loss)
                batch += 1
            end_time = time.time()
            actuals = np.array([l for sub in actuals for l in sub])
            predicts = np.array([l for sub in predicts for l in sub])
            accuracy = float(len(actuals == predicts)) / float(len(actuals))
            print("Epoch %d testing: Accuracy = %.4f, took: %s" %
                  (epoch + 1, accuracy, time.strftime("%H: %M: %S", time.gmtime(end_time - start_time))))

            # save model
            model.saver.save(sess, os.path.join(args.input_dir, args.model_name))
