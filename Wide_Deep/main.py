import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from Wide_Deep import Wide_Deep


def cross_columns(x_cols):
    crossed_cols = dict()
    col_names = ['_'.join(c) for c in x_cols]
    for cname, cols in zip(col_names, x_cols):
        crossed_cols[cname] = cols
    return crossed_cols


def val2idx(values):
    value_types = np.unique(values)
    val_to_idx = dict()
    for i, v in enumerate(value_types):
        val_to_idx[v] = i
    idx_values = [val_to_idx[v] for v in values]
    return idx_values


def load_dataset(file_path):
    # read data
    train_data = pd.read_csv(os.path.join(file_path, 'train.csv'))
    test_data = pd.read_csv(os.path.join(file_path, 'test.csv'))
    train_data['type'] = 'train'
    test_data['type'] = 'test'
    all_data = pd.concat([train_data, test_data], axis=0)

    cate_cols = ['user_type', 'item_type']
    cont_cols = [c for c in all_data.columns if c not in ['userid', 'itemid', 'label', 'type'] and c not in cate_cols]
    cate_data = all_data[cate_cols]
    cont_data = pd.DataFrame(StandardScaler().fit_transform(all_data[cont_cols]), columns=cont_cols)
    other_data = all_data[['type', 'label']]
    # cross feature
    corss_cols = [['user_type', 'item_type']]
    corss_cols = cross_columns(corss_cols)
    for k, v in corss_cols.items():
        all_data[k] = val2idx(all_data[v].astype(str).agg('-'.join, axis=1))
    cross_data = all_data[list(corss_cols.keys())]

    all_data = pd.concat([cross_data.reset_index(),
                          cate_data.reset_index(),
                          cont_data.reset_index(),
                          other_data.reset_index()], axis=1)

    train_data = all_data[all_data['type'] == 'train']
    X_train_cross = train_data[list(corss_cols.keys())].values
    X_train_cate = train_data[cate_cols].values
    X_train_cont = train_data[cont_cols].values
    y_train = train_data['label'].values

    test_data = all_data[all_data['type'] == 'test']
    X_test_cross = test_data[list(corss_cols.keys())].values
    X_test_cate = test_data[cate_cols].values
    X_test_cont = test_data[cont_cols].values
    y_test = test_data['label'].values

    cross_list = []
    for i in range(cross_data.values.shape[1]):
        u = np.unique(cross_data.values[:, i]).shape[0] + 1
        cross_list.append(u)

    cate_list = []
    for i in range(cate_data.values.shape[1]):
        u = np.unique(cate_data.values[:, i]).shape[0] + 1
        cate_list.append(u)

    return X_train_cross, X_train_cate, X_train_cont, y_train, X_test_cross, X_test_cate, X_test_cont, y_test, cross_list, cate_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='The directory of input', type=str, default='')
    parser.add_argument('--model_name', help='The directory of model', type=str, default='Wide_Deep.ckpt')
    parser.add_argument('--embed_size', help='size for embedding user and item', type=int, default=8)
    parser.add_argument('--hidden_layers', help='hidden layers', type=list, default=[32, 64, 128])
    parser.add_argument('--batch_size', help='size of mini-batch', type=int, default=128)
    parser.add_argument('--epoch', help='number of epochs', type=int, default=10)
    parser.add_argument('--regular_rate', help='regular rate', type=float, default=0.1)
    parser.add_argument('--learning_rate', help='learning rate', type=float, default=0.001)
    parser.add_argument('--dropout', help='dropout rate', type=float, default=0.2)
    parser.add_argument('--decay_steps', help='decay steps', type=int, default=10000)
    parser.add_argument('--decay_rate', help='decay rate', type=float, default=0.99)
    args = parser.parse_args(args=[])

    # load data set
    X_train_cross, X_train_cate, X_train_cont, y_train, X_test_cross, X_test_cate, X_test_cont, y_test, cross_list, cate_list = load_dataset(args.input_dir)

    cross_num = X_train_cross.shape[1]
    cate_num = X_train_cate.shape[1]
    cont_num = X_train_cont.shape[1]

    tf.reset_default_graph()
    with tf.Session() as sess:
        # define model
        model = Wide_Deep.Wide_Deep(args, cross_num, cate_num, cont_num, cross_list, cate_list)
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
            batch, total = 0, X_train_cate.shape[0]
            while end < total:
                start = batch * args.batch_size
                end = start + args.batch_size if start + args.batch_size < total else total
                loss = model.run(sess,
                                 X_train_cross[start:end],
                                 X_train_cate[start:end],
                                 X_train_cont[start:end],
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
            batch, total = 0, X_test_cate.shape[0]
            while end < total:
                start = batch * args.batch_size
                end = start + args.batch_size if start + args.batch_size < total else total
                actual, predict = model.run(sess,
                                            X_test_cross[start:end],
                                            X_test_cate[start:end],
                                            X_test_cont[start:end],
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
