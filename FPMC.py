import math
import numpy as np
import queue
import threading
from numba import jit


class PredThread(threading.Thread):
    def __init__(self, input_queue, output_queue, n_pred, pred_list, VUI, VIU, VIL, VLI):
        threading.Thread.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.n_pred = n_pred
        self.pred_list = pred_list
        self.VUI = VUI
        self.VIU = VIU
        self.VIL = VIL
        self.VLI = VLI

    def run(self):
        while True:
            idx = self.input_queue.get()
            if idx % 10000 == 0:
                print(idx)

            u = self.pred_list[0][idx]
            b_tm = self.pred_list[2][idx][self.pred_list[2][idx] != -1]
            scores = np.dot(self.VUI[u], self.VIU.T) + np.dot(self.VIL, self.VLI[b_tm].T).sum(1) / len(b_tm)
            ind = np.argpartition(scores, (-1 * self.n_pred))[(-1 * self.n_pred):]
            pred = ind[np.argsort(scores[ind])][::-1]
            score = scores[pred]
            # normalization
            s_max, s_min = max(score), min(score)
            if s_max - s_min > 0:
                for i in range(len(score)):
                    score[i] = (score[i] - s_min) / (s_max - s_min)
            else:
                for i in range(len(score)):
                    score[i] = 1

            self.output_queue.put((u, (pred, score)))
            self.input_queue.task_done()


class FPMC(object):
    def __init__(self, n_user, n_item, n_factor, learning_rate, regular):
        self.user2id = dict()
        self.item2id = dict()
        self.n_user = n_user
        self.n_item = n_item
        self.n_factor = n_factor
        self.learning_rate = learning_rate
        self.regular = regular

    def init_model(self, std=0.01):
        self.VUI = np.random.normal(0, std, size=(self.n_user, self.n_factor))
        self.VIU = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VIL = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VLI = np.random.normal(0, std, size=(self.n_item, self.n_factor))

    def load_model(self, model_save_dir):
        if model_save_dir is None:
            return
        model = np.load(model_save_dir)
        self.VUI = model['VUI']
        self.VIU = model['VIU']
        self.VIL = model['VIL']
        self.VLI = model['VLI']

    def learn_epoch(self, data_list, neg_batch_size, epoch):
        VUI, VIU, VIL, VLI = learn_epoch_jit(data_list, neg_batch_size, epoch, np.array(list(self.item2id.values())),
                                             self.VUI, self.VIU, self.VIL, self.VLI, self.learning_rate, self.regular)
        self.VUI = VUI
        self.VIU = VIU
        self.VIL = VIL
        self.VLI = VLI

    def evaluation(self, data_list):
        acc_3, acc_5, acc_10 = evaluation_jit(data_list, self.VUI, self.VIU, self.VIL, self.VLI)
        return acc_3, acc_5, acc_10

    def train(self, train_set, test_set=None, n_epoch=10, neg_batch_size=10, model_save_dir='./fpmc_model.npz', eval_per_epoch=False):
        train_list = data_to_list(train_set)
        test_list = None
        if test_set is not None:
            test_list = data_to_list(test_set)

        for epoch in range(1, n_epoch + 1):
            print('Epoch %d start' % epoch)
            self.learn_epoch(train_list, neg_batch_size, epoch)

            if eval_per_epoch:
                train_acc_3, train_acc_5, train_acc_10 = evaluation_jit(train_list, self.VUI, self.VIU, self.VIL, self.VLI)
                if test_list is not None:
                    test_acc_3, test_acc_5, test_acc_10 = evaluation_jit(test_list, self.VUI, self.VIU, self.VIL, self.VLI)
                    print('Train set:\tacc_3:%.4f\tacc_5:%.4f\tacc_10:%.4f' % (train_acc_3, train_acc_5, train_acc_10))
                    print('Test set:\tacc_3:%.4f\tacc_5:%.4f\tacc_10:%.4f' % (test_acc_3, test_acc_5, test_acc_10))
                else:
                    print('Train set:\tacc_3:%.4f\tacc_5:%.4f\tacc_10:%.4f' % (train_acc_3, train_acc_5, train_acc_10))
            else:
                print('Epoch %d done' % epoch)

            # save model
            np.savez(model_save_dir, VUI=self.VUI, VIU=self.VIU, VIL=self.VIL, VLI=self.VLI)

    def predict(self, pred_set, n_thread, n_pred):
        pred_list = data_to_list(pred_set)
        set_len = len(pred_list[0])
        input_queue = queue.Queue()
        output_queue = queue.PriorityQueue()
        # start multi threading
        for i in range(n_thread):
            thread = PredThread(input_queue, output_queue, n_pred, pred_list, self.VUI, self.VIU, self.VIL, self.VLI)
            thread.daemon = True
            thread.start()
        # put data into the input_queue
        for idx in range(set_len):
            input_queue.put(idx)
        input_queue.join()
        # get result from the output_queue
        pred_list = []
        while not output_queue.empty():
            try:
                u, (pred, score) = output_queue.get_nowait()
                pred_list.append((u, (pred, score)))
                output_queue.task_done()
            except queue.Empty:
                continue

        # get real predict item id
        result_list = [(int(self.user2id[u]), int(self.item2id[pred[i]]), float(score[i])) for (u, (pred, score)) in pred_list for i in range(len(pred))]
        return result_list


@jit(nopython=True)
def compute_x_jit(u, i, b_tm, VUI, VIU, VIL, VLI):
    acc_val = 0.0
    for l in b_tm:
        acc_val += np.dot(VIL[i], VLI[l])
    return np.dot(VUI[u], VIU[i]) + (acc_val / len(b_tm))


@jit(nopython=True)
def sigmoid_jit(x):
    if x >= 0:
        return math.exp(-np.logaddexp(0, -x))
    else:
        return math.exp(x - np.logaddexp(x, 0))


@jit(nopython=True)
def evaluation_jit(data_list, VUI, VIU, VIL, VLI):
    correct_3 = 0
    correct_5 = 0
    correct_10 = 0
    pred_list_3 = []
    pred_list_5 = []
    pred_list_10 = []
    VIL_m_VLI = np.dot(VIL, VLI.T)
    for d_idx in range(len(data_list[0])):
        u = data_list[0][d_idx]
        i = data_list[1][d_idx]
        b_tm = data_list[2][d_idx][data_list[2][d_idx] != -1]
        former = np.dot(VUI[u], VIU.T)
        latter = VIL_m_VLI[:, b_tm].sum(1) / len(b_tm)
        scores = former + latter
        pred_3 = scores.argsort()[-3:][::-1]
        pred_5 = scores.argsort()[-5:][::-1]
        pred_10 = scores.argsort()[-10:][::-1]

        if i in pred_3:
            correct_3 += 1
        if i in pred_5:
            correct_5 += 1
        if i in pred_10:
            correct_10 += 1

        pred_list_3.append(list(pred_3))
        pred_list_5.append(list(pred_5))
        pred_list_10.append(list(pred_10))

    try:
        acc_3 = float(correct_3) / len(pred_list_3)
        acc_5 = float(correct_5) / len(pred_list_5)
        acc_10 = float(correct_10) / len(pred_list_10)
        return acc_3, acc_5, acc_10
    except:
        return 0.0, 0.0, 0.0


@jit(nopython=True)
def learn_epoch_jit(data_list, neg_batch_size, epoch, item_set, VUI, VIU, VIL, VLI, learning_rate, regular):
    set_len = len(data_list[0])
    for iter_idx in range(set_len):
        if iter_idx % 10000 == 0:
            print(epoch, iter_idx, set_len)

        d_idx = np.random.randint(0, len(data_list[0]))
        u = data_list[0][d_idx]
        i = data_list[1][d_idx]
        b_tm = data_list[2][d_idx][data_list[2][d_idx] != -1]
        j_list = np.random.choice(item_set, size=neg_batch_size, replace=False)

        z1 = compute_x_jit(u, i, b_tm, VUI, VIU, VIL, VLI)
        for j in j_list:
            z2 = compute_x_jit(u, j, b_tm, VUI, VIU, VIL, VLI)
            delta = 1 - sigmoid_jit(z1 - z2)

            VUI_update = learning_rate * (delta * (VIU[i] - VIU[j]) - regular * VUI[u])
            VIUi_update = learning_rate * (delta * VUI[u] - regular * VIU[i])
            VIUj_update = learning_rate * (-delta * VUI[u] - regular * VIU[j])

            VUI[u] += VUI_update
            VIU[i] += VIUi_update
            VIU[j] += VIUj_update

            eta = np.zeros(VLI.shape[1])
            for l in b_tm:
                eta += VLI[l]
            eta = eta / len(b_tm)

            VILi_update = learning_rate * (delta * eta - regular * VIL[i])
            VILj_update = learning_rate * (-delta * eta - regular * VIL[j])
            VLI_updates = np.zeros((len(b_tm), VLI.shape[1]))
            for idx, l in enumerate(b_tm):
                VLI_updates[idx] = learning_rate * ((delta * (VIL[i] - VIL[j]) / len(b_tm)) - regular * VLI[l])

            VIL[i] += VILi_update
            VIL[j] += VILj_update
            for idx, l in enumerate(b_tm):
                VLI[l] += VLI_updates[idx]
    return VUI, VIU, VIL, VLI


def data_to_list(data_list):
    u_list = []
    i_list = []
    b_tm_list = []
    max_l = 0
    for d in data_list:
        u_list.append(d[0])
        i_list.append(d[1])
        b_tm_list.append(d[2])
        if len(d[2]) > max_l:
            max_l = len(d[2])
    for b_tm in b_tm_list:
        b_tm.extend([-1 for i in range(max_l - len(b_tm))])
    b_tm_list = np.array(b_tm_list)
    return u_list, i_list, b_tm_list


if __name__ == '__main__':
    user2id = {}
    outlet2id = {}
    train_set, test_set = None, None
    # define model
    fpmc = FPMC(n_user=max(set(user2id.values())) + 1, n_outlet=max(set(outlet2id.values())) + 1,
                n_factor=32, learning_rate=0.01, regular=0.001)
    fpmc.user2id = user2id
    fpmc.outlet2id = outlet2id
    fpmc.init_model()
    # training
    fpmc.train(train_set, test_set, n_epoch=20, neg_batch_size=10, model_save_dir='./fpmc_model.npz', eval_per_epoch=False)

    pred_set = None
    # convert to {value: key}, which is different from training
    fpmc.user2id = {v: k for k, v in user2id.items()}
    fpmc.outlet2id = {v: k for k, v in outlet2id.items()}
    # prediction
    pred_list = fpmc.predict(pred_set, n_thread=10, n_pred=100)
