import math
import numpy as np
from os import path
from scipy.sparse import csc_matrix, linalg, eye


class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_dataset(self):
        train_set = []
        with open(path.join(self.file_path, 'train_set.txt'), 'r') as f:
            lines = f.readlines()
            for l in lines:
                # read user_id, item_id, rate
                train_set.append(tuple(map(int, l.strip().split(',')[:2])))

        test_set = []
        with open(path.join(self.file_path, 'test_set.txt'), 'r') as f:
            lines = f.readlines()
            for l in lines:
                # read user_id, item_id, rate
                test_set.append(tuple(map(int, l.strip().split(',')[:2])))

        def convert_dict(data_set):
            data_dict = {}
            for user, item in data_set:
                data_dict.setdefault(user, set())
                data_dict[user].add(item)
            return data_dict

        return convert_dict(train_set), convert_dict(test_set)


class Metrics:
    def __init__(self, train_dict, test_dict, pred_dict):
        self.train_dict = train_dict
        self.test_dict = test_dict
        self.pred_dict = pred_dict

    def precision(self):
        total, hit = 0, 0
        for user, items in self.test_dict.items():
            if user not in self.pred_dict.keys(): continue
            pred_items = self.pred_dict[user]
            for item, score in pred_items:
                if item in items:
                    hit += 1
            total += len(pred_items)
        return round(hit / total * 100, 4)

    def recall(self):
        total, hit = 0, 0
        for user, items in self.test_dict.items():
            if user not in self.pred_dict.keys(): continue
            pred_items = self.pred_dict[user]
            for item, score in pred_items:
                if item in items:
                    hit += 1
            total += len(items)
        return round(hit / total * 100, 4)

    def coverage(self):
        all_items, rec_items = set(), set()
        for user, items in self.test_dict.items():
            for item in items:
                all_items.add(item)
            if user not in self.pred_dict.keys(): continue
            pred_items = self.pred_dict[user]
            for item, score in pred_items:
                rec_items.add(item)
        return round(len(rec_items) / len(all_items) * 100, 4)

    def popularity(self):
        items_dict = {}
        for user, items in self.train_dict.items():
            for item in items:
                items_dict.setdefault(item, 0)
                items_dict[item] += 1

        items_score, items_num = 0, 0
        for user, items in self.test_dict.items():
            if user not in self.pred_dict.keys(): continue
            pred_items = self.pred_dict[user]
            for item, score in pred_items:
                items_score += math.log(1 + items_dict[item])
                items_num += 1
        return round(items_score / items_num, 4)

    def evaluate(self):
        metrics = {
            'Precision': self.precision(),
            'Recall': self.recall(),
            'Converage': self.coverage(),
            'Popularity': self.popularity()
        }
        return metrics


class BPR:
    def __init__(self, train_dict, test_dict, alpha, n_rec):
        self.graph = None
        self.id2item = []
        self.users_dict = {}
        self.items_dict = {}
        self.train_dict = train_dict  # train set
        self.test_dict = test_dict  # test set
        self.alpha = alpha
        self.n_rec = n_rec  # TopN

        self.calc_graph()

    def calc_graph(self):
        # generate user and item dict
        all_items = []
        for user, items in self.train_dict.items():
            all_items.extend(items)
        self.id2item = list(set(all_items))
        self.users_dict = {u: i for i, u in enumerate(self.train_dict.keys())}
        self.items_dict = {u: i + len(self.users_dict) for i, u in enumerate(self.id2item)}

        # reverse order: key = item, value = list of users who have clicked
        item_user = {}
        for user, items in self.train_dict.items():
            for item in items:
                item_user.setdefault(item, set())
                item_user[item].add(user)

        data, row, col = [], [], []
        # for each item of each user
        for user, items in self.train_dict.items():
            for item in items:
                data.append(1 / len(items))
                row.append(self.users_dict[user])
                col.append(self.items_dict[item])
        # for each user of each item
        for item, users in item_user.items():
            for user in users:
                data.append(1 / len(users))
                row.append(self.items_dict[item])
                col.append(self.users_dict[user])
        # generate graph
        self.graph = csc_matrix((data, (row, col)), shape=(len(data), len(data)))

    def recommend(self, user):
        item_dict = {}
        # items have been seen by user
        seen_items = self.train_dict[user]

        r0 = [0] * self.graph.shape[0]
        r0[self.users_dict[user]] = 1
        r0 = csc_matrix(r0)
        r = r0 * (1 - self.alpha) * linalg.inv(eye(self.graph.shape[0]) - self.alpha * self.graph.T)
        r = r.T.toarray()[0][len(self.users_dict):]
        recs = [(self.id2item[i], r[i]) for i in np.argsort(-r)[0:self.n_rec]]
        return recs

    def predict(self):
        pred_dict = {}
        for i, user, in enumerate(self.test_dict):
            rec_items = self.recommend(user)
            if rec_items is not None:
                for item, score in rec_items:
                    pred_dict.setdefault(user, list())
                    pred_dict[user].append((item, score))
        return pred_dict


if __name__ == '__main__':
    file_path = ''
    dataset = Dataset(file_path)
    train_dict, test_dict = dataset.load_dataset()

    # BPR
    pr = BPR(train_dict, test_dict, n_rec=100)
    pred_dict = pr.predict()
    metrics = Metrics(train_dict, test_dict, pred_dict)
    metrics.evaluate()
