import math
import numpy as np
from os import path


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


class LFM:
    def __init__(self, train_dict, test_dict, vec_dim, learning_rate, alpha, steps, ratio, n_rec):
        self.user_vec = {}
        self.item_vec = {}
        self.train_dict = train_dict
        self.test_dict = test_dict
        self.vec_dim = vec_dim  # dimension of latent factor
        self.learning_rate = learning_rate  # learning rate
        self.alpha = alpha  # regularization
        self.steps = steps  # train steps
        self.ratio = ratio  # negative sample ratio
        self.n_rec = n_rec  # TopN
        self.train()

    def sample(self, all_items, all_times):
        step_train_dict = {}
        # postive sample
        for user, items in self.train_dict.items():
            step_train_dict.setdefault(user, {})
            for item in items:
                step_train_dict[user].setdefault(item, 0)
                step_train_dict[user][item] += 1
        # negative sample
        for user, items in step_train_dict.items():
            seen_items = set(items)
            sample_items = np.random.choice(all_items, int(len(seen_items) * self.ratio * 2), all_times)
            select_items = [x for x in sample_items if x not in seen_items][:int(len(seen_items) * self.ratio)]
            for item in select_items:
                step_train_dict[user].setdefault(item, 0)
        return step_train_dict

    def train(self):
        # key = item, value = clicked times
        item_times = {}
        for user, items in self.train_dict.items():
            for item in items:
                item_times.setdefault(item, 0)
                item_times[item] += 1
        all_items = [x[0] for x in item_times.items()]
        all_times = [x[1] for x in item_times.items()]
        print(all_items[:100], all_times[:100])

        for idx in range(self.steps):
            # sample train set
            step_train_dict = self.sample(all_items, all_times)
            print(step_train_dict)
            # train
            for u, related_items in step_train_dict.items():
                # initial if user not exist in vector
                self.user_vec.setdefault(u, np.random.randn(self.vec_dim))
                for v, score in related_items.items():
                    # initial if item not exist in vector
                    self.item_vec.setdefault(v, np.random.randn(self.vec_dim))
                    # calculate error
                    error = score - (self.user_vec[u] * self.item_vec[v]).sum()
                    self.user_vec[u] += self.learning_rate * (error * self.item_vec[v] - self.alpha * self.user_vec[u])
                    self.item_vec[v] += self.learning_rate * (error * self.user_vec[u] - self.alpha * self.item_vec[v])
            self.learning_rate *= 0.9

    def recommend(self, user):
        item_dict = {}
        if user not in self.train_dict.keys():
            return None

        # items have been seen by user
        seen_items = set(self.train_dict[user])
        for item in self.item_vec.keys():
            if item not in seen_items:
                item_dict.setdefault(item, 0)
                # sum the score of this item
                item_dict[item] = (self.user_vec[user] * self.item_vec[item]).sum()
        return sorted(item_dict.items(), key=lambda x: x[1], reverse=True)[0:self.n_rec]

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

    # LFM
    lfm = LFM(train_dict, test_dict, n_rec=100)
    pred_dict = lfm.predict()
    metrics = Metrics(train_dict, test_dict, pred_dict)
    metrics.evaluate()
