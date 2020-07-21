import math
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

        return convert_dict(train_set[:10000]), convert_dict(test_set)


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


class UserCF:
    def __init__(self, train_dict, test_dict, n_rec):
        self.cf_matrix = {}
        self.train_dict = train_dict  # train set
        self.test_dict = test_dict  # test set
        self.n_rec = n_rec  # TopN
        self.calc_similarity()

    def calc_similarity(self):
        # reverse order: key = item, value = list of users who have clicked
        item_user = {}
        for user, items in self.train_dict.items():
            for item in items:
                item_user.setdefault(item, set())
                item_user[item].add(user)

        # build user co-rated item matrix
        user_sim_matrix = {}
        for item, users in item_user.items():
            for u in users:
                # if u not exist in matrix, set default value
                user_sim_matrix.setdefault(u, {})
                for v in users:
                    if u == v: continue
                    user_sim_matrix[u].setdefault(v, 0)
                    user_sim_matrix[u][v] += 1
                    # regularization for hot items between u and v
                    user_sim_matrix[u][v] /= math.log(1 + len(users))

        # calculat similarity - cosine similarity
        for u, related_users in user_sim_matrix.items():
            for v in related_users:
                user_sim_matrix[u][v] /= math.sqrt(len(self.train_dict[u]) * len(self.train_dict[v]))

        # normalize score
        for u, related_users in user_sim_matrix.items():
            scores = []
            for v, score in related_users.items():
                scores.append(score)

            if len(scores) == 0: continue
            s_max, s_min = max(scores), min(scores)
            if s_max - s_min > 0:
                for v in related_users:
                    user_sim_matrix[u][v] = (user_sim_matrix[u][v] - s_min) / (s_max - s_min)
            else:
                for v in related_users:
                    user_sim_matrix[u][v] = 1

        # sort by similarity score
        self.cf_matrix = {k: list(sorted(v.items(), key=lambda x: x[1], reverse=True)) for k, v in user_sim_matrix.items()}

    def recommend(self, user):
        item_dict = {}
        # new user
        if user not in self.cf_matrix.keys():
            return None
        # items have been seen by user
        seen_items = self.train_dict[user]

        for v, wuv in self.cf_matrix[user]:
            for item in self.train_dict[v]:
                # filter out items have been seen
                if item not in seen_items:
                    item_dict.setdefault(item, 0)
                    # sum the item similarity score of similar users
                    item_dict[item] += wuv
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


class ItemCF:
    def __init__(self, train_dict, test_dict, n_rec):
        self.cf_matrix = {}
        self.train_dict = train_dict  # train set
        self.test_dict = test_dict  # test set
        self.n_rec = n_rec  # TopN
        self.calc_similarity()

    def calc_similarity(self):
        # key = item, value = clicked times
        item_times = {}
        for user, items in self.train_dict.items():
            for item in items:
                item_times.setdefault(item, 0)
                item_times[item] += 1

        # build item co-rated user matrix
        item_sim_matrix = {}
        for user, items in self.train_dict.items():
            for u in items:
                # if u not exist in matrix, set default value
                item_sim_matrix.setdefault(u, {})
                for v in items:
                    if u == v: continue
                    item_sim_matrix[u].setdefault(v, 0)
                    item_sim_matrix[u][v] += 1
                    # regularization for hot items between u and v
                    item_sim_matrix[u][v] /= math.log(1 + len(items))

        # calculat similarity - cosine similarity
        for u, related_items in item_sim_matrix.items():
            for v in related_items:
                item_sim_matrix[u][v] /= math.sqrt(item_times[u] * item_times[v])

        # normalize score
        for u, related_items in item_sim_matrix.items():
            scores = []
            for v, score in related_items.items():
                scores.append(score)

            if len(scores) == 0:
                continue
            s_max, s_min = max(scores), min(scores)
            if s_max - s_min > 0:
                for v in related_items:
                    item_sim_matrix[u][v] = (item_sim_matrix[u][v] - s_min) / (s_max - s_min)
            else:
                for v in related_items:
                    item_sim_matrix[u][v] = 1

        # sort by similarity score
        self.cf_matrix = {k: list(sorted(v.items(), key=lambda x: x[1], reverse=True)) for k, v in item_sim_matrix.items()}

    def recommend(self, user):
        item_dict = {}
        if user not in self.train_dict.keys():
            return None

        # items have been seen by user
        seen_items = set(self.train_dict[user])
        for item in seen_items:
            # new item
            if item not in self.cf_matrix.keys(): continue
            for v, wuv in self.cf_matrix[item]:
                # filter out items have been seen
                if v not in seen_items:
                    item_dict.setdefault(item, 0)
                    # sum the item similarity score of similar items
                    item_dict[item] += wuv
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

    # User CF
    usercf = UserCF(train_dict, test_dict, n_rec=100)
    pred_dict = usercf.predict()
    metrics = Metrics(train_dict, test_dict, pred_dict)
    metrics.evaluate()

    # Item CF
    itemcf = ItemCF(train_dict, test_dict, n_rec=100)
    pred_dict = itemcf.predict()
    metrics = Metrics(train_dict, test_dict, pred_dict)
    metrics.evaluate()
