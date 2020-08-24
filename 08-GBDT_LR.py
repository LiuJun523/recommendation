import pandas as pd
from os import path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_dataset(self):
        train_df = pd.read_csv(path.join(self.file_path, 'train.csv'))
        test_df = pd.read_csv(path.join(self.file_path, 'test.csv'))

        # delete Id
        train_df.drop(['Id'], axis=1, inplace=True)
        test_df.drop(['Id'], axis=1, inplace=True)
        test_df['Label'] = -1

        data_df = pd.concat([train_df, test_df])
        data_df = data_df.fillna(-1)
        return data_df


class gbdt_lr:
    def __init__(self, data_df, continuous_features, category_features, params):
        self.data_df = data_df
        self.continuous_features = continuous_features
        self.category_features = category_features

        # gbdt
        self.gbdt = GradientBoostingClassifier(n_estimators=params['n_estimators'],
                                               max_depth=params['max_depth'],
                                               min_samples_leaf=params['min_samples_leaf'],
                                               max_leaf_nodes=params['max_leaf_nodes'],
                                               learning_rate=params['learning_rate'],
                                               random_state=params['random_state'])
        # lr
        self.lr = LogisticRegression(C=params['C'], max_iter=params['max_iter'])

    def train(self):
        # one-hot
        for col in self.category_features:
            onehot_features = pd.get_dummies(self.data_df[col], prefix=col)
            self.data_df.drop([col], axis=1, inplace=True)
            self.data_df = pd.concat([self.data_df, onehot_features], axis=1)

        train_df = self.data_df[self.data_df['Label'] != -1]
        target_df = train_df.pop('Label')
        test_df = self.data_df[self.data_df['Label'] == 1]
        test_df.drop(['Label'], axis=1, inplace=True)
        train_df = train_df.fillna(-1)
        test_df = test_df.fillna(-1)

        # gbdt train
        train_x, val_x, train_y, val_y = train_test_split(train_df, target_df, test_size=0.3, random_state=1234)
        self.gbdt.fit(train_x, train_y)
        # convert to leaf
        gbdt_train_features = self.gbdt.apply(train_df)[:, :, 0]
        gbdt_test_features = self.gbdt.apply(test_df)[:, :, 0]
        gbdt_features_name = ['gbdt_leaf_' + str(i) for i in range(gbdt_train_features.shape[1])]
        gbdt_train_df = pd.DataFrame(gbdt_train_features, columns=gbdt_features_name)
        gbdt_test_df = pd.DataFrame(gbdt_test_features, columns=gbdt_features_name)

        train_df = pd.concat([train_df, gbdt_train_df], axis=1)
        test_df = pd.concat([test_df, gbdt_test_df], axis=1)
        data_df = pd.concat([train_df, test_df])
        train_len = train_df.shape[0]

        # one-hot
        for col in gbdt_features_name:
            onehot_features = pd.get_dummies(data_df[col], prefix=col)
            data_df.drop([col], axis=1, inplace=True)
            data_df = pd.concat([data_df, onehot_features], axis=1)

        train_df = data_df[:train_len]
        test_df = data_df[train_len:]
        train_df = train_df.fillna(-1)
        test_df = test_df.fillna(-1)
        train_x, val_x, train_y, val_y = train_test_split(train_df, target_df, test_size=0.3, random_state=1234)

        # lr
        self.lr.fit(train_x, train_y)
        train_logloss = log_loss(train_y, self.lr.predict_proba(train_x)[:, 1])
        val_logloss = log_loss(val_y, self.lr.predict_proba(val_x)[:, 1])
        print('train_logloss: %s, val_logloss: %s' % (train_logloss, val_logloss))

        # predict
        pred_y = self.lr.predict_proba(test_df)[:, 1]
        print(pred_y)


if __name__ == '__main__':
    file_path = ''
    dataset = Dataset(file_path)
    data_df = dataset.load_dataset()

    continuous_features = ['I' + str(i + 1) for i in range(13)]
    category_features = ['C' + str(i + 1) for i in range(26)]

    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_leaf': 30,
        'max_leaf_nodes': 5,
        'learning_rate': 0.1,
        'max_iter': 2770,
        'C': 0.8,
        'random_state': 1234
    }

    model = gbdt_lr(data_df, continuous_features, category_features, params)
    model.train()
