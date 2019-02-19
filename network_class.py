import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

class bp_network(object):
    def __init__(self, num_features, hidden_layer_size, num_class):
        self.num_features = num_features
        self.hidden_layer_size = hidden_layer_size
        self.num_class = num_class
        self.para_initialize()


    def para_initialize(self):
        layer_unit_num = self.hidden_layer_size.copy()
        layer_unit_num.append(self.num_class)

        self.bias = []
        for i in range(0, len(layer_unit_num)):
            b = np.random.randn(layer_unit_num[i])
            self.bias.append(b)

        layer_unit_num.insert(0, self.num_features)

        self.weights = []
        for i in range(0, len(layer_unit_num) - 1):  # weight:
            w = np.random.randn(layer_unit_num[i], layer_unit_num[i + 1])
            self.weights.append(w)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def fit(self, tr_d, tr_la, lr, max_iter):
        self.J = []
        for k in range(0, max_iter):

            # 向前传播
            m = tr_d.shape[0]
            y_h = []  # 每一层的输出
            y_h.append(tr_d)
            x_in = tr_d

            for i in range(0, len(self.bias)):  # 隐藏层的向前传导
                x_in = self.sigmoid(np.matmul(x_in, self.weights[i]) - self.bias[i])
                y_h.append(x_in)

            w_grad = [];
            b_grad = []  # 初始化weghts, biase的梯度
            for i in range(0, len(self.weights)):
                w_grad.append(np.zeros(self.weights[i].shape))
                b_grad.append(np.zeros(self.bias[i].shape))

            num = len(self.bias)
            for i in range(0, m):
                gi_temp = num * [None]  # 误差反向传播
                for j in range(num - 1, 0, -1):
                    if j == num - 1:
                        d = tr_la[i, :] - y_h[j + 1][i, :]
                    else:
                        d = np.sum(gi_temp[j + 1] * self.weights[j + 1], 1)
                    gi_temp[j] = y_h[j + 1][i, :] * (1 - y_h[j + 1][i, :]) * d
                    w_grad[j] = w_grad[j] + np.multiply(np.tile(gi_temp[j], [y_h[j][i, :].shape[0], 1]),
                                                        np.expand_dims(y_h[j][i, :], axis=1)) * lr / m
                    b_grad[j] = b_grad[j] + (-gi_temp[j] * lr) / m

            for i in range(0, num):  # 更新权重
                self.weights[i] = self.weights[i] + w_grad[i]
                self.bias[i] = self.bias[i] + b_grad[i]

            cost = np.sum(np.power(tr_la - y_h[-1], 2)) / (2 * m)
            self.J.append(cost)

    def plot_learning_curve(self):
        plt.plot(self.J)
        plt.title("Learning curve")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.show()


    def predict(self, test_data, test_label):
        x_in = test_data
        for i in range(0, len(self.bias)):  # 隐藏层的向前传导
            x_in = self.sigmoid(np.matmul(x_in, self.weights[i]) - self.bias[i])

        pred =  np.argmax(x_in, 1)
        print("The predict class is", pred)
        target_names = ['class0', 'class1', 'class2']
        print(classification_report(np.argmax(test_label, 1), pred, target_names=target_names))


def normalize(data):  # 数据标准化 x = (x - min)/(max - min)
    for i in range(0, data.shape[1]):
        max = np.max(data[:, i])
        min = np.min(data[:, i])
        data[:, i] = (max - data[:, i])/ (max - min)
    return data


def one_hot_encode(tr_label, te_label):  # for bp network

    num = np.unique(tr_label).shape[0]
    train_label = np.zeros([tr_label.shape[0], num])
    test_label = np.zeros([te_label.shape[0], num])
    for i in range(0, tr_label.shape[0]):
        train_label[i, int(tr_label[i])] = 1.0
    for i in range(0, te_label.shape[0]):
        test_label[i, int(te_label[i])] = 1.0

    return train_label, test_label


def load_data(path, percent):  # for MLP
    data = pd.read_csv(path, dtype=float)
    feature = data.iloc[:, 2::].copy().values  # 提取特征
    label = data[['type']].values   # 提取标签

    feature = normalize(feature)  # 标准化

    feature, label = shuffle(feature, label) # 打乱顺序，保证随机性
    train_size = int(feature.shape[0] * percent)

    train_data = feature[0:train_size, :]  # 提取对应比例的train data
    test_data = feature[train_size::, :]

    test_index = label[train_size::]
    train_index = label[0:train_size]

    return train_data, train_index.ravel(), test_data, test_index.ravel()


if __name__ == '__main__':
    tr_data, tr_label, te_data, te_label = load_data('iris.csv', 0.8)
    tr_label, te_label = one_hot_encode(tr_label, te_label)

    bp = bp_network(4, [6, 4], 3)
    bp.fit(tr_data, tr_label, 2, 3000)
    bp.predict(te_data, te_label)
    bp.plot_learning_curve()
