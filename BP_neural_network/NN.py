import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import classification_report



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


def sk_MLP(tr_data, tr_label, te_data, te_label):  # 基于sklearn库的神经网络实现， 用于对比实验结果
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (4, ), random_state = 1)
    clf.fit(tr_data, tr_label)

    pred = clf.predict(te_data)
    target_names = ['type1', 'type2', 'type3']
    print(classification_report(te_label, pred, target_names = target_names))


### 以下是关于BP network的函数
def bp_network(tr_d, tr_la, layers_size, learning_rate, max_iter):
    w, b = parameter_initialize(tr_d.shape[1], layers_size, tr_label.shape[1])
    w, b, cost = train(tr_d, tr_la, w, b, learning_rate, max_iter)

    return w, b, cost


def parameter_initialize(num_input, layers_size, num_output):  # 初始化weights biase
    layers_size.append(num_output)

    biase = []
    for i in range(0, len(layers_size)):
        b = np.random.randn(layers_size[i])
        biase.append(b)

    layers_size.insert(0, num_input)

    weights = []
    for i in range(0, len(layers_size)-1):  # weight:
        w = np.random.randn(layers_size[i], layers_size[i + 1])
        weights.append(w)

    return weights, biase


def train(tr_d, tr_la, weights, biase, lr, max_iter):
    J = []
    for k in range(0, max_iter):

        # 向前传播
        m = tr_d.shape[0]
        y_h = []  # 每一层的输出
        y_h.append(tr_d)
        x_in = tr_d

        for i in range(0, len(biase)):  # 隐藏层的向前传导
            x_in = sigmoid(np.matmul(x_in, weights[i]) - biase[i])
            y_h.append(x_in)

        w_grad = []; b_grad = []  # 初始化weghts, biase的梯度
        for i in range(0, len(weights)):
            w_grad.append(np.zeros(weights[i].shape))
            b_grad.append(np.zeros(biase[i].shape))


        num = len(biase)
        for i in range(0, m):
            gi_temp = num*[None]    # 误差反向传播
            for j in range(num -1, 0, -1):
                if j == num - 1:
                    d = tr_la[i, :] - y_h[j + 1][i, :]
                else:
                    d = np.sum(gi_temp[j + 1] * weights[j + 1], 1)
                gi_temp[j] = y_h[j + 1][i, :] * (1 - y_h[j + 1][i, :]) * d
                w_grad[j] = w_grad[j] + np.multiply(np.tile(gi_temp[j], [y_h[j][i, :].shape[0], 1]) ,
                                            np.expand_dims(y_h[j ][i, :], axis=1)) * lr / m
                b_grad[j] = b_grad[j] + (-gi_temp[j] * lr) / m

        for i in range(0, num):    # 更新权重
            weights[i] = weights[i] + w_grad[i]
            biase[i] = biase[i] + b_grad[i]

        cost = np.sum(np.power(tr_la - y_h[-1], 2)) / (2 * m)
        J.append(cost)

    return weights, biase, J


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def predict(x, weights, biase):
    x_in = x
    for i in range(0, len(biase)):  # 隐藏层的向前传导
        x_in = sigmoid(np.matmul(x_in, weights[i]) - biase[i])

    return np.argmax(x_in, 1)


if __name__ == '__main__':
    tr_data, tr_label, te_data, te_label = load_data('iris.csv', 0.8)
    # sk_MLP(tr_data, tr_label, te_data, te_label)

    tr_label, te_label = one_hot_encode(tr_label, te_label)
    w, b, J = bp_network(tr_data, tr_label, [6], 5, 5000)

    pred = predict(te_data, w, b)

    # print(pred)

    target_names = ['type1', 'type2', 'type3']
    print(classification_report(np.argmax(te_label, 1), pred, target_names = target_names))

    plt.plot(J)
    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()
