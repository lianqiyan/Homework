import argparse
import glob
import os
import random
import h5py
import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)

class Data(object):
    def __init__(self, data_path, class_path, img_size, batch_size, chanel, classes):
        self.data_path = data_path
        self.batch_size = batch_size
        self.class_path = class_path
        self.img_size = img_size
        self.classes = classes
        # self.indexs = []
        self.channel = chanel
        self.load()
        self.pos = 0
        return

    def next_batch(self):
        X = np.zeros([self.batch_size, self.img_size, self.img_size, self.channel], np.float32)
        Y = np.zeros([self.batch_size, len(self.classes)], np.float32)

        # if self.pos > len(self.compressed_samples) - self.batch_size:
        #     return None, None
        start = self.pos
        self.pos += self.batch_size

        if self.pos > self.images.shape[0]:
            rest = self.pos - self.images.shape[0]
            sav = self.images.shape[0] - start
            X[0:sav, :, :, :] = self.images[start::, :, :, :]
            Y[0:sav, :] = self.labels[start::, :]
            X[sav::, :, :, :] = self.images[0:rest, :, :, :]
            Y[sav::, :] = self.labels[0:rest, :]
            self.pos = rest
        else:
            end = self.pos
            X[:, :, :, :] = self.images[start:end, :, :, :]
            Y[:, :] = self.labels[start:end, :]

        return X, Y

    def shuffle(self):
        per = np.random.permutation(self.labels.shape[0])
        rand_x = self.images[per, :, :, :]
        rand_y = self.labels[per, :]
        return rand_x, rand_y

    def load(self):
        status = os.path.basename(self.data_path)
        # print(self.data_path)
        point_dir = os.path.join(os.getcwd(), 'true_class.h5')
        # file_dir = os.path.join(point_dir, status+'.h5')
        if not os.path.exists(point_dir):
            all_name = os.listdir(self.class_path)

            cls = []
            b_list = []
            for i, na in enumerate(all_name):
                s_list = []
                for index, c in enumerate(self.classes):
                    if na.startswith(c) and not na.endswith('query.txt'):
                        cls.append(index)
                        # print(na, index)
                        f_dir = os.path.join(self.class_path, na)
                        with open(f_dir) as f:
                            data = f.readlines()
                # print(data)
                        for ina in data:
                            ina = ina.replace('\n', '') + '.jpg'
                            s_list.append(ina)
                if s_list:
                    b_list.append(s_list)

            class_img = []
            # print(len(cls), len(b_list))
            for i in range(11):
                temp = []
                for index, item in enumerate(b_list):
                    # print(index)
                    if cls[index] == i:
                        temp.extend(item)
                class_img.append(temp)
            # print(class_img)
            # print(len(class_img))

            print("ready to read images")
            img_num = np.sum([len(item) for item in class_img])
            # return
            self.images = np.zeros([img_num, self.img_size, self.img_size, 3])
            self.labels = np.zeros([img_num, len(self.classes)])
            count = 0
            self.img_names = []
            for f_index,file in enumerate(class_img):
                for item in file:
                    print('reading No.' + str(count))
                    file_dir = os.path.join(self.data_path, item)
                    self.img_names.extend(file_dir)
                    image = cv2.imread(file_dir)
                    image = cv2.resize(image, (self.img_size, self.img_size), 0, 0, cv2.INTER_LINEAR)
                    image = image.astype(np.float32)
                    image = np.multiply(image, 1.0 / 255.0)
                    # print(image.shape)
                    self.images[count,...] = image
                    # images.append(image)
                    self.labels[count, f_index] = 1.0
                    count = count + 1

            # file_dir = os.path.join(os.getcwd(), 'train.h5')
            # self.file_name = file_name
            with h5py.File(point_dir, 'w') as hf:
                hf.create_dataset('data', data=self.images)
                hf.create_dataset('label', data=self.labels)
                # hf.create_dataset('file_name', data=self.file_name)
        else:
            all_name = os.listdir(self.class_path)
            cls = []
            b_list = []
            for i, na in enumerate(all_name):
                s_list = []
                for index, c in enumerate(self.classes):
                    if na.startswith(c) and not na.endswith('query.txt'):
                        cls.append(index)
                        f_dir = os.path.join(self.class_path, na)
                        with open(f_dir) as f:
                            data = f.readlines()
                        for ina in data:
                            ina = ina.replace('\n', '') + '.jpg'
                            s_list.append(ina)
                if s_list:
                    b_list.append(s_list)
            class_img = []
            for i in range(11):
                temp = []
                for index, item in enumerate(b_list):
                    if cls[index] == i:
                        temp.extend(item)
                class_img.append(temp)
            self.img_names = []
            for f_index, file in enumerate(class_img):
                for item in file:
                    file_dir = os.path.join(self.data_path, item)
                    self.img_names.extend(file_dir)
            print("reading stored file")
            with h5py.File(point_dir, 'r') as hf:
                self.images = np.array(hf.get('data'))
                self.labels = np.array(hf.get('label'))
                # self.file_name = hf.get('file_name')


class CNN(object):
    def __init__(self,image_size, train_data, channel, filter_num, filter_size, hidden_layer, max_iter, learning_rate):

        self.image_size = image_size
        self.train_data = train_data
        # self.test_data = test_data
        self.c_dim = channel
        self.filter_num = filter_num
        self.filter_size = filter_size
        self.hidden_layer = hidden_layer
        self.filter_channel = [channel]
        self.filter_channel += filter_num[0:-1]
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.batch_size = train_data.batch_size
        self.checkpoint_dir = 'checkpoint'
        # print(self.filter_channel)
        self.build_model()  # initialize tf variable

    def create_convolutional_layer(self, input, weights, biases):
        ## Creating the convolutional layer
        layer = tf.nn.conv2d(input=input,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='VALID')
        layer += biases
        ## We shall be using max-pooling.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID')
        ## Output of pooling is fed to Relu which is the activation function for us.
        layer = tf.nn.relu(layer)

        return layer

    def create_flatten_layer(self, layer):
        # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
        # But let's get it from the previous layer.
        layer_shape = layer.get_shape()

        ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
        num_features = layer_shape[1:4].num_elements()
        ## Now, we Flatten the layer so we shall have to reshape to num_features
        layer = tf.reshape(layer, [-1, num_features], name='dat_vector')

        return layer, num_features

    def create_fc_layer(self, input, weights, biases, use_relu, f_name):
        # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
        print('input::::', input, weights)
        layer = tf.add(tf.matmul(input, weights), biases, name = f_name)
        if use_relu:
            layer = tf.nn.relu(layer, name=f_name)

        return layer


    def build_model(self):  # set the variables

        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.train_data.labels.shape[1]], name='labels')

        self.weights = {}; self.biases = {}
        ## for convolution layer
        conv = [self.images]
        for i, sz in enumerate(self.filter_size):
            name = ['w'+str(i), 'b'+str(i)]
            print(self.filter_size[i], self.filter_size[i], self.filter_channel[i], self.filter_num[i])
            self.weights[name[0]] = tf.Variable(tf.random_normal([self.filter_size[i], self.filter_size[i], self.filter_channel[i], self.filter_num[i]],
                                                                 stddev=1e-3), name=name[0])  # initialize weights
            self.biases[name[1]] = tf.Variable(tf.zeros([self.filter_num[i]]), name=name[1])  # initialize biases
            ## create_convolutional_layer
            layer_conv = self.create_convolutional_layer(conv[i], self.weights[name[0]], self.biases[name[1]])
            conv.append(layer_conv)

        fc_in, num_features = self.create_flatten_layer(conv[-1])
        hidden_input = [num_features]
        hidden_input += self.hidden_layer
        hidden_input.append(self.train_data.labels.shape[1])
        print(hidden_input)
        ## for fully connected layer
        fc_input = [fc_in]
        for i, num in enumerate(hidden_input):
            if num == hidden_input[-1]:
                break
            # print("hidden layer shape", hidden_input[i], hidden_input[i+1])
            name = ['fcw' + str(i), 'fcb' + str(i)]
            self.weights[name[0]] = tf.Variable(
                tf.random_normal([hidden_input[i], hidden_input[i+1]], stddev=1e-3), name=name[0])  # initialize weights
            self.biases[name[1]] = tf.Variable(tf.zeros([hidden_input[i+1]], name=name[1]))  # initialize weights

            # print(fc_input[i], self.weights[name[0]],  self.biases[name[1])
            fc_name = 'fc_layer_'+str(i+1)
            if hidden_input[i+1] == hidden_input[-1]:
                layer_fc = self.create_fc_layer(fc_input[i], self.weights[name[0]],  self.biases[name[1]], False, fc_name)
            else:
                layer_fc = self.create_fc_layer(fc_input[i], self.weights[name[0]], self.biases[name[1]], False, fc_name)
            fc_input.append(layer_fc)

        # get the prediction
        y_pred = tf.nn.softmax(fc_input[-1], name='y_pred')
        y_pred_cls = tf.argmax(y_pred, dimension=1)

        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=self.train_data.labels))
        self.cost = tf.reduce_mean(tf.square(self.labels - y_pred))
        # p = 0.012
        # cost = (tf.reduce_mean(cross_entropy) + p * tf.nn.l2_loss(conv1_weights) + p * tf.nn.l2_loss(conv1_biase) +
        #         p * tf.nn.l2_loss(conv2_weights) + p * tf.nn.l2_loss(conv2_biase) +
        #         p * tf.nn.l2_loss(conv3_weights) + p * tf.nn.l2_loss(conv3_biase) +
        #         p * tf.nn.l2_loss(fc1_weights) + p * tf.nn.l2_loss(fc1_biase) +
        #         p * tf.nn.l2_loss(fc2_weights) + p * tf.nn.l2_loss(fc2_biase))
        self.saver = tf.train.Saver()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
        correct_prediction = tf.equal(y_pred_cls, tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("Oxford/", sess.graph)


        # self.saver = tf.train.Saver()

    def train(self):
        # self.build_net()
        with tf.Session() as sess:
            self.train_data.images, self.train_data.labels = self.train_data.shuffle()
            sess.run(tf.global_variables_initializer())
            if self.load(sess, self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            for i in range(self.max_iter):
                batch_x, batch_y = self.train_data.next_batch()
                sess.run(self.optimizer, feed_dict={self.images: batch_x, self.labels: batch_y})

                if np.mod(i, 50) == 0:
                # test
                    self.save(sess, self.checkpoint_dir, i)
                    # test_x, test_y = self.test_data.next_batch()
                    t_ac = sess.run(self.accuracy, feed_dict={self.images: batch_x, self.labels: batch_y})
                    t_cost = sess.run(self.cost, feed_dict={self.images: batch_x, self.labels: batch_y})

                    print('\nEpoch: ', i, 'cost is ' , t_cost, '\n', 'accrucy is:', t_ac)
        return





    def save(self, sess, checkpoint_dir, step):
        model_name = "CNN.model"
        model_dir = 'cnn'
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, sess, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = 'cnn'
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--test_size', type=int, default=128)
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--class_path', type=str, default=r'G:\Oxford\groundtruth')
    parser.add_argument('--data_path', type=str, default= r'G:\Oxford\images')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--classes', type=list, default=['all_souls', 'ashmolean', 'balliol', 'bodleian', 'christ_church', 'cornmarket',
        	'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera'])
    # parser.add_argument('--test_stride', type=int, default=32)

    config = parser.parse_args()

    print(config)

    train_data = Data(config.data_path,config.class_path, config.img_size, config.batch_size, config.channel, config.classes)

    # print(train_data.file_name)

    # temp_x, temp_y = train_data.next_batch()
    # print(temp_x.shape, temp_y.shape)

    # cnn = CNN(config.img_size, train_data, config.channel, [32, 16, 8], [3, 3, 3], [256], 10000, 0.0001)
    # cnn.train()
