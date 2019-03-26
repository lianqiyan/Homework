import numpy as np
import cv2
import tensorflow  as tf
import matplotlib.pyplot as plt
import os
import h5py
from tkinter.filedialog import askopenfilename


def gen_vec(meta_path, data):
    path = os.path.join(os.getcwd(), meta_path)

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(path)
    new_saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(meta_path)))

    graph = tf.get_default_graph()
    fc_vec = graph.get_tensor_by_name("fc_layer_1:0")
    images = graph.get_tensor_by_name("images:0")
    # t_name = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # print(t_name)

    all_vec = np.zeros([data.images.shape[0], 1000])
    for i in range(0, data.images.shape[0], 500):
        temp_data = data.images[i:i+500, ...]
        temp_vec = sess.run(fc_vec, feed_dict={images: temp_data})
        all_vec[i:i+500,...]=temp_vec

    # all_vec = sess.run(fc_vec, feed_dict={images: data.images})
    print(all_vec.shape)
    point_dir = os.path.join(os.getcwd(), 'data_vec.h5')
    with h5py.File(point_dir, 'w') as hf:
        hf.create_dataset('data_vec', data=all_vec)


def image2vec(meta_path):
    file_dir = askopenfilename()
    # print(filename)
    image = cv2.imread(file_dir)
    image = cv2.resize(image, (256, 256), 0, 0, cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    # image.reshape((1, 256, 256, 3))
    image = np.expand_dims(image, axis=0)
    # print(image.shape)

    path = os.path.join(os.getcwd(), meta_path)
    # print(path)
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(path)
    new_saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(meta_path)))

    graph = tf.get_default_graph()
    fc_vec = graph.get_tensor_by_name("fc_layer_1:0")
    images = graph.get_tensor_by_name("images:0")

    vec = sess.run(fc_vec, feed_dict={images: image})
    vec = np.squeeze(vec)
    # print(vec.shape)
    return vec, file_dir


def search_images(vec_path, meta_path, img_path, class_path, classes):
    vec, in_dir = image2vec(meta_path)
    with h5py.File(vec_path, 'r') as hf:
        all_vec = np.array(hf.get('data_vec'))
    dis = np.zeros(all_vec.shape[0])
    for i in range(0, dis.shape[0]):
        dis[i] = cosine_dis(all_vec[i, ...], vec)
    order = np.argsort(dis)
    top_3 = order[-3:]
    all_dir = get_all_filename(img_path, class_path, classes)
    # print(top_3)
    # print(top_3, len(all_dir))
    get_dir = [all_dir[i] for i in top_3]
    # print(get_dir)
    img_list = []
    for img_name in get_dir:
        full_path = os.path.join(img_path, img_name)
        img = cv2.imread(full_path)
        img_list.append(img)
    img_input = cv2.imread(in_dir)

    plt.subplot(141)
    plt.title("input_image")
    plt.imshow(img_input)
    plt.subplot(142)
    plt.title("image_1")
    plt.imshow(img_list[2])
    plt.subplot(143)
    plt.title("image_2")
    plt.imshow(img_list[1])
    plt.subplot(144)
    plt.title("image_3")
    plt.imshow(img_list[0])
    plt.show()


def get_all_filename(data_path,  class_path, classes):
    all_name = os.listdir(class_path)
    cls = []
    b_list = []
    for i, na in enumerate(all_name):
        s_list = []
        for index, c in enumerate(classes):
            if na.startswith(c) and not na.endswith('query.txt'):
                cls.append(index)
                f_dir = os.path.join(class_path, na)
                with open(f_dir) as f:
                    data = f.readlines()
                for ina in data:
                    ina = ina.replace('\n', '') + '.jpg'
                    s_list.append(ina)
        if s_list:
            b_list.append(s_list)
    # print(len(b_list))
    class_img = []
    for i in range(11):
        temp = []
        for index, item in enumerate(b_list):
            if cls[index] == i:
                temp.extend(item)
        class_img.append(temp)
    img_names = []
    for f_index, file in enumerate(class_img):
        for item in file:
            file_dir = os.path.join(data_path, item)
            img_names.append(file_dir)
    return img_names


def cosine_dis(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_img_path(vec_path, meta_path, img_path, class_path, classes):
    vec, in_dir = image2vec(meta_path)
    with h5py.File(vec_path, 'r') as hf:
        all_vec = np.array(hf.get('data_vec'))
    dis = np.zeros(all_vec.shape[0])
    for i in range(0, dis.shape[0]):
        dis[i] = cosine_dis(all_vec[i, ...], vec)
    order = np.argsort(dis)
    top_3 = order[-3:]
    all_dir = get_all_filename(img_path, class_path, classes)
    get_dir = [all_dir[i] for i in top_3]
    get_dir.append(in_dir)
    return get_dir



if __name__ == '__main__':
    classes = ['all_souls', 'ashmolean', 'balliol', 'bodleian', 'christ_church', 'cornmarket',
        	'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera']
    saved_path = r'F:\Paper reproduce\image_search\zip2\CNN.model-9950.meta'
    # gen_vec(saved_path, 1)
    vec_path = os.path.join(os.getcwd(), 'vec_256.h5')
    # print(cosine_dis(np.array([0,1]), np.array([1,0])))
    search_images(vec_path, saved_path, r'G:\Oxford\images', r'G:\Oxford\groundtruth', classes)
    # a = get_all_filename(r'G:\Oxford\images', r'G:\Oxford\groundtruth', classes)
    # print(len(a))

