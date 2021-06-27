import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os


def spilt_data(nPerson, nPicture, data, label):
    """
    分割数据集

    :param nPerson : 志愿者数量
    :param nPicture: 各志愿者选入训练集的照片数量
    :param data : 等待分割的数据集
    :param label: 对应数据集的标签
    :return: 训练集, 训练集标签, 测试集, 测试集标签
    """

    # 数据集大小和意义
    allPerson, allPicture, rows, cols = data.shape

    # 划分训练集和测试集
    train = data[:nPerson, :nPicture, :, :].reshape(nPerson * nPicture, rows * cols)
    train_label = label[:nPerson, :nPicture].reshape(nPerson * nPicture)
    test = data[:nPerson, nPicture:, :, :].reshape(nPerson * (allPicture - nPicture), rows * cols)
    test_label = label[:nPerson, nPicture:].reshape(nPerson * (allPicture - nPicture))

    # 返回: 训练集, 训练集标签, 测试集, 测试集标签
    return train, train_label, test, test_label


def show_img(img, h=112, w=92):
    """
    展示单张图片

    :param img: numpy array 格式的图片
    :return:
    """
    # 展示图片
    plt.imshow(img.reshape(h, w), 'gray')
    plt.axis('off')
    plt.show()


def plot_gallery(images, titles, n_row=3, n_col=5, h=112, w=92):  # 3行4列
    """
    展示多张图片

    :param images: numpy array 格式的图片
    :param titles: 图片标题
    :param h: 图像reshape的高
    :param w: 图像reshape的宽
    :param n_row: 展示行数
    :param n_col: 展示列数
    :return:
    """
    # 展示图片
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()


def letterbox_image(image, size):
    """
    调整图片尺寸
    :param image: 用于训练的图片
    :param size: 需要调整到网络输入的图片尺寸
    :return: 返回经过调整的图片
    """
    new_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return new_image


def read_one_img(path):
    """
    根据路径读取一张人脸图片

    :param path: 图片的路径
    :return:
    """
    # 图片路径
    # 以灰度模式读取图片
    img_sample = Image.open(path).convert('L')

    # 把图片格式转为 numpy array 格式
    img_sample = np.array(img_sample, 'uint8')

    return img_sample


def get_images(path):
    """
    读取输入的文件夹路径下的所有照片，读取输入的文件夹路径下的所有照片，将其转为 1 维，
    统一保存到一个矩阵中，然依据图片名提取标签，最终该函数将输出这个照片矩阵及其中每
    张照片的标签。

    照片的命名格式请参照"person41_01.png", 其含义为第41位志愿者的第01张人脸图像。

    :param path: 照片存放的文件夹路径
    :return: numpy matrix 格式的处理好的图片，及 list 格式的各个图片的标签
    """
    # 首先获取所有人脸图片的路径
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if
                   f.endswith('png')]

    # 所有的人脸区域图片都将保存在 images 这个矩阵中
    images = np.mat(np.zeros((len(image_paths), 112 * 92)))

    trainset_labels = []

    # 对于每一张图片
    for index, image_path in enumerate(image_paths):
        # 读取图片并将其转为灰度图
        image_pil = Image.open(image_path).convert('L')

        # 把图片转为 numpy array 格式
        image = np.array(image_pil, 'uint8')
        image = letterbox_image(image=image, size=(112, 92))

        # 把 2 维的平面图像转为 1 维
        img_1D = image.flatten()

        # 把处理后的图片保存到 images 中
        images[index, :] = img_1D

        # 提取图片名作为图片的标签
        trainset_labels.append(int(image_path.split('.')[-2][-2:]))

    # 得到最终处理好的人脸图片和各个图片的标签
    trainset_labels = np.array(trainset_labels)
    return images, trainset_labels


if __name__ == '__main__':
    datapath = './ORL.npz'
    ORL = np.load(datapath)
    data = ORL['data']
    label = ORL['label']

    print("数据格式(志愿者数, 各志愿者人脸数, height, width):", data.shape)
    print("标签格式(志愿者数, 各志愿者人脸数):", label.shape)

    train_vectors, train_label, test_vectors, test_label = spilt_data(40, 5, data, label)
    print("训练数据集:", train_vectors.shape)
    print("测试数据集:", test_vectors.shape)

    np.savez('data/train_data', train_vectors=train_vectors, train_label=train_label)
    np.savez('data/test_data', test_vectors=test_vectors, test_label=test_label)
