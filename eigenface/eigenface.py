import numpy as np
from train_data import plot_gallery, show_img
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle

def eigen_train(trainset, k = 20):
    """
    训练特征脸（eigenface）算法的实现

    :param trainset: 使用 get_images 函数得到的处理好的人脸数据训练集
    :param K: 希望提取的主特征数
    :return: 训练数据的平均脸, 特征脸向量, 中心化训练数据
    """

    # 均值人脸
    avg_img = np.mean(trainset, 0)
    # 中心化人脸
    norm_img = trainset - avg_img
    # 协方差矩阵
    covar = np.dot(norm_img, norm_img.T) / (norm_img.shape[0] - 1)
    e, u = np.linalg.eig(covar)
    # 左奇异向量
    sorted_indices = np.argsort(-e)
    lst = []
    for idx in sorted_indices[:k]:
        lst.append(u[idx])
    u_left = np.array(lst)
    # 右奇异向量
    feature = np.dot(u_left, norm_img)
    # 范数归一化
    feature_normalized = preprocessing.normalize(feature, norm='l2')

    # 返回：平均人脸、特征人脸、中心化人脸
    return avg_img, feature, norm_img


def rep_face(image, avg_img, eigenface_vects, numComponents=0):
    """
    用特征脸（eigenface）算法对输入数据进行投影映射，得到使用特征脸向量表示的数据

    :param image: 输入数据
    :param avg_img: 训练集的平均人脸数据
    :param eigenface_vects: 特征脸向量
    :param numComponents: 选用的特征脸数量
    :return: 输入数据的特征向量表示, 最终使用的特征脸数量
    """

    numEigenFaces = numComponents if numComponents < eigenface_vects.shape[0] else eigenface_vects.shape[0]
    eigenfaces_vects_use = eigenface_vects[:numEigenFaces, :]
    mean_img = image - avg_img
    representation = np.dot(mean_img, eigenfaces_vects_use.T)

    # 返回：输入数据的特征向量表示, 特征脸使用数量
    return representation, numEigenFaces


def recFace(representations, avg_img, eigenVectors, numComponents, sz=(112, 92)):
    """
    利用特征人脸重建原始人脸

    :param representations: 表征数据
    :param avg_img: 训练集的平均人脸数据
    :param eigenface_vects: 特征脸向量
    :param numComponents: 选用的特征脸数量
    :param sz: 原始图片大小
    :return: 重建人脸, str 使用的特征人脸数量
    """

    numEigenFaces = numComponents if numComponents < eigenVectors.shape[0] else eigenVectors.shape[0]
    eigenface_vects = eigenVectors[:numEigenFaces, :] + avg_img
    face = np.dot(representations, eigenface_vects)

    # 返回: 重建人脸, str 使用的特征人脸数量
    return face, 'numEigenFaces_{}'.format(numComponents)


if __name__ == '__main__':
    train_data, test_data = np.load('data/train_data.npz'), np.load('data/test_data.npz')

    train_vectors, train_label, test_vectors, test_label = train_data['train_vectors'], train_data['train_label'], \
                                                           test_data['test_vectors'], test_data['test_label']
    train_vectors = train_vectors / 255
    test_vectors = test_vectors / 255

    res = []
    x = [i for i in range(1, train_vectors.shape[0])]

    for i in range(150, 151):
        num_eigenface = i
        # 返回平均人脸、特征人脸、中心化人脸
        avg_img, eigenface_vects, trainset_vects = eigen_train(train_vectors, num_eigenface)

        # 打印两张特征人脸作为展示
        eigenfaces = eigenface_vects.reshape((num_eigenface, 112, 92))
        eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
        plot_gallery(eigenfaces, eigenface_titles, n_row=3, n_col=5)

        train_reps = []
        for img in train_vectors:
            train_rep, _ = rep_face(img, avg_img, eigenface_vects, num_eigenface)
            train_reps.append(train_rep)

        num = 0
        for idx, image in enumerate(test_vectors):
            label = test_label[idx]
            test_rep, _ = rep_face(image, avg_img, eigenface_vects, num_eigenface)

            results = []
            for train_rep in train_reps:
                similarity = np.sum(np.square(train_rep - test_rep))
                results.append(similarity)
            results = np.array(results)

            if label == np.argmin(results) // 5 + 1:
                num = num + 1

        print("N={}, 人脸识别准确率: {}%".format(i, num / 80 * 100))
        res.append(num / 80 * 100)

    with open("result/res", "wb") as f:
        pickle.dump(res, f)

    # 展示不同能量值对应测试结果
    plt.figure('eigenface')
    ax = plt.gca()
    ax.set_xlabel('num_eigenface')
    ax.set_ylabel('accuracy')
    ax.plot(x, res, color='b', linewidth=1, alpha=0.6)
    plt.show()

    print("重建训练集人脸")
    # 读取train数据
    image = train_vectors[100]

    faces = []
    names = []
    # 选用不同数量的特征人脸重建人脸
    for i in range(20, 200, 20):
        representations, numEigenFaces = rep_face(image, avg_img, eigenface_vects, i)
        face, name = recFace(representations, avg_img, eigenface_vects, numEigenFaces)
        faces.append(face)
        names.append(name)

    plot_gallery(faces, names, n_row=3, n_col=3)

    print("-" * 55)
    print("重建测试集人脸")

    # 读取test数据
    image = test_vectors[54]
    faces = []
    names = []
    # 选用不同数量的特征人脸重建人脸
    for i in range(20, 200, 20):
        representations, numEigenFaces = rep_face(image, avg_img, eigenface_vects, i)
        face, name = recFace(representations, avg_img, eigenface_vects, numEigenFaces)
        faces.append(face)
        names.append(name)

    plot_gallery(faces, names, n_row=3, n_col=3)
