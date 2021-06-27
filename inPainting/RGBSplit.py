import cv2  # opencv库
from matplotlib import pyplot as plt  # 展示图片
import numpy as np
import random

def read_image(img_path):
    """
    读取图片，图片是以 np.array 类型存储
    :param img_path: 图片的路径以及名称
    :return: img np.array 类型存储
    """
    # 读取图片
    img = cv2.imread(img_path)

    # # 如果图片是三通道，采用 matplotlib 展示图像时需要先转换通道
    # if len(img.shape) == 3:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def plot_image(image, image_title, is_axis=False):
    """
    展示图像
    :param image: 展示的图像，一般是 np.array 类型
    :param image_title: 展示图像的名称
    :param is_axis: 是否需要关闭坐标轴，默认展示坐标轴
    :return:
    """
    # 展示图片
    plt.imshow(image)

    # 关闭坐标轴,默认关闭
    if not is_axis:
        plt.axis('off')

    # 展示受损图片的名称
    plt.title(image_title)

    # 展示图片
    plt.show()


def RGBSplit(img, bias=0, padding=60, arr=[0, 1, 2]):
    res_img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), constant_values=0)
    res_img[padding:-padding, padding + bias:-padding + bias, arr[0]] = res_img[padding:-padding, padding:-padding, arr[0]]
    res_img[padding:-padding, padding - bias:-padding - bias, arr[1]] = res_img[padding:-padding, padding:-padding, arr[1]]
    res_img[padding + bias:-padding + bias, padding:-padding, arr[2]] = res_img[padding:-padding, padding:-padding, arr[2]]
    return res_img[padding:-padding, padding:-padding, ]


if __name__ == '__main__':
    img_path = 'H.bmp'

    img = read_image(img_path)
    # plot_image(img, 'origin image')

    fps, size = 24, (img.shape[0], img.shape[1])
    out = cv2.VideoWriter('H.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, size)

    for cnt in range(400):
        arr = [0, 1, 2]
        random.shuffle(arr)
        for i in range(1, fps+1):
            out_img = RGBSplit(img, i//4, arr=arr)
            # plot_image(out_img, str(i))
            cv2.imshow('image', out_img)
            cv2.waitKey(1000 // fps)
            out.write(out_img)

        for i in range(fps, 0, -1):
            out_img = RGBSplit(img, i//4, arr=arr)
            # plot_image(out_img, str(i))
            cv2.imshow('image', out_img)
            cv2.waitKey(1000 // fps)
            out.write(out_img)
    out.release()
