import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from genericpath import isdir
from operator import is_not
from os import mkdir

def words_segmentation(path):
    lines = read_lines(path)
    process_lines_img(lines)
    total_paws = 0
    paws_img_list = []
    for line in lines:
        hist_v = histogram_vertical(line)
        nb_paws, pos_paws, paws_img = detection_paws(line, hist_v)
        total_paws += nb_paws
        paws_img_list.extend(paws_img)

    save_paws(paws_img_list)
    show_paws(paws_img_list)


def histogram_vertical(bw):
    # take a binarised img
    # return histogram vertical d img
    w = bw.shape[1]
    hist_v = np.zeros((w), np.uint16)
    for i in range(w):
        hist_v[i] = np.count_nonzero(bw[:, i])
    return hist_v


def read_lines(path):

    files = os.listdir(path)
    lines_img = []
    for i in range(len(files)):
        print(path+files[i])
        img1 = cv.imread(str(path)+'/'+str(files[i]))
        lines_img.append(img1)
        cv.imshow(str(i), lines_img[i])
    cv.waitKey(0)
    cv.destroyAllWindows()
    return lines_img

def process_lines_img(lines_img):
    for line in lines_img:
        line = cv.cvtColor(line, cv.COLOR_BGR2GRAY)
        line = cv.threshold(line, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]


def detection_paws(im, hist):
    pos_paws = []
    paws_img = []
    nb_paws = 0
    h = len(hist)
    deb = 0
    fin = 0
    i = 0
    while(i < h-1):
        if((hist[i] == 0) and (hist[i + 1] > 0)):
            nb_paws += 1
            deb = i

            for j in range(deb, h-1):
                if((hist[j] > 0) and(hist[j + 1] == 0)):
                    fin = j+1
                    pos_paws.append([deb, fin])
                    paw = im[0:, deb:fin]
                    paws_img.append(paw)
                    break
                else:
                    j += 1
            i = fin
        else:
            i += 1
    return nb_paws, pos_paws, paws_img

def save_paws(paws):
    for i in range(len(paws)):
        cv.imwrite('Paws/paw'+str(i)+'.png', paws[i])

def show_paws(paws):
    plt.figure(figsize=(20,18))
    for i in range(1, len(paws)+1):
        plt.subplot(len(paws)//3+1,3,i)
        plt.imshow(~paws[-i], cmap=plt.cm.gray)
        plt.title('PAW'+str(i))
        plt.axis('off')

    plt.savefig('Paws/paws.png')
    plt.show()