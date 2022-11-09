import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from genericpath import isdir
from operator import is_not
from os import mkdir


def lines_segmentation(img):
    """
        takes an gray img as an entry
    """

    #img = cv.cvtColor(img, )
    cv.imshow('img', img)
    bw = binarisation(img)
    cv.imshow('bin', bw)
    hist_h = histogram_horizontal(bw)
    hist_v = histogram_vertical(bw)
    img_hist_h = create_img_hist_h(bw, hist_h)
    img_hist_v = create_img_hist_v(bw, hist_v)
    save_hist_results(img, img_hist_h, img_hist_v)
    show_results(img, img_hist_h, img_hist_v)
    nb_ligne, position_lignes, lignes_img = detection_ligne(bw, hist_h)
    save_lines(lignes_img)
    show_lines(lignes_img)





def binarisation(img):
    # take an gray image
    # return a binarised image
    return cv.threshold(img, 130, 255, cv.THRESH_BINARY_INV)[1]

# th,bw = cv.threshold(img, 130, 255, cv.THRESH_BINARY_INV)
# bw = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
# bw = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]

def histogram_horizontal(bw):
    # take a binarised img
    # return histogram horizontal d img
    h = bw.shape[0]
    hist_h = np.zeros((h), np.uint16)
    for i in range(h):
        hist_h[i] = np.count_nonzero(bw[i, :])
    return hist_h

def histogram_vertical(bw):
    # take a binarised img
    # return histogram vertical d img
    w = bw.shape[1]
    hist_v = np.zeros((w), np.uint16)
    for i in range(w):
        hist_v[i] = np.count_nonzero(bw[:, i])
    return hist_v

def create_img_hist_h(img, hist_h):
    im_hit_h = np.zeros(img.shape, np.uint8)
    im_hit_h[:, :] = 255
    h = img.shape[0]
    for i in range(h):
        im_hit_h[i, 0:hist_h[i]] = 0
    return im_hit_h

def create_img_hist_v(img, hist_v):
    im_hit_v = np.zeros(img.shape, np.uint8)
    im_hit_v[:, :] = 255
    w = img.shape[1]
    for i in range(w):
        im_hit_v[0:hist_v[i], i] = 0
    return im_hit_v

def save_hist_results(img, im_hist_h, im_hist_v):
    if not(isdir('results')):
        mkdir('results')
    cv.imwrite('results/hist_h.png', cv.hconcat([img, im_hist_h]))
    cv.imwrite('results/hist_v.png', cv.vconcat([img, im_hist_v]))

def show_results(img, im_hist_h, im_hist_v):
    cv.imshow('histHor', cv.hconcat([img, im_hist_h]))
    cv.imshow('histVert', cv.vconcat([img, im_hist_v]))
    cv.waitKey(0)
    cv.destroyAllWindows()

def detection_ligne(bw, hist):
    position_lignes = []
    lignes_img = []
    nb_ligne = 0
    h = len(hist)
    deb = 0
    fin = 0
    i = 0
    while(i < h-1):
        if((hist[i] == 0) and (hist[i + 1] > 0)):
            nb_ligne += 1
            deb = i

            for j in range(deb, h-1):
                if((hist[j] > 0) and(hist[j + 1] == 0)):
                    fin = j
                    position_lignes.append([deb, fin])
                    ligne = bw[deb:fin, 0:]
                    lignes_img.append(ligne)
                    break
                else:
                    j += 1
            i = fin +1
        else:
            i += 1
    return nb_ligne, position_lignes, lignes_img

def save_lines(lines_img):
    for i in range(len(lines_img)):
        cv.imwrite('results/line' + str(i) +'.png', lines_img[i])

def show_lines(lines):
    for line in lines:
        hist_h, hist_v = histogram_horizontal(line), histogram_vertical(line)
        im_hist_h = np.zeros(line.shape, np.uint8)
        im_hist_h[:,:] = 255
        im_hist_v = np.zeros(line.shape, np.uint8)
        im_hist_v[:, :] = 255
        cv.imshow('histH', cv.hconcat([~line,im_hist_h]))
        cv.imshow('histV', cv.hconcat([~line, im_hist_v]))
    cv.waitKey(0)
    cv.destroyAllWindows()
