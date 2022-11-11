import cv2 as cv

from preprocessing_image import preprocessing_image
from line_segmentation import lines_segmentation
from word_segmentation import words_segmentation

def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

def show_image(img):
    cv.imshow('s', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    img = cv.imread('data/Page3.png', cv.COLOR_BGR2GRAY)
    img = resize_image(img, 20)

    # img = preprocessing_image(img)
    # show_image(img)
    # print('finish preprocessing')

    lines_segmentation(img)
    print('finish line segmentation')

    words_segmentation('lines')
    print('finish word segmentation')

