import cv2 as cv

from line_segmentation import lines_segmentation
from word_segmentation import words_segmentation

if __name__ == '__main__':
    img = cv.imread('data/Page3.png', cv.COLOR_BGR2GRAY)
    lines_segmentation(img)
    path = 'lines'
    words_segmentation(path)

