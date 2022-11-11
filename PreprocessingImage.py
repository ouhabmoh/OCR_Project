import cv2 as cv
import numpy as np

# get gray scale image
def get_gray_scale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# remove noise
def remove_noise(img):
    return cv.medianBlur(img, 5)

#thresholding
def thresholding(img):
    return cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

#dilation
def dilate(img):
    kernel = np.ones((5,5),np.uint8)
    return cv.dilate(img, kernel, iterations = 1)

#erode
def erode(img):
    kernel = np.ones((5,5),np.uint8)
    return cv.erode(img, kernel, iterations = 1)

#erode followed by dilation
def opening(img):
    kernel = np.ones((5,5),np.uint8)
    return cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

#canny edge detection
def canny(img):
    return cv.Canny(img, 100, 200)

#skew correction
def deskew(img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(img, template):
    return cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)



# # main project exemple
# img = cv.imread('alpha.jpg')
# gray = get_gray_scale(img)
# thresh = thresholding(gray)
# opening = opening(thresh)
# canny = canny(opening)
#
# # for test image
# cv.imshow('image', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
