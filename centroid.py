import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian2d(sigma, size):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-size//2)**2 + (y-size//2)**2)/(2*sigma**2)), (size, size))
    return kernel

def apply_gaussian_filter(image, sigma, size):
    kernel = gaussian2d(sigma, size)
    return cv2.filter2D(image, -1, kernel)

def otsu(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])
        q1, q2 = Q[i], Q[255] - Q[i]
        b1, b2 = np.hsplit(bins, [i])
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh


def apply_binarisation(image):
    thresh = otsu(image)
    return cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]


def apply_morphological_operations(image):
    kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def get_blobs_stats(image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    return num_labels, labels, stats, centroids


def remove_small_blobs(image, labels, num_labels, stats, min_area):
  for i in range(1, num_labels):
    _, _, _, _, area = stats[i]
    if area < min_area:
      image[labels == i] = 0
  return image


def draw_centroids(image, num_labels, centroids):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):
        x, y = centroids[i]
        cv2.circle(image, (int(x), int(y)), 1, (255, 0, 0), -1)
    return image


def pipeline(image, sigma, size, min_area):
    image = apply_gaussian_filter(image, sigma, size)
    image = apply_binarisation(image)
    image = apply_morphological_operations(image)
    num_labels, labels, stats, centroids = get_blobs_stats(image)
    image = remove_small_blobs(image, labels, num_labels, stats, min_area)
    num_labels, labels, stats, centroids = get_blobs_stats(image)
    image = draw_centroids(image, num_labels, centroids)
    return image, num_labels, centroids


