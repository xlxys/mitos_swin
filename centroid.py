import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
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
        epsilon = 1e-8 
        m1, m2 = np.sum(p1 * b1) / (q1 + epsilon), np.sum(p2 * b2) / (q2 + epsilon) 
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / (q1 + epsilon), np.sum(((b2 - m2) ** 2) * p2) / (q2 + epsilon)
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





def draw_centroids(image, num_labels, centroids):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):
        x, y = centroids[i]
        cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)
    return image



def get_blobs_stats(image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image.astype(np.uint8), connectivity=8)
    return num_labels, labels, stats, centroids

def remove_small_blobs(image, labels, num_labels, stats, min_area):
    filtered_image = np.zeros_like(image)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered_image[labels == label] = image[labels == label]
    return filtered_image

def pipeline(image, min_area):
    num_labels, labels, stats, centroids = get_blobs_stats(image)
    image = remove_small_blobs(image, labels, num_labels, stats, min_area)
    num_labels, labels, stats, centroids = get_blobs_stats(image)
    return image, num_labels, centroids

def score_filtering(segmentation_map, area_threshold=590, confidence_threshold=0.5, sigma=1.5):
    # Step 1: Gaussian Smoothing
    smoothed_map = gaussian_filter(segmentation_map, sigma=sigma)

    # Step 2: Binarization using Otsu's method
    otsu_threshold = threshold_otsu(smoothed_map)
    binary_map = smoothed_map > otsu_threshold

    # Step 3: Label connected components
    num_labels, labels_im = cv2.connectedComponents(binary_map.astype(np.uint8))

    # Step 4: Area and Score Filtering
    final_detection_map = np.zeros_like(segmentation_map)
    
    for label in range(1, num_labels):
        blob = (labels_im == label)
        area = np.sum(blob)
        mean_score = np.mean(segmentation_map[blob])

        if area >= area_threshold and mean_score >= confidence_threshold:
            final_detection_map[blob] = binary_map[blob]

    return final_detection_map

image_dir = 'stitchedImages'

if not os.path.exists('results/masks'):
    os.makedirs('results/masks')
if not os.path.exists('results/csv'):
    os.makedirs('results/csv')

for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(f"Processing image: {image_name}")
    image_filtered = score_filtering(image, area_threshold=600, confidence_threshold=0.5, sigma=1.5)

    _, num_labels, center = pipeline(image_filtered, 590)


    # Normalize the filtered image to 0-255 range before saving
    norm_filtered = cv2.normalize(image_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    norm_filtered = norm_filtered.astype(np.uint8)

    cv2.imwrite(f'results/masks/{image_name}', norm_filtered)

    # Save centroids to a CSV file
    with open(f'results/csv/{image_name}.csv', 'w') as file:
        file.write('x,y\n')
        for x, y in center[1:]:
            file.write(f'{x},{y}\n')




