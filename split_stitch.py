import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# we divide the test image into 500x500px patches 
def divide_image(image, size):
    patches = []
    for i in range(0, image.shape[0], size):
        for j in range(0, image.shape[1], size):
            patches.append(image[i:i+size, j:j+size])
    return patches

# we process the patches by resizing 
def process_image(image, size):
    processed_image = cv2.resize(image, (size, size))
    return processed_image


# we stitch the patches back together to form the original image
def stitch_image(patches, image_shape):
		image = np.zeros(image_shape)
		idx = 0
		size = patches[0].shape[0]
		for i in range(0, image.shape[0], size):
				for j in range(0, image.shape[1], size):
						image[i:i+size, j:j+size] = patches[idx]
						idx += 1
		return image


def split(image_path, size):
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		patches = divide_image(image, size)
		processed_patches = [process_image(patch, size) for patch in patches]
		return processed_patches


def stitch(patches, image_shape):
		# process the patches by resizing them to the original size
		patches = [cv2.resize(patch, (500, 500)) for patch in patches]
		# stitch the patches back together to form the original image
		image = stitch_image(patches, image_shape)
		return image









# img = cv2.imread('test/images/13/06.tif', cv2.IMREAD_GRAYSCALE)
# patches = split('test/images/13/06.tif', 500)


# # display the patches plot
# fig, axs = plt.subplots(1, len(patches), figsize=(20, 20))
# for i, patch in enumerate(patches):
# 		axs[i].imshow(patch, cmap='gray')
# 		axs[i].axis('off')
# plt.show()

# # stitch the patches back together to form the original image
# stitched_image = stitch(patches, img.shape)

# # save the stitched image
# cv2.imwrite('stitched_image.png', stitched_image)


# # display the stitched image and the original image
# fig, axs = plt.subplots(1, 2, figsize=(20, 20))
# axs[0].imshow(img, cmap='gray')
# axs[0].set_title('Original Image')
# axs[0].axis('off')
# axs[1].imshow(stitched_image, cmap='gray')
# axs[1].set_title('Stitched Image')
# axs[1].axis('off')
# plt.show()











