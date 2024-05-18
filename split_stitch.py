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



# # Define the directory containing the output patches
# output_dir = 'output'
# stitched_output_dir = 'StitchedImages'
# os.makedirs(stitched_output_dir, exist_ok=True)

# # Get sorted list of image files
# image_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])

# # load all image with the same prefix
# patches = []
# name='1301'
# for images in image_files:
# 		if images.endswith('.png'):
# 				image_path = os.path.join('output', images)
# 				if images.startswith(name):
# 						patches.append(image_path)
# 						print(f"{len(patches)} name: {name}")
# 				else:
# 					patches = sorted(patches)

# 					patches = [cv2.imread(patch, cv2.IMREAD_GRAYSCALE) for patch in patches]
					

# 					# # display the patches plot
# 					# fig, axs = plt.subplots(1, len(patches), figsize=(20, 20))
# 					# for i, patch in enumerate(patches):
# 					# 		axs[i].imshow(patch, cmap='gray')
# 					# 		axs[i].axis('off')
# 					# plt.show()
# 					print('Stitching patches together...')
# 					# stitch the patches back together to form the original image
# 					stitched_image = stitch(patches, (2000, 2000))

# 					# save the stitched image
# 					cv2.imwrite('StitchedImages/'+name+'.png', stitched_image)
# 					patches=[]
# 					stitched_image=[]
# 					print(images[0:4])
# 					name=images[0:4]


# Define the directory containing the output patches
output_dir = 'output'
stitched_output_dir = 'stitchedImages'
os.makedirs(stitched_output_dir, exist_ok=True)

# Get sorted list of image files
image_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])

# Load all images with the same prefix
patches = []
name = '1301'
for image_file in image_files:
    image_path = os.path.join(output_dir, image_file)
    if image_file.startswith(name):
        patches.append(image_path)
        print(f"{len(patches)} patches collected for prefix {name}")
    else:
        if patches:
            patches = sorted(patches)
            patches = [cv2.imread(patch, cv2.IMREAD_GRAYSCALE) for patch in patches]

            

            print('Stitching patches together...')
            # Stitch the patches back together to form the original image
            stitched_image = stitch(patches, (2000, 2000))

            # Save the stitched image
            stitched_image_name = os.path.join(stitched_output_dir, f'{name}.png')
            cv2.imwrite(stitched_image_name, stitched_image)
            patches = []

        # Update name and collect new patches
        name = image_file[:4]
        patches.append(image_path)

# Process remaining patches if any
if patches:
    patches = sorted(patches)
    print(f"Processing {len(patches)} patches for prefix {name}")
    patches = [cv2.imread(patch, cv2.IMREAD_GRAYSCALE) for patch in patches]

    print('Stitching patches together...')
    # Stitch the patches back together to form the original image
    stitched_image = stitch(patches, (2000, 2000))

    # Save the stitched image
    stitched_image_name = os.path.join(stitched_output_dir, f'{name}.png')
    cv_file_name = os.path.join(stitched_output_dir, stitched_image_name)
    cv2.imwrite(cv_file_name, stitched_image)

print("Stitching completed.")















