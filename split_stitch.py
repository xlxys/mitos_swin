import os 
import cv2
import glob
import numpy as np


# we divide the test image into 500x500px patches 
def divide_image(image, size):
    patches = []
    for i in range(0, image.shape[0], size):
        for j in range(0, image.shape[1], size):
            patches.append(image[i:i+size, j:j+size])
    return patches

# we process the patches by resizing them to 224x224px
def process_image(image, size):
    processed_image = cv2.resize(image, (size, size))
    return processed_image


def stitchPatchImg(testpath, dir1, imgname, savefolder):
  # stitch the visual feature map of patches to feature map of full image
	name = os.path.join(dir1, imgname)
	# print("name:"+name)
	Im = testpath + name
	I = [None]*16
	for i in range(9):
		print(Im+'_0'+str(i+1)+'.bmp')
		patch = cv2.imread(Im+'_0'+str(i+1)+'.bmp')
		I[i] = patch
	for i in range(9,16):
		print(Im+'_'+str(i+1)+'.bmp')
		patch = cv2.imread(Im+'_'+str(i+1)+'.bmp')
		I[i] = patch
	#print(I)
	A = np.zeros((4*500,4*500,3))
	for row in range(4):
		for col in range(4):
			A[row*500:(row+1)*500,col*500:(col+1)*500] = I[row*4+col]
	#print(A)

	
	cv2.imwrite(savefolder+'.jpg', A)
	# print( savefolder + '.jpg')


def resize_output_image(image_path, output_save_path):
    # Load the output image
	image_files = [f for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
	for image_file in image_files:
		image_path2=image_path+image_file
		img = cv2.imread(image_path2)
		# print("this is the output: "+image_path2)
		# print("------------------------------------------")

        # Resize the output image back to 500x500 pixels
		resized_img = cv2.resize(img, (500, 500), interpolation = cv2.INTER_CUBIC)

        # Save the resized image
		cv2.imwrite(output_save_path+image_file, resized_img)

# Usage:
#resize_output_image('D:\\PFE\\MASTER2\\2nd_test\\AMIDA13\\test\\test224\\1.jpg', 'D:\\PFE\\MASTER2\\2nd_test\\AMIDA13\\test\\test224\\2.jpg')

# root_folder = 'D:\\PFE\\MASTER2\\2nd_test\\AMIDA13\\test\\4imgs\\'
# dir1='13'
# imgname='01.bmp'
# savefolder="D:\\PFE\\MASTER2\\2nd_test\\AMIDA13\\test\\test_stitch\\13\\"
#stitchPatchImg(root_folder,dir1,imgname,savefolder)


def stitch(imagefolder224,savefolderto500,savefolderto2000,dirname,testpath):
	for j in dirname:
		imagefolder=imagefolder224+j+'\\'
		resize_output_image(imagefolder, savefolderto500+j+'\\')
	for i in dirname:
		files = glob.glob(testpath + i + '\\*')
		files.sort()
		last_file = files[-1]
		# print("Last file in the folder:", last_file)
		for k in range(int(last_file[-9:-7])):
			stitchPatchImg(testpath,i,f"{k:02d}",savefolderto2000)


dirname = ['13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23','24','25','26']
imagefolder224='D:\\PFE\\MASTER2\\2nd_test\\AMIDA13\\test\\test224\\'
savefolderto500='D:\\PFE\\MASTER2\\2nd_test\\AMIDA13\\test\\test500\\'
savefolderto2000='D:\\PFE\\MASTER2\\2nd_test\\AMIDA13\\test\\test2000\\'
testpath='D:\\PFE\\MASTER2\\2nd_test\\AMIDA13\\test\\4imgs\\'
stitch(imagefolder224,savefolderto500,savefolderto2000,dirname,testpath)