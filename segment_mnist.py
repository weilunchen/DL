from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torchvision.datasets as datasets
import torchvision.transforms.functional as functional
from pathlib import Path

trains = [True, False]

for train in trains:
	if train:
		print('Generating train data in data/MNIST/segmentation/train/')
	else:
		print('Generating test data in data/MNIST/segmentation/test/')

	mnist_trainset = datasets.MNIST(root='./data', train=train, download=True, transform=None)

	# posterize = functional.posterize(bits=1)
	# mnist_post = datasets.MNIST(root='./data', train=True, download=True, transform=posterize)


	index = 0

	for orig_image, label in mnist_trainset:
		#MyImg = Image.new( 'RGB', (250,250), "black")
		#Imported_Img = Image.open('ImageName.jpg') 
		#use the commented code to import from our own computer

		image = orig_image.copy()
		pixels = image.load()

		for i in range(image.size[0]):        #for each column
			for j in range(image.size[1]):    #For each row
				pixels[i,j] = 255 if pixels[i,j] > 127 else 0    #set the colour according to your wish

		if train:
			Path("data/MNIST/segmentation/train/original/").mkdir(parents=True, exist_ok=True)
			Path("data/MNIST/segmentation/train/processed/").mkdir(parents=True, exist_ok=True)
			orig_image.save(f'data/MNIST/segmentation/train/original/{index}.bmp')
			image.save(f'data/MNIST/segmentation/train/processed/{index}.bmp')
		else:
			Path("data/MNIST/segmentation/test/original/").mkdir(parents=True, exist_ok=True)
			Path("data/MNIST/segmentation/test/processed/").mkdir(parents=True, exist_ok=True)
			orig_image.save(f'data/MNIST/segmentation/test/original/{index}.bmp')
			image.save(f'data/MNIST/segmentation/test/processed/{index}.bmp')

		index += 1
		# f, axarr = plt.subplots(2)
		# axarr[0].imshow(orig_image)
		# axarr[1].imshow(image)
		# plt.show()

	