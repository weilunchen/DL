from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torchvision.datasets as datasets
import torchvision.transforms.functional as functional
from pathlib import Path

trains = [True, False]
sizes = [32, 128, 256, 320]

print(f"Sizes: {sizes}")
for train in trains:
	if train:
		print('Generating train data in data/MNIST/segmentation/train/')
	else:
		print('Generating test data in data/MNIST/segmentation/test/')

	mnist_dataset = datasets.MNIST(root='./data', train=train, download=True, transform=None)

	index = 0

	for orig_image, label in mnist_dataset:
		if index % 1000 == 0:
			print(f"{index + 1} out of {len(mnist_dataset)}")

		for size in sizes:
			scaled_orig_image = orig_image.copy()
			scaled_orig_image = scaled_orig_image.resize((size, size))
			# image = orig_image.copy()
			# image = image.resize((size, size))

			# pixels = image.load()

			# for i in range(image.size[0]):        #for each column
			# 	for j in range(image.size[1]):    #For each row
			# 		pixels[i,j] = 255 if pixels[i,j] > 127 else 0    #set the colour according to your wish

			if train:
				Path(f"data/MNIST/segmentation/train/original/{size}/").mkdir(parents=True, exist_ok=True)
				#Path(f"data/MNIST/segmentation/train/processed/{size}/").mkdir(parents=True, exist_ok=True)
				scaled_orig_image.save(f'data/MNIST/segmentation/train/original/{size}/{index}.bmp')
				#image.save(f'data/MNIST/segmentation/train/processed/{size}/{index}.bmp')
			else:
				Path(f"data/MNIST/segmentation/test/original/{size}/").mkdir(parents=True, exist_ok=True)
				#Path(f"data/MNIST/segmentation/test/processed/{size}/").mkdir(parents=True, exist_ok=True)
				scaled_orig_image.save(f'data/MNIST/segmentation/test/original/{size}/{index}.bmp')
				#image.save(f'data/MNIST/segmentation/test/processed/{size}/{index}.bmp')

			# f, axarr = plt.subplots(2)
			# axarr[0].imshow(orig_image)
			# axarr[1].imshow(image)
			# plt.show()
		index += 1

	