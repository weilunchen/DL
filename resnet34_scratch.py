import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader
from torchinfo import summary
import pdb
import traceback
from rich import inspect
from rich.console import Console
from rich.traceback import install

install()
post_mortem = True 
disable_cuda = True
torch.autograd.set_detect_anomaly(True)

try:
	if disable_cuda:
		os.environ["CUDA_VISIBLE_DEVICES"]=""

	class ResNetStart(nn.Module):
		def __init__(self, in_channels, out_channels, stride=2):
			super(ResNetStart, self).__init__()
			self.conv = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=(7, 7), stride=stride, padding = 3),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(),
			)
		
		def forward(self, x):
			return self.conv(x)

	class ResNetUnit(nn.Module):
		def __init__(self, in_channels, out_channels, stride=1):
			super(ResNetUnit, self).__init__()
			self.conv = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(),
				nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
				nn.BatchNorm2d(out_channels),
				nn.ReLU()
			)

		def forward(self, x):
			return self.conv(x)

	class View(nn.Module):
		def __init__(self, shape):
			super(View, self).__init__()

			self.shape = shape

		def forward(self, x):
			return x.view(self.shape)


	class UpSampleBlock(nn.Module):
		def __init__(self, in_channels, out_channels, reduction=2):
			super(UpSampleBlock, self).__init__()

			self.in_channels = in_channels
			self.out_channels = out_channels
			self.reduction = reduction

		def forward(self, x):
			x = nn.ReLU(inplace=True)(x)
			x = nn.BatchNorm2d(self.in_channels)(x)

			#Concurrent Spatial, Channel Squeeze and Excitation layer (scSE)
			x_before_squeeze = x

			original_height = x.shape[2]
			original_width = x.shape[3]
			
			# Global average pool
			x = nn.AvgPool2d(kernel_size=(original_height, original_height))(x)

			tensor_shape = x.shape
			n = tensor_shape[0]
			channel_size = tensor_shape[1]
			# Height and width should be 1 after global average pool

			linear_in_channels = np.product(channel_size)
			linear_shape = [n, linear_in_channels]

			# Spatial Squeeze and Channel Excitation (cSE)
			x = View(linear_shape)(x)
			x = nn.Linear(linear_in_channels, linear_in_channels // self.reduction)(x)
			x = nn.ReLU(inplace=True)(x)
			x = nn.Linear(linear_in_channels // self.reduction, linear_in_channels)(x)
			x = nn.Sigmoid()(x)
			x = View(tensor_shape)(x)

			x_cSE = x_before_squeeze * x

			# Channel Squeeze and Spatial Excitation (sSE)
			x = x_before_squeeze

			x = nn.Conv1d(self.in_channels, 1, kernel_size=(1, 1))(x)
			x = nn.Sigmoid()(x)

			x_sSE = x_before_squeeze * x

			# Combining cSE and sSE

			x = x_sSE + x_cSE

			# After squeeze, transpose Conv with kernel of 2x2 and stride of 2
			x = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=(2, 2), stride=2, bias=False)(x)

			return x
		

	class ResNet34(nn.Module):
		def __init__(self, in_channels=3, out_channels=1000):
			super(ResNet34, self).__init__()

			self.start_block = nn.ModuleList()
			self.res_blocks = nn.ModuleList()
			self.end_block = nn.ModuleList()

			channel_size = 64
			double_channel_size = channel_size * 2

			# Start block (64)
			self.start_block.append(nn.Conv2d(in_channels, channel_size, 3, stride=2, padding=1))
			self.start_block.append(nn.MaxPool2d(kernel_size=2, stride=2))

			# First block list (64)
			res_block = nn.ModuleList()
			res_block.append(ResNetUnit(channel_size, channel_size))
			res_block.append(ResNetUnit(channel_size, channel_size))
			res_block.append(ResNetUnit(channel_size, channel_size))
			self.res_blocks.append(res_block)

			# Second block list (128)
			res_block = nn.ModuleList()
			res_block.append(ResNetUnit(channel_size, double_channel_size, stride=2))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			self.res_blocks.append(res_block)

			channel_size = double_channel_size
			double_channel_size = channel_size * 2

			# Third block list (256)
			res_block = nn.ModuleList()
			res_block.append(ResNetUnit(channel_size, double_channel_size, stride=2))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			self.res_blocks.append(res_block)

			channel_size = double_channel_size
			double_channel_size = channel_size * 2

			# Fourth block list (512)
			res_block = nn.ModuleList()
			res_block.append(ResNetUnit(channel_size, double_channel_size, stride=2))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			self.res_blocks.append(res_block)

			# End block (1000)
			self.end_block.append(nn.AvgPool2d(kernel_size=1, stride=1))
			self.end_block.append(nn.Linear(double_channel_size, 1000))

		def forward(self, x):
			x = self.start_block(x)

			for block_index, res_block in enumerate(self.res_blocks):
				for unit_index, res_unit in enumerate(res_block):
					downsampling_needed = block_index > 0 and unit_index == 0
					
					if downsampling_needed:
						in_channels = x.shape[1]
						out_channels = in_channels * 2
						identity = nn.Conv2d(in_channels, out_channels, 1, stride = 2)(x)
					else:
						identity = x

					x = res_unit(x)
					x += identity
					x = nn.ReLU(x, inplace=True)

			return self.end_block(x)


	class UNet(nn.Module):
		def __init__(self, in_channels=3, out_channels=1):
			super(UNet, self).__init__()

			self.res_blocks = nn.ModuleList()
			self.ups = nn.ModuleList()

			channel_size = 64
			double_channel_size = channel_size * 2

			# Zeroth/Start block (64)
			self.start_block = ResNetStart(in_channels, channel_size)

			# First block list (64)
			res_block = nn.ModuleList()
			res_block.append(ResNetUnit(channel_size, channel_size))
			res_block.append(ResNetUnit(channel_size, channel_size))
			res_block.append(ResNetUnit(channel_size, channel_size))
			self.res_blocks.append(res_block)

			# Second block list (128)
			res_block = nn.ModuleList()
			res_block.append(ResNetUnit(channel_size, double_channel_size, stride=2))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			self.res_blocks.append(res_block)

			channel_size = double_channel_size
			double_channel_size = channel_size * 2

			# Third block list (256)
			res_block = nn.ModuleList()
			res_block.append(ResNetUnit(channel_size, double_channel_size, stride=2))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			self.res_blocks.append(res_block)

			channel_size = double_channel_size
			double_channel_size = channel_size * 2

			# Fourth block list (512)
			res_block = nn.ModuleList()
			res_block.append(ResNetUnit(channel_size, double_channel_size, stride=2))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			self.res_blocks.append(res_block)

			# Fourth level (This is bottom now)
			self.ups.append(UpSampleBlock(double_channel_size, 128))
			self.ups.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))

			# Third level
			self.ups.append(UpSampleBlock(256, 128))
			self.ups.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))

			# Second level
			self.ups.append(UpSampleBlock(256, 128))
			self.ups.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))

			# First level
			self.ups.append(UpSampleBlock(256, 128))
			self.ups.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))

			# Zeroth level. Final convolution to obtain output
			self.final_conv = UpSampleBlock(256, 1)


		def forward(self, x):
			# UNet skip connections
			skip_connections = []

			x = self.start_block(x)

			for block_index, res_block in enumerate(self.res_blocks):
				skip_in_channels = x.shape[1]
				skip_out_channels = 128

				skip_connection = nn.Conv2d(skip_in_channels, skip_out_channels, 1, stride = 1)(x)
				skip_connections.append(skip_connection)

				if block_index == 0:
					x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

				for unit_index, res_unit in enumerate(res_block):
					downsampling_needed = block_index > 0 and unit_index == 0

					if downsampling_needed:
						in_channels = x.shape[1]
						out_channels = in_channels * 2
						identity = nn.Conv2d(in_channels, out_channels, 1, stride = 2)(x)
					else:
						identity = x

					x = res_unit(x)
					x += identity
					x = nn.ReLU(inplace=True)(x)

			# Mid edge at the bottom of UNet
			x = nn.Conv2d(512, 512, 1, stride = 1)(x)

			skip_connections = skip_connections[::-1]

			for i in range(0, len(self.ups), 2):
				skip_connection = skip_connections[i//2]
				x_skip = skip_connection
				x = self.ups[i](x)
				
				concat_skip_connection = torch.cat((x_skip, x), dim=1)
				x = self.ups[i+1](concat_skip_connection) 

			x = self.final_conv(x)

			return x
		
	def test():
		n = 3
		in_channels = 3
		height = 320
		width = 320

		x = torch.randn((n, in_channels, height, width))
		model = UNet(in_channels=in_channels, out_channels=1)
		preds = model(x)
		print(preds.shape)
		print(x.shape)

	def train(model, train_loader, test_loader, epochs, criterion, optimizer, schedular):
		best_model = model.state_dict()
		best_index = 0

		max_acc = 0.0
		max_precision = 0.0
		max_recall = 0.0
		max_f1_score = 0.0
		min_loss = 0.0

		losses = []
		accuracies = []
		precisions = []
		recalls = []
		f1_scores = []

		for epoch in range(epochs):
			epoch_loss = 0.0
			epoch_acc = 0.0

			count = 0

			for data, target in train_loader:
				model.train(True)

				print("Epoch: " + str(epoch) + " Train: " + str(count))

				input = Variable(data.type(torch.FloatTensor))
				target = Variable(target.type(torch.LongTensor))

				optimizer.zero_grad()

				if torch.cuda.is_available():
					input = input.cuda()
					target = target.cuda()

				output = model(input)
				loss = criterion(output, target)

				loss.backward()
				optimizer.step()
				count += 1
			
			count = 0
			run_loss = 0.0
			run_total = 0
			run_corrects = 0
			run_true_positive = 0
			run_false_positive = 0
			run_true_negative = 0
			run_false_negative = 0

			for data, target in test_loader:
				print("Epoch: " + str(epoch) + " Test: " + str(count))

				model.train(False)

				input = Variable(data.type(torch.FloatTensor))
				target = Variable(target.type(torch.LongTensor))

				if torch.cuda.is_available():
					input = input.cuda()
					target = target.cuda()
				
				output = model(input)
				preds = torch.clamp(torch.round(output.data.squeeze(1)), min=0, max=1)
				loss = criterion(output, target)

				run_loss += loss.item()
				run_total += np.product(preds.shape)
				run_corrects += torch.sum(preds == target.data.squeeze(1))
				run_true_positive += torch.sum((preds == 1) & (target.data.squeeze(1) == 1))
				run_false_positive += torch.sum((preds == 1) & (target.data.squeeze(1) == 0))
				run_true_negative += torch.sum((preds == 0) & (target.data.squeeze(1) == 0))
				run_false_negative += torch.sum((preds == 0) & (target.data.squeeze(1) == 1))

				count += 1


			precision = run_false_positive.true_divide(run_true_positive + run_false_positive)
			recall = run_true_positive.true_divide(run_true_positive + run_false_negative)
			f1_score = (2 * precision * recall).true_divide(precision + recall)
			epoch_acc = run_true_positive.true_divide(run_total)

			print("==============================================")
			print(f'Epoch {epoch} stats:')
			print(f'Dice loss: {run_loss}')
			print(f'Precision: {precision}')
			print(f'Recall: {recall}')
			print(f'F1 score: {f1_score}')
			print(f'Accuracy: {epoch_acc}')
			print('')
			print(f'TP: {run_true_positive}')
			print(f'FP: {run_false_positive}')
			print(f'TN: {run_true_negative}')
			print(f'FN: {run_false_negative}')
			print("==============================================")

			accuracies.append(epoch_acc)
			precisions.append(precision)
			recalls.append(recall)
			f1_scores.append(f1_score)
			losses.append(run_loss)


			if run_loss < min_loss:
				print("Best loss yet: " + str(run_loss))
				max_precision = precision

			if precision > max_precision:
				print("Best precision yet: " + str(precision))
				max_precision = precision
				best_index = epoch
				best_model = model.state_dict()

			if recall > max_recall:
				print("Best recall yet: " + str(recall))
				max_recall = recall

			if f1_score > max_f1_score:
				print("Best F1 score yet: " + str(f1_score))
				max_f1_score = f1_score

			if epoch_acc > max_acc:
				print("Best accuracy yet: " + str(epoch_acc))
				max_acc = epoch_acc
		

		print("Eval Done")
		print("==============================================")
		print(f"Best model precision: {precisions[best_index]}")
		print(f"Best model recall: {recalls[best_index]}")
		print(f"Best model F1 score: {f1_scores[best_index]}")
		print(f"Best model accuracy: {accuracies[best_index]}")
		print(f"Best model loss: {losses[best_index]}")

		print(f"All precisions: {precisions}")
		print(f"All recalls: {recalls}")
		print(f"All F1 scores: {f1_scores}")
		print(f"All accuracies: {accuracies}")
		print(f"All losses: {losses}")
		print("==============================================")

		return best_model

	class diceCoefficientLoss(nn.Module):
		def __init__(self):
			super(diceCoefficientLoss, self).__init__()
		def forward(self, pred, target):
			pred_flat = torch.clamp(torch.round(pred.contiguous().view(-1)), min=0, max=1)
			targ_flat = torch.clamp(target.contiguous().view(-1), min=0, max=1)
			intersection = (pred_flat * targ_flat).sum()
			pred_sum = torch.sum(pred_flat * pred_flat)
			targ_sum = torch.sum(targ_flat * targ_flat)
			return 1 - ((2. * intersection) / (pred_sum + targ_sum))

	class MnistDataset(Dataset):
		def __init__(self, original_dir, processed_dir, image_size=32, batch_size=256):
			self.original_dir = original_dir
			self.processed_dir = processed_dir
			self.images = os.listdir(str(os.path.join(original_dir, str(image_size))))
			self.batch_size = batch_size
			# Image sizes for MNIST: 32, 128, 256, 320
			self.image_size = image_size
		
		def __len__(self):
			return len(self.images)

		def __getitem__(self, index):
			original_path = os.path.join(self.original_dir, str(self.image_size), f'{str(index)}.bmp')
			processed_path = os.path.join(self.original_dir, str(self.image_size), f'{str(index)}.bmp')

			original = np.array(Image.open(original_path).convert("L"))
			original = np.expand_dims(original, axis=0)
			processed = np.array(Image.open(processed_path).convert("L"))
			processed[processed == 255.0] = 1.0
			processed = np.expand_dims(processed, axis=0)

			return original, processed

		def get_loader(self):
			return DataLoader(self, batch_size=self.batch_size)


	# class AdamW(optim.optimizer):
	# 	def __init__(self, params):
	# 		defaults = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
	# 		super(AdamW, self).__init__(params, defaults)
		

	if __name__ == "__main__":
		#test()

		device = 'cuda' if torch.cuda.is_available() else 'cpu'

		in_channels = 1
		model = UNet(in_channels=in_channels, out_channels=1).to(device)
		if torch.cuda.is_available():
			model = model.cuda()

		epochs = 2
		criterion = diceCoefficientLoss().to(device)
		optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
		schedular = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
		
		train_set = MnistDataset('data/MNIST/segmentation/train/original/', 'data/MNIST/segmentation/train/processed/')
		train_loader = train_set.get_loader()

		test_set = MnistDataset('data/MNIST/segmentation/test/original/', 'data/MNIST/segmentation/test/processed/')
		test_loader = test_set.get_loader()

		#print(summary(self, input_size=(256, 1, 28, 28), device='cpu', verbose=2))

		best_model = train(model, train_loader, test_loader, epochs, criterion, optimizer, schedular)


except Exception as e:
	if post_mortem:
		Console().print_exception()
		pdb.post_mortem() # Post mortem debugging
	else:
		raise e
