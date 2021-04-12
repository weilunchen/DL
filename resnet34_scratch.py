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
from torchviz import make_dot
import matplotlib.pyplot as plt

install()
post_mortem = True 
disable_cuda = False
torch.autograd.set_detect_anomaly(True)

try:
	if disable_cuda:
		os.environ["CUDA_VISIBLE_DEVICES"]=""

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

	class UpSampleBlock(nn.Module):
		def __init__(self, in_channels, out_channels, reduction=2):
			super(UpSampleBlock, self).__init__()

			self.in_channels = in_channels
			self.out_channels = out_channels
			self.reduction = reduction

			self.relu_batch = nn.Sequential(
				nn.ReLU(inplace=True),
				nn.BatchNorm2d(self.in_channels)
			)
			self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))

			self.sSE = nn.Sequential(
				nn.Linear(in_channels, in_channels // self.reduction),
				nn.ReLU(inplace=True).to(device),
				nn.Linear(in_channels // self.reduction, in_channels),
				nn.Sigmoid()
			)

			self.cSE = nn.Sequential(
				nn.Conv2d(self.in_channels, 1, kernel_size=(1, 1)),
				nn.Sigmoid()
			)

			self.final_conv = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=(2, 2), stride=2, bias=False)

		def forward(self, x):
			x = self.relu_batch(x)

			#Concurrent Spatial, Channel Squeeze and Excitation layer (scSE)
			x_before_squeeze = x

			original_height = x.shape[2]
			original_width = x.shape[3]
			
			# Global average pool
			x = self.global_avg_pool(x)

			tensor_shape = x.shape
			n = tensor_shape[0]
			channel_size = tensor_shape[1]
			# Height and width should be 1 after global average pool

			linear_in_channels = np.product(channel_size)
			linear_shape = [n, linear_in_channels]

			# Spacial Squeeze and Channel Excitation (cSE)
			x = x.view(linear_shape)
			x = self.sSE(x)
			x = x.view(tensor_shape)

			x_cSE = x_before_squeeze * x

			# Channel Squeeze and Spatial Excitation (sSE)
			x = x_before_squeeze
			x = self.cSE(x)

			x_sSE = x_before_squeeze * x

			# Combining cSE and sSE

			x = x_sSE + x_cSE

			# After squeeze, transpose Conv with kernel of 2x2 and stride of 2
			x = self.final_conv(x)

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
			self.skip_convs = nn.ModuleList()
			self.downsample_convs = nn.ModuleList()

			channel_size = 64
			double_channel_size = channel_size * 2

			# Zeroth/Start block (64)
			self.start_block = ResNetStart(in_channels, channel_size)
			self.start_max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

			self.relu = nn.ReLU(inplace=True)

			skip_conv = nn.Conv2d(channel_size, 128, 1, stride = 1)
			self.skip_convs.append(skip_conv)

			# First block list (64)
			res_block = nn.ModuleList()
			res_block.append(ResNetUnit(channel_size, channel_size))
			res_block.append(ResNetUnit(channel_size, channel_size))
			res_block.append(ResNetUnit(channel_size, channel_size))
			self.res_blocks.append(res_block)

			skip_conv = nn.Conv2d(channel_size, 128, 1, stride = 1)
			self.skip_convs.append(skip_conv)

			downsample_conv = nn.Conv2d(channel_size, double_channel_size, 1, stride = 2)
			self.downsample_convs.append(downsample_conv)

			# Second block list (128)
			res_block = nn.ModuleList()
			res_block.append(ResNetUnit(channel_size, double_channel_size, stride=2))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			self.res_blocks.append(res_block)

			channel_size = double_channel_size
			double_channel_size = channel_size * 2

			skip_conv = nn.Conv2d(channel_size, 128, 1, stride = 1)
			self.skip_convs.append(skip_conv)

			downsample_conv = nn.Conv2d(channel_size, double_channel_size, 1, stride = 2)
			self.downsample_convs.append(downsample_conv)

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

			skip_conv = nn.Conv2d(channel_size, 128, 1, stride = 1)
			self.skip_convs.append(skip_conv)

			downsample_conv = nn.Conv2d(channel_size, double_channel_size, 1, stride = 2)
			self.downsample_convs.append(downsample_conv)

			# Fourth block list (512)
			res_block = nn.ModuleList()
			res_block.append(ResNetUnit(channel_size, double_channel_size, stride=2))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			res_block.append(ResNetUnit(double_channel_size, double_channel_size))
			self.res_blocks.append(res_block)


			# Mid edge at the bottom of UNet
			self.mid_conv = nn.Conv2d(512, 512, 1, stride = 1)


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
				skip_connection = self.skip_convs[block_index](x)
				skip_connections.append(skip_connection)

				if block_index == 0:
					x = self.start_max_pool(x)

				for unit_index, res_unit in enumerate(res_block):
					downsampling_needed = block_index > 0 and unit_index == 0

					if downsampling_needed:
						identity = self.downsample_convs[block_index - 1](x) # zeroth block has max pool instead of conv
					else:
						identity = x

					x = res_unit(x)
					x += identity
					x = self.relu(x)

			# Mid edge at the bottom of UNet
			x = self.mid_conv(x)

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

			model.train(True)

			for data, target in train_loader:
				optimizer.zero_grad()

				input = Variable(data.type(torch.FloatTensor)).to(device)
				target = Variable(target.type(torch.FloatTensor)).to(device)

				output = model(input)

				output = torch.clamp(torch.round(output.contiguous().squeeze(1)), min=0, max=1)
				target = torch.clamp(target.contiguous(), min=0, max=1)

				loss = criterion(output, target)


				loss.backward()

				if count % 10 == 0:
					print("Epoch: " + str(epoch) + " Train: " + str(count))
					print(f'Current learning rate: {optimizer.param_groups[0]["lr"]}')
					print(f'Loss for batch {str(count)}: {str(loss.item())}')
					# for param in model.parameters():
					# 	print(float(param.grad.data.sum()))

				if count % 50 == 0:
					plt.imshow(TF.to_pil_image(output.to('cpu')))
					plt.show()
				# make_dot(output.mean(), params=dict(model.named_parameters())).render("rnn_torchviz", format="png")

				optimizer.step()
				schedular.step()
				count += 1
			
			count = 0
			run_loss = 0.0
			run_total = 0
			run_corrects = 0
			run_true_positive = 0
			run_false_positive = 0
			run_true_negative = 0
			run_false_negative = 0

			model.eval()

			for data, target in test_loader:
				print("Epoch: " + str(epoch) + " Test: " + str(count))

				input = Variable(data.type(torch.FloatTensor)).to(device)
				target = Variable(target.type(torch.FloatTensor)).to(device)

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


			precision = run_true_positive.true_divide(run_true_positive + run_false_positive)
			recall = run_true_positive.true_divide(run_true_positive + run_false_negative)
			f1_score = (2 * precision * recall).true_divide(precision + recall)
			epoch_acc = run_corrects.true_divide(run_total)

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

		in_channels = 1
		model = UNet(in_channels=in_channels, out_channels=1).to(device)
		if torch.cuda.is_available():
			model = model.cuda()

		epochs = 10
		batch_size = 2
		criterion = diceCoefficientLoss().to(device)
		# criterion = nn.BCEWithLogitsLoss()
		# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
		# optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
		optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
		# schedular = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
		schedular = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda = lambda epoch: 0.999)
		
		train_set = MnistDataset('data/MNIST/segmentation/train/original/', 'data/MNIST/segmentation/train/processed/', batch_size=batch_size)
		train_loader = train_set.get_loader()

		test_set = MnistDataset('data/MNIST/segmentation/test/original/', 'data/MNIST/segmentation/test/processed/',  batch_size=batch_size)
		test_loader = test_set.get_loader()

		#print(summary(self, input_size=(256, 1, 28, 28), device='cpu', verbose=2))

		best_model = train(model, train_loader, test_loader, epochs, criterion, optimizer, schedular)


except Exception as e:
	if post_mortem:
		Console().print_exception()
		pdb.post_mortem() # Post mortem debugging
	else:
		raise e
