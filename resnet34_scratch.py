import torch
import torch.nn
import torchvision.transforms.functional.nn as TF

class ResNetBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1):
		super(ResNetBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
			nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
			nn.ReLU()
		)

	def forward(x):
		return self.conv(x)


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
		res_block.append(ResNetBlock(channel_size, channel_size))
		res_block.append(ResNetBlock(channel_size, channel_size))
		res_block.append(ResNetBlock(channel_size, channel_size))
		self.res_blocks.append(res_block)

		# Second block list (128)
		res_block = nn.ModuleList()
		res_block.append(ResNetBlock(channel_size, double_channel_size, stride=2))
		res_block.append(ResNetBlock(double_channel_size, double_channel_size))
		res_block.append(ResNetBlock(double_channel_size, double_channel_size))
		res_block.append(ResNetBlock(double_channel_size, double_channel_size))
		self.res_blocks.append(res_block)

		channel_size = double_channel_size
		double_channel_size = channel_size * 2

		# Third block list (256)
		res_block = nn.ModuleList()
		res_block.append(ResNetBlock(channel_size, double_channel_size, stride=2))
		res_block.append(ResNetBlock(double_channel_size, double_channel_size))
		res_block.append(ResNetBlock(double_channel_size, double_channel_size))
		res_block.append(ResNetBlock(double_channel_size, double_channel_size))
		res_block.append(ResNetBlock(double_channel_size, double_channel_size))
		res_block.append(ResNetBlock(double_channel_size, double_channel_size))
		self.res_blocks.append(res_block)

		channel_size = double_channel_size
		double_channel_size = channel_size * 2

		# Fourth block list (512)
		res_block = nn.ModuleList()
		res_block.append(ResNetBlock(channel_size, double_channel_size, stride=2))
		res_block.append(ResNetBlock(double_channel_size, double_channel_size))
		res_block.append(ResNetBlock(double_channel_size, double_channel_size))
		self.res_blocks.append(res_block)

		# End block (1000)
		self.start_block.append(nn.AvgPool2d(kernel_size=1, stride=1))
		self.end_block = nn.Linear(double_channel_size, 1000)

	def forward(self, x):
		pass