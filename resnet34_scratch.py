import torch
import torch.nn
import torchvision.transforms.functional.nn as TF

class ResNetUnit(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1):
		super(ResNetUnit, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(x):
		return self.conv(x)

class UpSampleBlock(nn.Module):
	def __init__(self, in_channels=256, out_channels=128, reduction=2):
		super(UpSampleBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(in_channels),
			#Concurrent Spatial, Channel Squeeze and Excitation layer
			nn.Linear(in_channels, in_channels // reduction),
			nn.ReLU(inplace=True),
			nn.Linear(in_channels // reduction, in_channels),
			nn.Sigmoid(),
			#Transpose Conv with kernel of 2x2 and stride of 2
			nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2, bias=False)
		)
	
	def forward(self, x):
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
		self.start_block.append(nn.AvgPool2d(kernel_size=1, stride=1))
		self.end_block = nn.Linear(double_channel_size, 1000)

	def forward(self, x):
		x = self.start_block(x)

		for block_index, res_block in enumerate(self.res_blocks):
			for unit_index, res_unit in enumerate(res_block):
				downsampling_needed = block_index > 0 && unit_index == 0
				
				if downsampling_needed:
					in_channels = x.shape[2]
					out_channels = in_channels * 2
					identity = nn.Conv2d(in_channels, out_channels, 3, stride = 2, padding = 1)
				else
					identity = x

				x = res_block(x)
				x += identity
				x = nn.ReLU(x, inplace=True)

		return self.end_block(x)


class UNet(nn.Module):
	def __init__(self, in_channels=3, out_channels=1):
		super(UNet, self).__init__()

		self.start_block = nn.ModuleList()
		self.res_blocks = nn.ModuleList()
		self.ups = nn.ModuleList()

		channel_size = 64
		double_channel_size = channel_size * 2

		# Start block (64)
		self.start_block.append(nn.Conv2d(in_channels, channel_size, 3, kernel_size=(7, 7), stride=2, padding=1))
		self.start_block.append(nn.BatchNorm2d(64, eps=1e-05, momemtum=0.1, affine=True, track_running_stats=True))
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

		self.bottom = nn.Conv2d(double_channel_size, double_channel_size).append(
			nn.ConvTranspose2d(double_channel_size, 128, kernel_size=(2, 2), stride=2, bias=False)
			)

		# Right part of the UNet (the up part)
		for res_block in self.res_blocks:
			self.ups.append(nn.Conv2d(res_block.out_channels, 128))
			self.ups.append(UpSampleBlock(in_channels, out_channels))

		self.final_conv = UpSampleBlock(in_channels, out_channels=1)


	def forward(self, x):
		# UNet skip connections
		skip_connections = []

		x = self.start_block(x)

		for block_index, res_block in enumerate(self.res_blocks):
			for unit_index, res_unit in enumerate(res_block):
				downsampling_needed = block_index > 0 && unit_index == 0

				if downsampling_needed:
					in_channels = x.shape[2]
					out_channels = in_channels * 2
					identity = nn.Conv2d(in_channels, out_channels, 3, stride = 2, padding = 1)
				else
					identity = x

				x = res_block(x)
				x += identity
				x = nn.ReLU(x, inplace=true)

			skip_connections.append(x)

		x = self.bottom(x)
		skip_connections = skip_connections[::-1]

		for i in range(0, len(self.ups), 2):
			x = self.ups[i](x)
			skip_connection = skip_connections[i//2]
			concat_skip_connection = torch.cat((skip_connection, x), dim=1)
			x = self.ups[i+1](concat_skip_connection) 

		return self.final_conv(x)
