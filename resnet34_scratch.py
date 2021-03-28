import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np

class ResNetStart(nn.Module):
	def __init__(self, in_channels, out_channels, stride=2):
		super(ResNetStart, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=(7, 7), stride=stride, padding = 3),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)
	
	def forward(self, x):
		return self.conv(x)

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

		#Concurrent Spatial, Channel Squeeze and Excitation layer
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

		x = View(linear_shape)(x)
		x = nn.Linear(linear_in_channels, linear_in_channels // self.reduction)(x)
		x = nn.ReLU(inplace=True)(x)
		x = nn.Linear(linear_in_channels // self.reduction, linear_in_channels)(x)
		x = nn.Sigmoid()(x)
		x = View(tensor_shape)(x)

		x = x_before_squeeze * x

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
					in_channels = x.shape[2]
					out_channels = in_channels * 2
					identity = nn.Conv2d(in_channels, out_channels, 1, stride = 2)
				else:
					identity = x

				x = res_block(x)
				x += identity
				x = nn.ReLU(x, inplace=True)

		return self.end_block(x)


class UNet(nn.Module):
	def __init__(self, in_channels=3, out_channels=1):
		super(UNet, self).__init__()

		# self.start_block = nn.ModuleList()
		self.res_blocks = nn.ModuleList()
		self.ups = nn.ModuleList()

		channel_size = 64
		double_channel_size = channel_size * 2

		# Start block (64)
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

		# Final convolution to obtain output
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

if __name__ == "__main__":
	test()
