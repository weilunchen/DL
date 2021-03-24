import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchsummary import summary

class ResNetStart(nn.Module):
	def __init__(self, in_channels, out_channels, stride=2):
		super(ResNetStart, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=(7, 7), stride=stride, padding=3),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)
	
	def forward(self, x):
		return self.conv(x)

class ResNetUnit(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1):
		super(ResNetUnit, self).__init__()
		self.conv = nn.Sequential(
			nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True),
				nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
				nn.BatchNorm2d(out_channels),
			)
		)
	
	def forward(self, x):
		return self.conv(x)

class UpSampleUnit(nn.Module):
	def __init__(self, in_channels, out_channels, reduction=2):
		super(UpSampleUnit, self).__init__()
		self.conv = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(in_channels),
			#Concurrent Spatial, Channel Squeeze and Excitation layer
			nn.Linear(in_channels, in_channels // reduction),
			nn.ReLU(inplace=True),
			nn.Linear(in_channels // reduction, in_channels),
			nn.Sigmoid(),
			#Transpose Conv with kernel of 2x2 and stride of 2
			nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2, bias=False),
		)
	
	def forward(self, x):
		return self.conv(x)

class UNet(nn.Module):
	def __init__(self, in_channels=3, out_channels=1):
		super(UNet, self).__init__()
		
		self.ups = nn.ModuleList()
		self.downs = nn.ModuleList()
		
		channel_size = 64
		double_channel_size = channel_size * 2
		
		#self.start_block = nn.Sequential(
		#	nn.Conv2d(in_channels, channel_size, kernel_size=(7, 7), stride=2, padding=3),
		#	nn.BatchNorm2d(channel_size),
		#	nn.ReLU(inplace=True),
		#)
		
		self.start_block = ResNetStart(in_channels, channel_size)
		
		#First block list (64)
		res_block = []
		res_block.append(ResNetUnit(channel_size, channel_size))
		res_block.append(ResNetUnit(channel_size, channel_size))
		res_block.append(ResNetUnit(channel_size, channel_size))
		self.downs.append(nn.Sequential(*res_block))
		
		#Second block list (128)
		res_block = []
		res_block.append(ResNetUnit(channel_size, double_channel_size, stride=2))
		res_block.append(ResNetUnit(double_channel_size, double_channel_size))
		res_block.append(ResNetUnit(double_channel_size, double_channel_size))
		res_block.append(ResNetUnit(double_channel_size, double_channel_size))
		self.downs.append(nn.Sequential(*res_block))
		
		channel_size = double_channel_size
		double_channel_size = channel_size * 2
		
		#Third block list (256)
		res_block = []
		res_block.append(ResNetUnit(channel_size, double_channel_size, stride=2))
		res_block.append(ResNetUnit(double_channel_size, double_channel_size))
		res_block.append(ResNetUnit(double_channel_size, double_channel_size))
		res_block.append(ResNetUnit(double_channel_size, double_channel_size))
		res_block.append(ResNetUnit(double_channel_size, double_channel_size))
		res_block.append(ResNetUnit(double_channel_size, double_channel_size))
		self.downs.append(nn.Sequential(*res_block))
		
		channel_size = double_channel_size
		double_channel_size = channel_size * 2
		
		#Fourth block list (512)
		res_block = []
		res_block.append(ResNetUnit(channel_size, double_channel_size, stride=2))
		res_block.append(ResNetUnit(double_channel_size, double_channel_size))
		res_block.append(ResNetUnit(double_channel_size, double_channel_size))
		self.downs.append(nn.Sequential(*res_block))
		
		#Bottom connection
		bottom_block = []
		bottom_block.append(nn.Conv2d(double_channel_size, double_channel_size, kernel_size=(1, 1)))
		bottom_block.append(nn.ConvTranspose2d(double_channel_size, 128, kernel_size=(2, 2), stride=2, bias=False))
		self.bottom_block = nn.Sequential(*bottom_block)
		
		for i in range(0, 3):
			self.ups.append(UpSampleUnit(256, 128))
		
		self.final_conv = UpSampleUnit(256, 1)
		
	def forward(self, x):
		# UNet skip connections
		skip_connections = []

		x = self.start_block(x)

		for block_index, res_block in enumerate(self.downs):
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

			skip_in_channels = x.shape[1]
			skip_out_channels = 128
			skip_connection = nn.Conv2d(skip_in_channels, skip_out_channels, 1, stride = 1)
			skip_connections.append(skip_connection)

		x = self.bottom(x)

		skip_connections = skip_connections[::-1]

		for i in range(0, len(skip_connections)):
			concat_skip_connection = torch.cat((skip_connections[i], x), dim=1)
			x = self.ups[i](concat_skip_connection)

		return self.final_conv(x)



def get_summary():
	model = UNet(3, 1)
	summary(model, (3, 64, 64))

def test():
	x = torch.randn((3, 2, 320, 320))
	model = UNet(in_channels=3, out_channels=1)
	preds = model(x)
	print(preds.shap)
	print(x.shape)

if __name__ == "__main__":
	get_summary()