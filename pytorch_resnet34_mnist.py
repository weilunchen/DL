import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torch import nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

#Torch version 1.8.0
#Torchvision version 0.9.0

#Load ResNet 34 from pytorch
#model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
model = torchvision.models.resnet34(pretrained=False)
model.load_state_dict(torch.load('resnet34-333f7ec4.pth'))
#model.avgpool = Identity()
model.fc = nn.Linear(512, 10)
model.eval()

#Transform the images usable for ResNet34
# All pre-trained models expect input images normalized in the same way, i.e. mini-batches 
# of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. 
# The images have to be loaded in to a range of [0, 1] and then normalized using 
# mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([
        transforms.Grayscale(3),
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device('cpu')
#Load MNIST data set from torchvision
#MNIST contains 70.000 images = 60.000 train + 10.000 test
#Images are all 28 x 28 pixels
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=preprocess)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=preprocess)


# Check whether GPU processing is enabled
if torch.cuda.is_available():
    print('CUDA GPU available')
    device = torch.device('cuda')
    model.to('cuda')
else:
    print('CUDA GPU not available')

#input_batch = mnist_train.unsqueeze(0)
print("Begin eval")

batch_counter = 0
for x in enumerate (mnist_train):
    if batch_counter % 500 == 0:
        print(f'Started with input batch {batch_counter + 1}')
    input_batch = x[1][0].unsqueeze(0)
    input_batch = input_batch.to(device=device)
    output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    #print(probabilities)

    batch_counter += 1


# Feed the input into the model
#output = model(input_batch)

print("Eval done")

#print(output[0])
#probabilities = torch.nn.functional.softmax(output[0], dim=0)
#print(probabilities)
