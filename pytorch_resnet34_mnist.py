import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms

#Torch version 1.8.0
#Torchvision version 0.9.0

#Load ResNet 34 from pytorch
#model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
model = torchvision.models.resnet34(pretrained=False)
model.load_state_dict(torch.load('resnet34-333f7ec4.pth'))
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

#Load MNIST data set from torchvision
#MNIST contains 70.000 images = 60.000 train + 10.000 test
#Images are all 28 x 28 pixels
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=preprocess)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=preprocess)

#input_batch = mnist_train.unsqueeze(0)
print("Begin eval")
for x in enumerate (mnist_train):
    input_batch = x[1][0].unsqueeze(0)
    output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)


# Feed the input into the model
#output = model(input_batch)

print("Eval done")

#print(output[0])
#probabilities = torch.nn.functional.softmax(output[0], dim=0)
#print(probabilities)
