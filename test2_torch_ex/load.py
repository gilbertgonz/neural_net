import torch 
from torchvision.transforms import ToTensor
import cv2

from main import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model
model = Net()
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.to(device)
model.eval()

# Show sample prediction
img = cv2.imread('img_1.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("test", img) 

img_tensor = ToTensor()(img).unsqueeze(0).to(device) # convert img to tensor

print(torch.argmax(model(img_tensor))) # print prediction

cv2.waitKey(0)