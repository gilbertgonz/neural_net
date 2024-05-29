import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

import argparse
import matplotlib.pyplot as plt
import numpy as np

class CNN(nn.Module):
    '''
    Convolutional neural network (two convolutional layers)
    '''
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes) # output layer

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def imshow(img):
    '''
    Show image(s)
    '''
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_results(model, device, test_loader, len_of_test=9):
    '''
    Show results and data
    '''
    print("\nHere are some results:")

    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    random_list = np.random.randint(0, len(images), size=len_of_test) # choose random imgs from dataset

    images = images[random_list]
    labels = labels[random_list]

    outputs = model(images.to(device))

    _, predicted = torch.max(outputs, 1)

    print(f'Predicted   : {predicted}')

    print(f'Ground truth: {labels}')

    imshow(torchvision.utils.make_grid(images))

def train(args, model, device, train_loader):
    '''
    Train model
    '''
    print("\nTraining...")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training sequence
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0: # print every 100 steps
                print (f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}')

def test(model, device, test_loader):
    '''
    Test model
    '''
    print("\nTesting...")

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of model on 10000 test images: {} %'.format(100 * correct / total))


def main():
    # Parameter args
    parser = argparse.ArgumentParser(description='Learning MNIST dataset')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='Save model')
    args = parser.parse_args()

    print(f"Welcome! Let's get started.")

    # Device config
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   

    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data/',
                                            train=True, 
                                            transform=transforms.ToTensor(),
                                            download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size, 
                                            shuffle=True)

    test_dataset = torchvision.datasets.MNIST(root='./data/',
                                            train=False, 
                                            transform=transforms.ToTensor())    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size, 
                                            shuffle=False)
        
    # Init model
    num_classes = len(set(train_dataset.targets.numpy()))
    model = CNN(num_classes).to(device)

    # Train
    train(args, model, device, train_loader)

    # Test
    test(model, device, test_loader)

    # Save model
    if args.save_model:
        torch.save(model, 'model.pt')

    # Show results
    show_results(model, device, test_loader)

if __name__ == '__main__':
    main()