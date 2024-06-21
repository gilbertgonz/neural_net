# neural_net
Implementing simple neural networks with PyTorch to train on MNIST dataset

## How to run:
1. Install [docker](https://docs.docker.com/engine/install/)

2. Clone repo

3. Build:
```
$ docker build -t neural_net .
```

4. Run:
```
$ xhost +local:docker
$ docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix neural_net
```

## Expected output:
```
Welcome! Let's get started.

Training...
Epoch [1/2], Step [100/600], Loss: 0.8929394483566284
Epoch [1/2], Step [200/600], Loss: 0.5471228957176208
Epoch [1/2], Step [300/600], Loss: 0.6402640342712402
Epoch [1/2], Step [400/600], Loss: 0.5361933708190918
Epoch [1/2], Step [500/600], Loss: 0.6597360372543335
Epoch [1/2], Step [600/600], Loss: 0.48591378331184387
Epoch [2/2], Step [100/600], Loss: 0.3047449588775635
Epoch [2/2], Step [200/600], Loss: 0.5093909502029419
Epoch [2/2], Step [300/600], Loss: 0.5683649778366089
Epoch [2/2], Step [400/600], Loss: 0.2906494438648224
Epoch [2/2], Step [500/600], Loss: 0.2594521641731262
Epoch [2/2], Step [600/600], Loss: 0.2539266049861908

Testing...
Accuracy of model on 10000 test images: 95.21 %

Here are some results:
Predicted   : tensor([7, 3, 6, 7, 6, 3, 1, 0, 4])
Ground truth: tensor([7, 3, 6, 7, 6, 3, 1, 0, 4])
```