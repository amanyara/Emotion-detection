import paddle
import paddle.nn as nn
import numpy as np
from PIL import Image


class Net(nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2D(
                in_channels=1,
                out_channels=10,
                kernel_size=3,
                stride=1
            ),
            nn.MaxPool2D(
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2D(
                in_channels=10,
                out_channels=20,
                kernel_size=3,
                stride=1
            ),
            nn.MaxPool2D(
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2000, out_features=4)
        )
    def forward(self, *inputs, **kwargs):
        x = inputs
        x = self.net(x)
        return x

Mynet = Net()
para_dicr = paddle.load('mnist.pdparams')
Mynet.load_dict(para_dicr)
Mynet.eval()
img = Image.open('../Data/anger/anger_2.jpg')
img = np.array(img)
img = img.reshape(1, 1, 48, 48)
img = paddle.to_tensor(img).astype('float32')


result = Mynet(img)
result = paddle.nn.functional.softmax(result)
express = ['anger', 'happy', 'normal', 'surprised']
result = np.array(result)


print('The predict result is {}'.format(express[int(np.argmax(result))]))




