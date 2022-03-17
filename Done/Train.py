import os
import numpy as np
from PIL import Image
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import matplotlib.pyplot as plt
#2304

datafile = '../Data/'
datas = list()
labels = list()
BATCH_SIZE = 20
EPOCH_NUM = 5
errorlist = list()

for i, file in enumerate(os.listdir(datafile)):
    img_path = datafile + file
    for path in os.listdir(img_path):
        labels.append(i)
        path = img_path + '/' + path
        img = Image.open(path)
        img = np.array(img)
        img = img.reshape(1, 48, 48)
        #img = img.reshape(-1)
        #img = list(img)
        #img.append(i)
        #data.append(img)
        datas.append(img)

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


class MyDataset(paddle.io.Dataset):
    def __init__(self, num_samples, datas, labels):
        self.num_samples = num_samples
        self.datas = datas
        self.labels = labels

    def __getitem__(self, idx):

        image = self.datas[idx].astype('float32')
        label = np.array(self.labels[idx]).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples



dataset = MyDataset(len(datas), datas, labels)
model = Net()
model.train()
print(dataset.__len__())

opt = paddle.optimizer.SGD(learning_rate=1e-3, parameters=model.parameters())

loader = paddle.io.DataLoader(dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=True,
                    num_workers=200)

for epoch_id in range(EPOCH_NUM):
    for batch_id, data in enumerate(loader()):
        # 准备数据，变得更加简洁
        images, labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)
        #print(images.shape)
        # 前向计算的过程
        predicts = model(images)

        # 计算损失，取一个批次样本损失的平均值
        loss = F.cross_entropy(predicts, labels)
        avg_loss = paddle.mean(loss)
        error = np.array(avg_loss)
        errorlist.append(error)
        # 每训练了100批次的数据，打印下当前Loss的情况
        if batch_id % 200 == 0:
            print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))

        # 后向传播，更新参数的过程
        avg_loss.backward()
        opt.step()
        opt.clear_grad()





paddle.save(model.state_dict(), 'mnist.pdparams')
