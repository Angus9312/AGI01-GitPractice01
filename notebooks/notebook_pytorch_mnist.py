"""
Module: notebook_pytorch_mnist
Stage: Notebook
Author: Angus9312
Date: 2026-07-15
Description: train a CNN using pytorch
"""

# %%

import ssl

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from IPython.display import clear_output
from matplotlib import pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

train_data = tv.datasets.MNIST(
    "../dataset/MNIST/",
    train=True,
    transform=None,
    target_transform=None,
    download=True,
)  # 下載並匯入MNIST訓練資料
test_data = tv.datasets.MNIST(
    "../dataset/MNIST/",
    train=False,
    transform=None,
    target_transform=None,
    download=True,
)  # 下載並匯入MNIST測試資料

print("Number of samples in train_data is: ", len(train_data))
print("Number of samples in test_data is: ", len(test_data))
# %%

x = train_data.data[0]  # 讀取訓練集中的第一張圖片
plt.imshow(x)  # 把圖片顯示出來


# %%
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3))  # 第一卷積層
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))  # 最大池化層
        self.lin1 = nn.Linear(800, 128)
        self.out = nn.Linear(
            128, 10
        )  # 模型的最後一層為線性層，用來處理CNN扁平化後的輸出

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)  # 選用ReLU為激活函數
        x = self.maxpool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)
        x = x.flatten(start_dim=1)  # 扁平化
        x = self.lin1(x)
        x = nn.functional.relu(x)
        x = self.out(x)
        x = nn.functional.log_softmax(
            x, dim=1
        )  # 使用log_softmax( )，以機率的形式進行分類
        return x


# %%
def prepare_images(xt):  # 對圖片進行預處理，以符合PyTorch的格式需求
    out = torch.zeros(xt.shape)
    for i in range(xt.shape[0]):
        img = xt[i].unsqueeze(dim=0)  # 加入批次軸
        out[i] = img
    return out


a = train_data.data[0:5].detach()  # 讀取訓練集中的前五張圖片
b = prepare_images(a)  # 對圖片進行預處理
print(a.shape)
print(b.shape)  # 可以看到，經過prepare_images( )後，圖片的形狀已經符合PyTorch的格式需求


# %%
model = CNN()  # 建立CNN模組的實例
epochs = 100  # 定義訓練迴圈
batch_size = 500  # 定義批次大小
lr = 1e-3
opt = torch.optim.Adam(params=model.parameters(), lr=lr)
lossfn = nn.NLLLoss()
# %%
losses = []
acc_cnn = []

for i in range(epochs):
    clear_output(wait=True)
    print("Current training epoch: ", i)  # 顯示當前訓練進度
    opt.zero_grad()
    batch_ids_cnn = np.random.randint(
        0, 60000, size=batch_size
    )  # 隨機從訓練集中選取樣本，組成訓練批次
    xt = train_data.data[batch_ids_cnn].detach()  # 產生訓練批次
    xt = prepare_images(xt).unsqueeze(dim=1)  # 加入通道軸
    yt = train_data.train_labels[batch_ids_cnn].detach()  # 取得正確的標籤資訊
    pred = model(xt)
    pred_labels = torch.argmax(
        pred, dim=1
    )  # 找出擁有最大機率值的類別，即為模型的預測結果
    acc_ = 100.0 * (pred_labels == yt).sum() / batch_size  # 計算預測準確率
    print("Current training accuracy: ", acc_.item())  # 顯示該訓練迴圈的預測準確率
    acc_cnn.append(acc_)
    loss = lossfn(pred, yt)  # 計算損失
    print("Current training loss: ", loss.item())  # 顯示該訓練迴圈的損失
    losses.append(loss.item())
    loss.backward()
    opt.step()

acc_cnn = np.array(acc_cnn)  # 將每一訓練迴圈的預測準確率存進陣列，以便稍後進行視覺化
losses = np.array(losses)  # 將每一訓練迴圈的損失存進陣列，以便稍後進行視覺化
# %%
plt.figure(figsize=(10, 7))
plt.xlabel("Training Epochs", fontsize=16)
plt.ylabel("Training accuracy", fontsize=16)
plt.plot(acc_cnn)
# %%
