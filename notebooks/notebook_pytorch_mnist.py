"""
Module: notebook_pytorch_mnist
Stage: Notebook
Author: Angus9312
Date: 2026-07-15
Description: train a CNN using pytorch
"""

# %%

import ssl

import torchvision as tv
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
