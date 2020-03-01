import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist

(x_train, y_train), (x_test, y_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(y_train.shape)

# 무작위로 10 장을 빼냄 미니배치
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
y_batch = y_train[batch_mask]

print(np.random.choice(6000,10))
