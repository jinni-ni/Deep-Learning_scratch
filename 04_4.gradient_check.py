import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, y_train), (x_test, y_test) =  \
load_mnist(normalize=True,one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=50)

x_batch = x_train[:3]
y_batch = y_train[:3]

grad_numerical = network.numerical_gradient(x_batch, y_batch)
grad_backprop = network.gradient(x_batch,y_batch)

for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
