import numpy as np
import matplotlib.pyplot as plt
def step_function(x):
    if x>0:
        return 1
    else:
        return 0


def step_function2(x):
    return np.array(x>0, dtype=np.int)

x = np.array([-1.0, 1.0, 2.0])
y = x > 0
y = y.astype(np.int)
print(y)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function2(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

def sigmoid(x):
    return 1/(1 + np.exp(-x))

x= np.array([-1.0, 1.0, 2.0])

print(sigmoid(x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

def relu(x):
    return np.maximum(0,x)
