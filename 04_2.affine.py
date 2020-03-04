import numpy as np

X = np.random.rand(2)
W = np.random.rand(2,3)
B = np.random.rand(3)

print(X.shape) #(2, )
print(W.shape) #(2,3)
print(B.shape) #(3, )

Y = np.dot(X,W) + B
