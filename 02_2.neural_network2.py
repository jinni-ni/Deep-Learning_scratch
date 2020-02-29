import numpy as np

# 다차원 배열의 게선
A = np.array([1,2,3,4])
print(A)
# 차원의 수 확
print(np.ndim(A))
print(A.shape)
print(A.shape[0])

B = np.array([[1,2],[3,4],[5,6]])
print(B)
print('차원:' + str(np.ndim(B)))
print(B.shape)

# 행렬의 곱셈
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[1,2],[3,4],[5,6]])

print(np.dot(A,B))
print(np.dot(B,A))

# 신경망에서 행렬의 곱
print("\n신경망 행렬 곱")

X = np.array([1,2])
print(X.shape)

W = np.array([[1,3,5],[2,4,6]])
print(W.shape)
Y = np.dot(X,W)

print(Y)
