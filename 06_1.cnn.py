import  numpy as np

# 10 : 데이터 개수
# 1 : 채널 수
# 28,28 : 높어 너비
x = np.random.rand(10,1,28,28)
print(x.shape)

print(x[0].shape)
print(x[1].shape)

print(x[0][0])

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW ) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshaple(FN, -1).T
        out = np.dot(col,col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)

        return out
