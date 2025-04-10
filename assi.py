import numpy as np
import time

class NumPyFreckersNet:
    def __init__(self, params):
        self.params = params

    def im2col(self, input, kernel_size):
        C, H, W = input.shape
        kh, kw = kernel_size
        H_out = H - kh + 1
        W_out = W - kw + 1
        strides = input.strides
        cols = np.lib.stride_tricks.as_strided(
            input,
            shape=(C, H_out, W_out, kh, kw),
            strides=(strides[0], strides[1], strides[2], strides[1], strides[2])
        )
        cols = cols.transpose(1, 2, 0, 3, 4).reshape(H_out * W_out, -1)
        return cols

    def conv2d(self, x, weight, bias, padding=1):
        C_in, H, W = x.shape
        kh, kw = weight.shape[2], weight.shape[3]
        x_padded = np.pad(x, [(0, 0), (padding, padding), (padding, padding)], mode='constant')
        cols = self.im2col(x_padded, (kh, kw))
        weight_flat = weight.reshape(weight.shape[0], -1)
        out = cols @ weight_flat.T  # 形状 (H_out*W_out, out_channels)
        out += bias.reshape(1, -1)  # 正确广播形状
        out = out.reshape(H, W, -1).transpose(2, 0, 1)
        return out

    def residual_block(self, x, conv1_w, conv1_b, conv2_w, conv2_b):
        residual = x.copy()
        x = self.conv2d(x, conv1_w, conv1_b)
        x = np.maximum(x, 0)
        x = self.conv2d(x, conv2_w, conv2_b)
        x += residual
        x = np.maximum(x, 0)
        return x

    def forward(self, x):
        # 公共特征提取
        x = self.conv2d(x, self.params['conv1_weight'], self.params['conv1_bias'])
        x = np.maximum(x, 0)
        
        # 残差块
        for i in [1, 2, 3]:
            conv1_w = self.params[f'res{i}_conv1_weight']
            conv1_b = self.params[f'res{i}_conv1_bias']
            conv2_w = self.params[f'res{i}_conv2_weight']
            conv2_b = self.params[f'res{i}_conv2_bias']
            x = self.residual_block(x, conv1_w, conv1_b, conv2_w, conv2_b)
        
        # 图像输出头
        img_out = self.conv2d(x, self.params['img_head_weight'], self.params['img_head_bias'])
        
        # 数值概率分支
        prob = self.conv2d(x, self.params['prob_conv_weight'], self.params['prob_conv_bias'])
        prob = np.maximum(prob, 0)
        prob_flat = prob.reshape(-1)
        
        fc1_out = np.dot(self.params['fc1_weight'], prob_flat) + self.params['fc1_bias']
        fc1_out = np.maximum(fc1_out, 0)
        
        fc2_out = np.dot(self.params['fc2_weight'], fc1_out) + self.params['fc2_bias']
        prob_out = 1 / (1 + np.exp(-fc2_out))
        
        return img_out, prob_out

# 生成虚拟参数（保持与PyTorch一致的形状）
def generate_dummy_params():
    params = {}
    
    # conv1参数 [out_channels, in_channels, h, w]
    params['conv1_weight'] = np.random.randn(128, 16, 3, 3).astype(np.float32)
    params['conv1_bias'] = np.random.randn(128).astype(np.float32)
    
    # 残差块参数 
    for i in [1, 2, 3]:
        params[f'res{i}_conv1_weight'] = np.random.randn(128, 128, 3, 3).astype(np.float32)
        params[f'res{i}_conv1_bias'] = np.random.randn(128).astype(np.float32)
        params[f'res{i}_conv2_weight'] = np.random.randn(128, 128, 3, 3).astype(np.float32)
        params[f'res{i}_conv2_bias'] = np.random.randn(128).astype(np.float32)
    
    # 图像输出头
    params['img_head_weight'] = np.random.randn(65, 128, 3, 3).astype(np.float32)
    params['img_head_bias'] = np.random.randn(65).astype(np.float32)
    
    # 概率分支参数
    params['prob_conv_weight'] = np.random.randn(5, 128, 3, 3).astype(np.float32)
    params['prob_conv_bias'] = np.random.randn(5).astype(np.float32)
    params['fc1_weight'] = np.random.randn(16, 5*8*8).astype(np.float32)
    params['fc1_bias'] = np.random.randn(16).astype(np.float32)
    params['fc2_weight'] = np.random.randn(1, 16).astype(np.float32)
    params['fc2_bias'] = np.random.randn(1).astype(np.float32)
    
    return params

# 测试代码
params = generate_dummy_params()
model = NumPyFreckersNet(params)
x = np.random.randn(16, 8, 8).astype(np.float32)

# 预热运行
for _ in range(10):
    model.forward(x)

# 正式计时
num_runs = 6000
start = time.time()
for _ in range(num_runs):
    img, prob = model.forward(x)
total_time = time.time() - start

print(f"总推理次数: {num_runs}")
print(f"总耗时: {total_time:.4f}秒")
print(f"平均单次耗时: {total_time/num_runs*1000:.2f}毫秒")
print(f"FPS: {num_runs/total_time:.2f}次/秒")