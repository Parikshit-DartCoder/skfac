import torch
import torch.nn as nn
import numpy as np

class Conv2d_SKFAC_GPU(nn.Module):
    """Conv2d_SKFAC"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 damping=0.03,
                 loss_scale=1,
                 frequency=278,
                 batch_size=32):
        super(Conv2d_SKFAC_GPU, self).__init__()
        self.skfac = True
        self.hw = kernel_size * kernel_size
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)
        self.conv2d.reset_parameters()

        self.matrix_A_dim = in_channels * kernel_size * kernel_size
        self.matrix_G_dim = out_channels
        self.matrix_A_inv = nn.Parameter(torch.zeros((self.matrix_A_dim, self.matrix_A_dim), dtype=torch.float32),
                                         requires_grad=False)
        self.matrix_G_inv = nn.Parameter(torch.zeros((self.matrix_G_dim, self.matrix_G_dim), dtype=torch.float32),
                                         requires_grad=False)

        self.cov_step = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.loss_scale = 1 / loss_scale
        self.batch_size = batch_size
        self.damping = nn.Parameter(torch.tensor(damping), requires_grad=False)
        self.dampingA = torch.eye(batch_size, dtype=torch.float32)
        self.dampingG = torch.eye(batch_size, dtype=torch.float32)
        self.I_G = torch.eye(out_channels, dtype=torch.float32)
        self.I_A = torch.eye(self.matrix_A_dim, dtype=torch.float32)
        self.freq = frequency
        self.sqrt = nn.functional.sqrt
        self.cholesky = nn.functional.cholesky

    def save_gradient(self, dout):
        """save_gradient"""
        out = dout
        dout = dout * self.batch_size
        dout = dout.permute(1, 0, 2, 3)  # [out_channels, batch_size, sp, sp]
        dout = dout.view(out.shape[1], out.shape[0], -1)  # [out_channels, batch_size, sp*sp]
        dout = dout.mean(dim=2)  # [out_channels, batch_size]
        dout = dout.float()
        dout = dout * (1 / 32) ** 0.5

        damping_step = self.damping[self.cov_step]
        damping = self.sqrt(damping_step)
        dout_cov = torch.matmul(dout.t(), dout)
        damping_G = self.dampingG * damping
        dout_cov = dout_cov + damping_G
        dout_cov_inv = self.cholesky(dout_cov)
        dout_cov_inv = torch.matmul(dout_cov_inv, dout_cov_inv.t())
        dout_cov_inv = (self.I_G - dout.matmul(dout_cov_inv).matmul(dout.t())) * (1 / damping)
        self.matrix_G_inv = dout_cov_inv
        return out

    def forward(self, x):
        if self.skfac:
            matrix_A = nn.functional.im2col(x, self.conv2d.kernel_size, padding=self.conv2d.padding,
                                             stride=self.conv2d.stride)
            matrix_A_shape = matrix_A.shape
            matrix_A = matrix_A.view(matrix_A_shape[0] * matrix_A_shape[1] * matrix_A_shape[2],
                                     matrix_A_shape[3], -1)
            matrix_A = matrix_A.mean(dim=2)
            damping_step = self.damping[self.cov_step]
            damping = self.sqrt(damping_step)
            matrix_A_cov = matrix_A.t().matmul(matrix_A)
            damping_A = self.dampingA * damping
            matrix_A_cov = matrix_A_cov + damping_A
            matrix_A_cov_inv = self.cholesky(matrix_A_cov)
            matrix_A_cov_inv = matrix_A_cov_inv.matmul(matrix_A_cov_inv.t())
            matrix_A_cov_inv = (self.I_A - matrix_A.matmul(matrix_A_cov_inv).matmul(matrix_A.t())) * (1 / damping)
            self.matrix_A_inv = matrix_A_cov_inv
            output = self.conv2d(x)
            output = self.save_gradient(output)
        else:
            output = self.conv2d(x)

        return output


class Dense_SKFAC_GPU(nn.Module):
    """Dense_SKFAC"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 damping=0.03,
                 loss_scale=1,
                 frequency=278,
                 batch_size=32,
                 has_bias=True,
                 activation=None):
        super(Dense_SKFAC_GPU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_bias = has_bias
        self.skfac = True

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.normal_(self.weight)

        if self.has_bias:
            self.bias = nn.Parameter(torch.empty
