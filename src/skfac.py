import torch
from torch.optim import Optimizer

class SKFAC_GPU(Optimizer):
    """
    SKFAC optimizer for GPU.
    """

    def __init__(self, params, learning_rate, momentum, matrix_A, matrix_G, A_inv_max, G_inv_max,
                 weight_decay=0.0, use_nesterov=False, decay_filter=lambda x: x.name not in []):
        defaults = dict(learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)
        super(SKFAC_GPU, self).__init__(params, defaults)

        self.matrix_A = matrix_A
        self.matrix_G = matrix_G
        self.A_inv_max = A_inv_max
        self.G_inv_max = G_inv_max
        self.use_nesterov = use_nesterov
        self.decay_filter = decay_filter

    def __setstate__(self, state):
        super(SKFAC_GPU, self).__setstate__(state)

    def step(self, gradients):
        success = True
        lr = self.defaults['learning_rate']
        momentum = self.defaults['momentum']
        weight_decay = self.defaults['weight_decay']

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if not p.requires_grad:
                    continue

                if self.decay_filter(p):
                    p.grad = p.grad + weight_decay * p.data

                g = gradients[i]
                g_shape = g.shape
                g = g.view(g_shape[0], -1)
                matrix_A = self.matrix_A[i]
                matrix_G = self.matrix_G[i]
                g = torch.matmul(torch.matmul(matrix_G, g), matrix_A)

                if self.use_nesterov:
                    m = group['momentum_buffer']
                    m.mul_(momentum).add_(g)
                    p.data.add_(-lr, m)
                else:
                    p.data.add_(-lr, g)

        return success
