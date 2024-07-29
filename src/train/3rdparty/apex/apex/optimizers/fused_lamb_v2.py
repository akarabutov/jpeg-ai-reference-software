import torch
from apex.multi_tensor_apply import multi_tensor_applier

def print_sum(tensor_name, input_tensor):
    """print sum of tensor, only used for debug"""
    if isinstance(input_tensor, torch.Tensor):
        tensor_sum = input_tensor.detach().cpu().numpy().flatten().sum()
        print("\033[1m\033[45;33m {} {} \033[0m\n".format(tensor_name, tensor_sum))
    else:
        print("\033[1m\033[45;33m {} {} \033[0m\n".format(tensor_name, 'is not tensor'))

class FusedLAMBV2(torch.optim.Optimizer):
    """Implements LAMB algorithm.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        adam_w_mode=False
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam = 1 if adam_w_mode else 0
        super(FusedLAMBV2, self).__init__(params, defaults)
        if multi_tensor_applier.available:
            import amp_C
            self.multi_tensor_l2norm=amp_C.multi_tensor_l2norm
            # Skip buffer
            self._dummy_overflow_buf = torch.tensor(
                    [0],
                    dtype=torch.int,
                    device=self.param_groups[0]["params"][0].device
                )
            self.multi_tensor_lamb_v2 = amp_C.multi_tensor_lamb_v2
        else:
            raise RuntimeError('apex.optimizers.FusedLAMBV2 requires cuda extensions')

    def zero_grad(self, set_to_none=False):
        if set_to_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedLAMBV2, self).zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        g_all_32, g_all_16 = [], []
        for group in self.param_groups:
            for p in group['params']:
#                print_sum('grad', p.grad.data)
#                print_sum('grad', p.data)
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients.')
                if p.dtype == torch.float32:
                    g_all_32.append(p.grad.data)
                elif p.dtype == torch.float16:
                    g_all_16.append(p.grad.data)
                    # TODO: add support for TF32
                else:
                    raise RuntimeError('FusedLAMBV2 only support fp16 and fp32.')

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            g_16, p_16, m_16, v_16 = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['step'] += 1

                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                    v_16.append(state['exp_avg_sq'])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])
                else:
                    raise RuntimeError('FusedLAMBV2 only support fp16 and fp32.')
            #print('len_g_32', len(g_32), 'len_g_16', len(g_16))

            if len(g_16) > 0:
                multi_tensor_applier(
                        self.multi_tensor_lamb_v2,
                        self._dummy_overflow_buf,
                        [g_16, p_16, m_16, v_16],
                        group['lr'],
                        beta1,
                        beta2,
                        group['eps'],
                        group['weight_decay'],
                        self.adam
                )

            if len(g_32) > 0:
                multi_tensor_applier(
                        self.multi_tensor_lamb_v2,
                        self._dummy_overflow_buf,
                        [g_32, p_32, m_32, v_32],
                        group['lr'],
                        beta1,
                        beta2,
                        group['eps'],
                        group['weight_decay'],
                        self.adam
                )
        #print('------------------------------------')
        #print('------------------------------------')
        #for group in self.param_groups:
        #    for p in group['params']:
        #        state = self.state[p]
        #        print_sum('grad', p.data)
        #for group in self.param_groups:
        #    for p in group['params']:
        #        state = self.state[p]
        #        if state['step'] == 3:
        #            exit()
        return loss
