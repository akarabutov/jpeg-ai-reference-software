"""Fused Lamb optimizer."""
import sys
try:
    from apex.optimizers import FusedLAMBV2 as FusedLAMB_IMPL

    __all__ = ['FusedLambWrapper']


    class FusedLambWrapper(FusedLAMB_IMPL):
        r"""Wrapper of Fused Lamb algorithm.
        """
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, adam=False):
            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adam_w_mode=adam)
            super(FusedLambWrapper, self).__init__(params, **defaults)

        def zero_grad(self, set_grad_none=False):
            if set_grad_none:
                for group in self.param_groups:
                    for p in group['params']:
                        p.grad = None
            else:
                super(FusedLambWrapper, self).zero_grad()
except:
    print('APEX lib is required')
