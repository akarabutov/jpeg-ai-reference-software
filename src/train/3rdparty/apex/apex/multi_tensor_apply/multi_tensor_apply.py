import torch

class MultiTensorApply(object):
    available = False
    warned = False

    def __init__(self, chunk_size):
        try:
            import amp_C
            MultiTensorApply.available = True
            self.chunk_size = chunk_size
        except ImportError as err:
            MultiTensorApply.available = False
            MultiTensorApply.import_err = err

    def check_avail(self):
        if MultiTensorApply.available == False:
            raise RuntimeError(
                "Attempted to call MultiTensorApply method, but MultiTensorApply "
                "is not available, possibly because Apex was installed without "
                "--cpp_ext --cuda_ext.  Original import error message:",
                MultiTensorApply.import_err)

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        self.check_avail()

        return op(self.chunk_size,
                  noop_flag_buffer,
                  tensor_lists,
                  *args)

#    multi_tensor_applier(self,
#                         op,
#                         noop_flag_buffer,
#                         tensor_lists,
#                         *args):

#    multi_tensor_applier(self.multi_tensor_lamb,     # op
#                         self._dummy_overflow_buf,   # noop_flag_buffer
#                         [g_16, p_16, m_16, v_16],   # tensor_lists
#                         group['lr'],                # *args
#                         beta1,
#                         beta2,
#                         group['eps'],
#                         group['step'],
#                         bias_correction,
#                         group['weight_decay'],
#                         grad_averaging,
#                         self.adam_w_mode,
#                         global_grad_norm,
#                         max_grad_norm,
#                         self.use_nvlamb)

#void multi_tensor_lamb_cuda(
#    chunk_size = 2048*32,
#    noop_flag = self._dummy_overflow_buf,
#    tensor_lists = [g_16, p_16, m_16, v_16],
#    lr = group['lr'],
#    beta1 = beta1,
#    beta2 = beta2,
#    epsilon = group['eps'],
#    step = group['step'],
#    bias_correction = bias_correction,
#    weight_decay = group['weight_decay'],
#    grad_averaging = grad_averaging,
#    mode = self.adam_w_mode,
#    global_grad_norm = global_grad_norm,
#    use_nvlamb_python = self.use_nvlamb_python
#)

#void multi_tensor_lamb_cuda(
#  int chunk_size,
#  at::Tensor noop_flag,
#  std::vector<std::vector<at::Tensor>> tensor_lists,
#  const float lr,
#  const float beta1,
#  const float beta2,
#  const float epsilon,
#  const int step,
#  const int bias_correction,
#  const float weight_decay,
#  const int grad_averaging,
#  const int mode,
#  at::Tensor global_grad_norm,
#  const float max_grad_norm,
#  at::optional<bool> use_nvlamb_python);
