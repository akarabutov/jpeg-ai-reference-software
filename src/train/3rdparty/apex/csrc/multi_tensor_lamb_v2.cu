#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include <assert.h>
#include <iostream>

#include "type_shim.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

template<typename T>
__device__ __forceinline__ bool is_aligned(T* p){
  return ((uint64_t)p) % (ILP*sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

typedef enum{
  MOMENT_MODE_0   =0, // L2 regularization mode
  MOMENT_MODE_1   =1  // Decoupled weight decay mode
} adamMode_t;

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::optional<bool> per_tensor_python);

using MATH_T = float;

template<typename T>
struct LAMBStage1Functor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<4>& tl,
    const float beta1,
    const float beta2,
    const float epsilon,
    adamMode_t mode,
    const float decay)
  {

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

//    float clipped_global_grad_norm = (*global_grad_norm) > max_global_grad_norm ? (*global_grad_norm) / max_global_grad_norm : 1.0f;

    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    T* m = (T*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;

    T* v = (T*)tl.addresses[3][tensor_loc];
    v += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    MATH_T r_g[ILP];
    MATH_T r_p[ILP];
    MATH_T r_m[ILP];
    MATH_T r_v[ILP];
    // to make things simple, we put aligned case in a different code path
    if(n % ILP == 0 &&
       chunk_size % ILP == 0 &&
       is_aligned(g) &&
       is_aligned(p) &&
       is_aligned(m) &&
       is_aligned(v))
    {
      T l_g[ILP];
      T l_p[ILP];
      T l_m[ILP];
      T l_v[ILP];
      for(int i_start = threadIdx.x; i_start*ILP < n && i_start*ILP < chunk_size; i_start += blockDim.x)
      {
        // load
        load_store(l_g, g, 0, i_start);
        if (decay != 0)
          load_store(l_p, p, 0, i_start);
        load_store(l_m, m, 0, i_start);
        load_store(l_v, v, 0, i_start);
        // unpack
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          r_g[ii] = l_g[ii];
          if (decay == 0) {
            r_p[ii] = MATH_T(0);
          }
          else {
            r_p[ii] = l_p[ii];
          }
          r_m[ii] = l_m[ii];
          r_v[ii] = l_v[ii];
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          if (mode == MOMENT_MODE_0) {
            MATH_T scaled_grad = r_g[ii];
            r_m[ii] = r_m[ii] * beta1 + (1 - beta1) * scaled_grad;
            r_v[ii] = r_v[ii] * beta2 + (1 - beta2) * scaled_grad * scaled_grad;
            MATH_T next_m_unbiased = r_m[ii];
            MATH_T next_v_unbiased = r_v[ii];
            MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
            r_p[ii] = (next_m_unbiased / denom) + (decay * r_p[ii]);
          }
          else {
            MATH_T scaled_grad = r_g[ii];
            r_m[ii] = r_m[ii] * beta1 + (1 - beta1) * scaled_grad;
            r_v[ii] = r_v[ii] * beta2 + (1 - beta2) * scaled_grad * scaled_grad;
            MATH_T next_m_unbiased = r_m[ii];
            MATH_T next_v_unbiased = r_v[ii];
            MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
            r_p[ii] = (next_m_unbiased / denom) + (decay * r_p[ii]);
          }
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          l_p[ii] = r_p[ii];
          l_m[ii] = r_m[ii];
          l_v[ii] = r_v[ii];
        }
        // store
        load_store(g, l_p, i_start, 0);
        load_store(m, l_m, i_start, 0);
        load_store(v, l_v, i_start, 0);
      }
    }
    else
    {
      // see note in multi_tensor_scale_kernel.cu
      for(int i_start = 0;
          i_start < n && i_start < chunk_size;
          i_start += blockDim.x*ILP)
      {
        MATH_T r_g[ILP];
        MATH_T r_p[ILP];
        MATH_T r_m[ILP];
        MATH_T r_v[ILP];
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            r_g[ii] = g[i];
            // special ?optimization? for lamb stage 1
            if (decay == 0) {
              r_p[ii] = MATH_T(0);
            }
            else {
              r_p[ii] = p[i];
            }
            r_m[ii] = m[i];
            r_v[ii] = v[i];
          } else {
            r_g[ii] = MATH_T(0);
            r_p[ii] = MATH_T(0);
            r_m[ii] = MATH_T(0);
            r_v[ii] = MATH_T(0);
          }
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          if (mode == MOMENT_MODE_0) {
            MATH_T scaled_grad = r_g[ii];
            r_m[ii] = r_m[ii] * beta1 + (1 - beta1) * scaled_grad;
            r_v[ii] = r_v[ii] * beta2 + (1 - beta2) * scaled_grad * scaled_grad;
            MATH_T next_m_unbiased = r_m[ii];
            MATH_T next_v_unbiased = r_v[ii];
            MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
            r_p[ii] = (next_m_unbiased / denom) + (decay * r_p[ii]);

          }
          else {
            MATH_T scaled_grad = r_g[ii];
            r_m[ii] = r_m[ii] * beta1 + (1 - beta1) * scaled_grad;
            r_v[ii] = r_v[ii] * beta2 + (1 - beta2) * scaled_grad * scaled_grad;
            MATH_T next_m_unbiased = r_m[ii];
            MATH_T next_v_unbiased = r_v[ii];
            MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
            r_p[ii] = (next_m_unbiased / denom) + (decay * r_p[ii]);
          }
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            g[i] = r_p[ii];
            m[i] = r_m[ii];
            v[i] = r_v[ii];
          }
        }
      }
    }
  }
};

// Step 2 reads in 'update' value and per-tensor param_norm and update_norm.
// It computes new parameter value.
template<typename T>
struct LAMBStage2Functor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<2>& tl,
    const float* per_tensor_param_norm,
    const float* per_tensor_update_norm,
    const float learning_rate,
    const float epsilon,
    const float decay)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    MATH_T ratio = learning_rate;
    float param_norm = per_tensor_param_norm[tensor_num];
    float update_norm = per_tensor_update_norm[tensor_num];
    if (update_norm != 0.0f && param_norm != 0.0f)
    {
        ratio = learning_rate * (min(param_norm, 10.0) / (update_norm + epsilon));
    }

    T* update = (T*)tl.addresses[0][tensor_loc];
    update += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // to make things simple, we put aligned case in a different code path
    if(n % ILP == 0 &&
       chunk_size % ILP == 0 &&
       is_aligned(p) &&
       is_aligned(update))
    {
      T r_p[ILP];
      T r_update[ILP];
      for(int i_start = threadIdx.x; i_start*ILP < n && i_start*ILP < chunk_size; i_start += blockDim.x)
      {
        // load
        load_store(r_p, p, 0, i_start);
        load_store(r_update, update, 0, i_start);
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          r_p[ii] = static_cast<MATH_T>(r_p[ii]) - (ratio * static_cast<MATH_T>(r_update[ii]));
        }
        load_store(p, r_p, i_start, 0);
      }
    }
    else
    {
      for(int i_start = 0;
          i_start < n && i_start < chunk_size;
          i_start += blockDim.x*ILP)
      {
        MATH_T r_p[ILP];
        MATH_T r_update[ILP];
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            r_p[ii] = p[i];
            r_update[ii] = update[i];
          }
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          r_p[ii] = r_p[ii] - (ratio * r_update[ii]);
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            p[i] = r_p[ii];
          }
        }
      }
    }
  }
};


void multi_tensor_lamb_cuda_v2(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  const float lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const float weight_decay,
  const int mode)
{
  using namespace at;

  std::vector<std::vector<at::Tensor>> grad_list(tensor_lists.begin(), tensor_lists.begin()+1);
  std::vector<std::vector<at::Tensor>> param_list(tensor_lists.begin()+1, tensor_lists.begin()+2);

  // Compute per tensor param norm
  auto param_norm_tuple = multi_tensor_l2norm_cuda(chunk_size, noop_flag, param_list, true);
  //const float *ptr = std::get<1>(param_norm_tuple).to(at::kCPU).DATA_PTR<float>();
  //auto cpu_tensor = std::get<1>(param_norm_tuple).to(at::kCPU);
  //std::cout << cpu_tensor << " CPU_Tensor\n";
  //std::cout << cpu_tensor.sizes()[0] << " CPU_Tensor size n\n";
  //std::cout << cpu_tensor.sizes()[1] << " CPU_Tensor size c\n";
  //std::cout << cpu_tensor.sizes()[2] << " CPU_Tensor size h\n";
  //std::cout << cpu_tensor.sizes()[3] << " CPU_Tensor size w\n";

  // We now in-place modify grad to store update before compute its norm
  // Generally this is not a issue since people modify grad in step() method all the time
  // We can also grab list of empty tensor to avoid this, but I'd like to save space/cpu code
  DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "lamb_stage_1",
      multi_tensor_apply<4>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        LAMBStage1Functor<scalar_t_0>(),
        beta1,
        beta2,
        epsilon,
        (adamMode_t) mode,
        weight_decay);)

  //std::vector<std::vector<at::Tensor>> mom_list(tensor_lists.begin()+2, tensor_lists.begin()+3);
  //std::vector<std::vector<at::Tensor>> vel_list(tensor_lists.begin()+3, tensor_lists.begin()+4);
  //std::cout << mom_list.size() << " mom_size\n";
  //std::cout << vel_list.size() << " vel_size\n";
  //std::cout << mom_list[0].size() << " mom0_size\n";
  //std::cout << vel_list[0].size() << " vel0_size\n";
  //for (int i = 0; i < mom_list[0].size(); i++) {
  //        std::cout << mom_list[0][i].to(at::kCPU) << " mom\n";
  //        std::cout << vel_list[0][i].to(at::kCPU) << " vel\n";
  //}

  // Compute update norms
  auto update_norm_tuple = multi_tensor_l2norm_cuda(chunk_size, noop_flag, grad_list, true);
  //auto cpu_tensor = std::get<1>(update_norm_tuple).to(at::kCPU);
  //std::cout << cpu_tensor << " CPU_Tensor\n";

  std::vector<std::vector<at::Tensor>> grad_param_list(tensor_lists.begin(), tensor_lists.begin()+2);

  DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "lamb_stage_2",
      multi_tensor_apply<2>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        grad_param_list,
        LAMBStage2Functor<scalar_t_0>(),
        std::get<1>(param_norm_tuple).DATA_PTR<float>(),
        std::get<1>(update_norm_tuple).DATA_PTR<float>(),
        lr,
        epsilon,
        weight_decay); )

  AT_CUDA_CHECK(cudaGetLastError());

}
