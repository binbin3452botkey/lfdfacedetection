#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

#include <cfloat>


#define DIVUP(a, b) (((a) + (b) - 1) / (b))

template <typename scalar_t>
__global__ void SigmoidFocalLossForward(const int nthreads,
                                       const scalar_t *logits,
                                       const int64_t *targets,
                                       const int num_classes,
                                       const float gamma, 
                                       const float alpha,
                                       const int num, 
                                       scalar_t *losses) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nthreads) return;

  const int n = index / num_classes;
  const int d = index % num_classes;
  const int t = targets[n];

  const scalar_t c1 = (t == d);
  const scalar_t c2 = (t >= 0 & t != d);

  const scalar_t zn = (1.0 - alpha);
  const scalar_t zp = (alpha);

  const scalar_t logit = logits[index];
  const scalar_t p = 1. / (1. + expf(-logit));

  // term1 = (1-p)^gamma * log(p)
  const scalar_t term1 = powf((1. - p), gamma) * logf(max(p, FLT_MIN));

  // term2 = p^gamma * log(1-p)
  const scalar_t term2 = powf(p, gamma) *
      (-logit * (logit >= 0) - 
       logf(1. + expf(logit - 2. * logit * (logit >= 0))));

  losses[index] = (-c1 * term1 * zp) + (-c2 * term2 * zn);
}

template <typename scalar_t>
__global__ void SigmoidFocalLossBackward(
    const int nthreads, 
    const scalar_t *logits,
    const int64_t *targets,
    const scalar_t *d_losses,
    const int num_classes,
    const float gamma,
    const float alpha,
    const int num,
    scalar_t *d_logits) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nthreads) return;

  const int n = index / num_classes;
  const int d = index % num_classes;
  const int t = targets[n];

  const scalar_t c1 = (t == d);
  const scalar_t c2 = (t >= 0 & t != d);

  const scalar_t zn = (1.0 - alpha);
  const scalar_t zp = (alpha);
  const scalar_t logit = logits[index];
  const scalar_t p = 1. / (1. + expf(-logit));

  // term1 = (1-p)^g * (1 - p - g*p*log(p))
  scalar_t term1 = powf((1. - p), gamma) * 
      (1. - p - (p * gamma * logf(max(p, FLT_MIN))));

  // term2 = p^g * (g*(1-p)*log(1-p) - p)
  scalar_t term2 = powf(p, gamma) *
      ((-logit * (logit >= 0) - 
        logf(1. + expf(logit - 2. * logit * (logit >= 0)))) *
       (1. - p) * gamma - p);

  d_logits[index] = ((-c1 * term1 * zp) + (-c2 * term2 * zn)) * d_losses[index];
}

at::Tensor SigmoidFocalLoss_forward_cuda(
    const at::Tensor &logits,
    const at::Tensor &targets,
    const int num_classes,
    const float gamma, 
    const float alpha) {
  
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "logits should be NxClass");

  const int num_samples = logits.size(0);
  const int num_classes_ = logits.size(1);
  auto losses = at::empty({num_samples, num_classes_}, logits.options());
  const int64_t losses_size = num_samples * num_classes_;

  const int blocks = DIVUP(losses_size, 512);
  const dim3 grid(std::min(blocks, 4096));
  const dim3 block(512);

  if (losses.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return losses;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      logits.scalar_type(), "SigmoidFocalLoss_forward", [&] {
        SigmoidFocalLossForward<scalar_t>
            <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                losses_size,
                logits.contiguous().data_ptr<scalar_t>(),
                targets.contiguous().data_ptr<int64_t>(),
                num_classes_, gamma, alpha,
                num_samples,
                losses.data_ptr<scalar_t>());
      });
  
  AT_CUDA_CHECK(cudaGetLastError());
  return losses;
}

at::Tensor SigmoidFocalLoss_backward_cuda(
    const at::Tensor &logits,
    const at::Tensor &targets,
    const at::Tensor &d_losses,
    const int num_classes,
    const float gamma,
    const float alpha) {
  
  TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
  TORCH_CHECK(d_losses.is_cuda(), "d_losses must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "logits should be NxClass");

  const int num_samples = logits.size(0);
  const int num_classes_ = logits.size(1);
  auto d_logits = at::zeros({num_samples, num_classes_}, logits.options());
  const int64_t d_logits_size = num_samples * num_classes_;

  const int blocks = DIVUP(d_logits_size, 512);
  const dim3 grid(std::min(blocks, 4096));
  const dim3 block(512);

  if (d_logits.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return d_logits;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      logits.scalar_type(), "SigmoidFocalLoss_backward", [&] {
        SigmoidFocalLossBackward<scalar_t>
            <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                d_logits_size,
                logits.contiguous().data_ptr<scalar_t>(),
                targets.contiguous().data_ptr<int64_t>(),
                d_losses.contiguous().data_ptr<scalar_t>(),
                num_classes_, gamma, alpha,
                num_samples,
                d_logits.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
  return d_logits;
}
