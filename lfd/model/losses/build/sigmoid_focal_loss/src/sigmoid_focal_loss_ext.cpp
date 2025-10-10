// modify from
// https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/csrc/SigmoidFocalLoss.h
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <cfloat> // for FLT_MIN

template <typename scalar_t>
at::Tensor SigmoidFocalLoss_forward_cpu(
    const at::Tensor& logits,
    const at::Tensor& targets,
    const int num_classes,
    const float gamma, 
    const float alpha) {
    
    auto logits_data = logits.data_ptr<scalar_t>();
    auto targets_data = targets.data_ptr<int64_t>();
    const int num_samples = logits.size(0);
    const int num_classes_val = logits.size(1);
    
    auto losses = at::empty({num_samples, num_classes_val}, logits.options());
    auto losses_data = losses.data_ptr<scalar_t>();
    
    const int total_elements = num_samples * num_classes_val;
    
    for (int i = 0; i < total_elements; i++) {
        int n = i / num_classes_val;
        int d = i % num_classes_val;
        int64_t t = targets_data[n];
        
        scalar_t c1 = (t == static_cast<int64_t>(d));
        scalar_t c2 = (t >= 0 && t != static_cast<int64_t>(d));
        
        scalar_t zn = (1.0 - alpha);
        scalar_t zp = alpha;
        
        scalar_t logit_val = logits_data[i];
        scalar_t p = 1.0 / (1.0 + std::exp(-logit_val));
        
        scalar_t term1 = std::pow(1.0 - p, gamma) * 
                         std::log(std::max(p, static_cast<scalar_t>(FLT_MIN)));
                         scalar_t term2;
                         if (logit_val >= 0) {
                             term2 = std::pow(p, gamma) * 
                                     (-logit_val - std::log(1.0 + std::exp(-logit_val)));
                         } else {
                             term2 = std::pow(p, gamma) * 
                                     (-logit_val + std::log(1.0 + std::exp(logit_val)));
                         }
                         
                         losses_data[i] = 0;
                         if (c1) losses_data[i] -= term1 * zp;
                         if (c2) losses_data[i] -= term2 * zn;
                     }
                     return losses;
                 }
                 template <typename scalar_t>
                 at::Tensor SigmoidFocalLoss_backward_cpu(
                     const at::Tensor& logits,
                     const at::Tensor& targets,
                     const at::Tensor& d_losses,
                     const int num_classes,
                     const float gamma,
                     const float alpha) {
                     
                     auto logits_data = logits.data_ptr<scalar_t>();
                     auto targets_data = targets.data_ptr<int64_t>();
                     auto d_losses_data = d_losses.data_ptr<scalar_t>();
                     
                     const int num_samples = logits.size(0);
                     const int num_classes_val = logits.size(1);
                     
                     auto d_logits = at::zeros({num_samples, num_classes_val}, logits.options());
                     auto d_logits_data = d_logits.data_ptr<scalar_t>();
                     
                     const int total_elements = num_samples * num_classes_val;
                     
                     for (int i = 0; i < total_elements; i++) {
                         int n = i / num_classes_val;
                         int d = i % num_classes_val;
                         int64_t t = targets_data[n];
                         
                         scalar_t c1 = (t == static_cast<int64_t>(d));
                         scalar_t c2 = (t >= 0 && t != static_cast<int64_t>(d));
                         
                         scalar_t zn = (1.0 - alpha);
                         scalar_t zp = alpha;
                         
                         scalar_t logit_val = logits_data[i];
                         scalar_t p = 1.0 / (1.0 + std::exp(-logit_val));
                         scalar_t term1 = std::pow(1.0 - p, gamma) * 
                         (1.0 - p - (p * gamma * std::log(std::max(p, static_cast<scalar_t>(FLT_MIN)))));
        
        scalar_t term2;
        if (logit_val >= 0) {
            term2 = std::pow(p, gamma) * 
                    ((1.0 - p) * gamma * (-logit_val - std::log(1.0 + std::exp(-logit_val))) - p);
        } else {
            term2 = std::pow(p, gamma) * 
                    ((1.0 - p) * gamma * (-logit_val + std::log(1.0 + std::exp(logit_val))) - p);
        }
        
        d_logits_data[i] = 0;
        if (c1) d_logits_data[i] -= term1 * zp;
        if (c2) d_logits_data[i] -= term2 * zn;
          
        
        d_logits_data[i] *= d_losses_data[i];
    }
    return d_logits;
}

#ifdef WITH_CUDA
at::Tensor SigmoidFocalLoss_forward_cuda(const at::Tensor &logits,
                                         const at::Tensor &targets,
                                         const int num_classes,
                                         const float gamma, const float alpha);

at::Tensor SigmoidFocalLoss_backward_cuda(const at::Tensor &logits,
                                          const at::Tensor &targets,
                                          const at::Tensor &d_losses,
                                          const int num_classes,
                                          const float gamma, const float alpha);
#endif


at::Tensor SigmoidFocalLoss_forward(const at::Tensor &logits,
  const at::Tensor &targets,
  const int num_classes, const float gamma,
  const float alpha) {
if (logits.device().is_cuda()) {
#ifdef WITH_CUDA
at::DeviceGuard guard(logits.device());
return SigmoidFocalLoss_forward_cuda(logits, targets, num_classes, gamma,
       alpha);
#else
AT_ERROR("SigmoidFocalLoss is not compiled with GPU support");
#endif
}

return AT_DISPATCH_FLOATING_TYPES(
  logits.scalar_type(), "SigmoidFocalLoss_forward", [&] {
    return SigmoidFocalLoss_forward_cpu<scalar_t>(
        logits, targets, num_classes, gamma, alpha);
  });
}

at::Tensor SigmoidFocalLoss_backward(const at::Tensor &logits,
  const at::Tensor &targets,
  const at::Tensor &d_losses,
  const int num_classes, const float gamma,
  const float alpha) {
if (logits.device().is_cuda()) {
#ifdef WITH_CUDA
at::DeviceGuard guard(logits.device());
return SigmoidFocalLoss_backward_cuda(logits, targets, d_losses,
       num_classes, gamma, alpha);
#else
AT_ERROR("SigmoidFocalLoss is not compiled with GPU support");
#endif
}

return AT_DISPATCH_FLOATING_TYPES(
  logits.scalar_type(), "SigmoidFocalLoss_backward", [&] {
    return SigmoidFocalLoss_backward_cpu<scalar_t>(
        logits, targets, d_losses, num_classes, gamma, alpha);
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("forward", &SigmoidFocalLoss_forward,
    "SigmoidFocalLoss forward (CPU and GPU)");
m.def("backward", &SigmoidFocalLoss_backward,
    "SigmoidFocalLoss backward (CPU and GPU)");
}