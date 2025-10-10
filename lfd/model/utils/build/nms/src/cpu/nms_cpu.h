#pragma once
#include <torch/extension.h>

at::Tensor nms_cpu(const at::Tensor& dets, const float threshold);
at::Tensor soft_nms_cpu(const at::Tensor& dets, const float threshold,
                        const unsigned char method, const float sigma,
                        const float min_score);
std::vector<std::vector<int> > nms_match_cpu(const at::Tensor& dets, const float threshold);