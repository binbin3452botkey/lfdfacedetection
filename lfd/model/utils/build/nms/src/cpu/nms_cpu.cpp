#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "nms_cpu.h" 


#if defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif


template <typename scalar_t>
inline scalar_t iou(const scalar_t* RESTRICT a, const scalar_t* RESTRICT b) {
    scalar_t left = std::max(a[0], b[0]);
    scalar_t top = std::max(a[1], b[1]);
    scalar_t right = std::min(a[2], b[2]);
    scalar_t bottom = std::min(a[3], b[3]);
    
    scalar_t width = std::max(static_cast<scalar_t>(0), right - left);
    scalar_t height = std::max(static_cast<scalar_t>(0), bottom - top);
    scalar_t inter = width * height;
    
    scalar_t area_a = (a[2] - a[0]) * (a[3] - a[1]);
    scalar_t area_b = (b[2] - b[0]) * (b[3] - b[1]);
    
    return inter / (area_a + area_b - inter + 1e-8);
}


template <typename scalar_t>
at::Tensor nms_cpu_kernel(const at::Tensor& dets, const float threshold) {
    AT_ASSERTM(!dets.device().is_cuda(), "dets must be a CPU tensor");

    if (dets.numel() == 0) {
        return at::empty({0}, dets.options().dtype(at::kLong));
    }

    auto scores = dets.select(1, 4);
    auto order_t = std::get<1>(scores.sort(0, true));  
    
    auto ndets = dets.size(0);
    auto dets_ptr = dets.data_ptr<scalar_t>();
    auto order = order_t.data_ptr<int64_t>();
    
    
    std::vector<uint8_t> suppressed(ndets, 0);
    std::vector<int64_t> keep_vec;
    keep_vec.reserve(ndets);

    for (int64_t i = 0; i < ndets; i++) {
        auto idx = order[i];
        if (suppressed[idx]) continue;
        
        keep_vec.push_back(idx);
        
        auto* box1 = dets_ptr + idx * 5;
        for (int64_t j = i + 1; j < ndets; j++) {
            auto jdx = order[j];
            if (suppressed[jdx]) continue;
            
            auto* box2 = dets_ptr + jdx * 5;
            if (iou(box1, box2) > threshold) {
                suppressed[jdx] = 1;
            }
        }
    }
    
   
    auto keep = at::from_blob(
        keep_vec.data(), 
        {static_cast<int64_t>(keep_vec.size())}, 
        at::kLong
    ).clone();
    
    return keep;
}

at::Tensor nms_cpu(const at::Tensor& dets, const float threshold) {
    return AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms", [&] {
        return nms_cpu_kernel<scalar_t>(dets, threshold);
    });
}

template <typename scalar_t>
at::Tensor soft_nms_cpu_kernel(const at::Tensor& dets, const float threshold,
                               const unsigned char method, const float sigma,
                               const float min_score) {
                                AT_ASSERTM(!dets.device().is_cuda(), "dets must be a CPU tensor");

                                if (dets.numel() == 0) {
                                  return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
                                }
                              
                                auto x1_t = dets.select(1, 0).contiguous();
                                auto y1_t = dets.select(1, 1).contiguous();
                                auto x2_t = dets.select(1, 2).contiguous();
                                auto y2_t = dets.select(1, 3).contiguous();
                                auto scores_t = dets.select(1, 4).contiguous();
                              
                                at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);
                              
                                auto ndets = dets.size(0);
                                auto x1 = x1_t.data_ptr<scalar_t>();
                                auto y1 = y1_t.data_ptr<scalar_t>();
                                auto x2 = x2_t.data_ptr<scalar_t>();
                                auto y2 = y2_t.data_ptr<scalar_t>();
                                auto scores = scores_t.data_ptr<scalar_t>();
                                auto areas = areas_t.data_ptr<scalar_t>();
                              
                                int64_t pos = 0;
                                at::Tensor inds_t = at::arange(ndets, dets.options());
                                auto inds = inds_t.data_ptr<scalar_t>();
                              
                                for (int64_t i = 0; i < ndets; i++) {
                                  auto max_score = scores[i];
                                  auto max_pos = i;
                              
                                  auto ix1 = x1[i];
                                  auto iy1 = y1[i];
                                  auto ix2 = x2[i];
                                  auto iy2 = y2[i];
                                  auto iscore = scores[i];
                                  auto iarea = areas[i];
                                  auto iind = inds[i];
                              
                                  pos = i + 1;
                                  // get max box
                                  while (pos < ndets) {
                                    if (max_score < scores[pos]) {
                                      max_score = scores[pos];
                                      max_pos = pos;
                                    }
                                    pos = pos + 1;
                                  }
                                  // add max box as a detection
                                  x1[i] = x1[max_pos];
                                  y1[i] = y1[max_pos];
                                  x2[i] = x2[max_pos];
                                  y2[i] = y2[max_pos];
                                  scores[i] = scores[max_pos];
                                  areas[i] = areas[max_pos];
                                  inds[i] = inds[max_pos];
                              
                                  // swap ith box with position of max box
                                  x1[max_pos] = ix1;
                                  y1[max_pos] = iy1;
                                  x2[max_pos] = ix2;
                                  y2[max_pos] = iy2;
                                  scores[max_pos] = iscore;
                                  areas[max_pos] = iarea;
                                  inds[max_pos] = iind;
                              
                                  ix1 = x1[i];
                                  iy1 = y1[i];
                                  ix2 = x2[i];
                                  iy2 = y2[i];
                                  iscore = scores[i];
                                  iarea = areas[i];
                              
                                  pos = i + 1;
                                  // NMS iterations, note that N changes if detection boxes fall below
                                  // threshold
                                  while (pos < ndets) {
                                    auto xx1 = std::max(ix1, x1[pos]);
                                    auto yy1 = std::max(iy1, y1[pos]);
                                    auto xx2 = std::min(ix2, x2[pos]);
                                    auto yy2 = std::min(iy2, y2[pos]);
                              
                                    auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1);
                                    auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1);
                                    auto inter = w * h;
                                    auto ovr = inter / (iarea + areas[pos] - inter);
                              
                                    scalar_t weight = 1.;
                                    if (method == 1) {
                                      if (ovr > threshold) weight = 1 - ovr;
                                    } else if (method == 2) {
                                      weight = std::exp(-(ovr * ovr) / sigma);
                                    } else {
                                      // original NMS
                                      if (ovr > threshold) {
                                        weight = 0;
                                      } else {
                                        weight = 1;
                                      }
                                    }
                                    scores[pos] = weight * scores[pos];
                                    // if box score falls below threshold, discard the box by
                                    // swapping with last box update N
                                    if (scores[pos] < min_score) {
                                      x1[pos] = x1[ndets - 1];
                                      y1[pos] = y1[ndets - 1];
                                      x2[pos] = x2[ndets - 1];
                                      y2[pos] = y2[ndets - 1];
                                      scores[pos] = scores[ndets - 1];
                                      areas[pos] = areas[ndets - 1];
                                      inds[pos] = inds[ndets - 1];
                                      ndets = ndets - 1;
                                      pos = pos - 1;
                                    }
                                    pos = pos + 1;
                                  }
                                }
                                at::Tensor result = at::zeros({6, ndets}, dets.options());
                                result[0] = x1_t.slice(0, 0, ndets);
                                result[1] = y1_t.slice(0, 0, ndets);
                                result[2] = x2_t.slice(0, 0, ndets);
                                result[3] = y2_t.slice(0, 0, ndets);
                                result[4] = scores_t.slice(0, 0, ndets);
                                result[5] = inds_t.slice(0, 0, ndets);
                              
                                result = result.t().contiguous();
                                return result;
}

at::Tensor soft_nms_cpu(const at::Tensor& dets, const float threshold,
                        const unsigned char method, const float sigma,
                        const float min_score) {
                            at::Tensor result;
                            AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "soft_nms", [&] {
                              result = soft_nms_cpu_kernel<scalar_t>(dets, threshold, method, sigma,
                                                                     min_score);
                            });
                            return result;
}


template <typename scalar_t>
std::vector<std::vector<int> > nms_match_cpu_kernel(const at::Tensor& dets,
                                                    const float threshold) {
                                                        AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");

                                                        auto x1_t = dets.select(1, 0).contiguous();
                                                        auto y1_t = dets.select(1, 1).contiguous();
                                                        auto x2_t = dets.select(1, 2).contiguous();
                                                        auto y2_t = dets.select(1, 3).contiguous();
                                                        auto scores = dets.select(1, 4).contiguous();
                                                      
                                                        at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);
                                                      
                                                        auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
                                                      
                                                        auto ndets = dets.size(0);
                                                        at::Tensor suppressed_t =
                                                            at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));
                                                      
                                                        auto suppressed = suppressed_t.data_ptr<uint8_t>();
                                                        auto order = order_t.data_ptr<int64_t>();
                                                        auto x1 = x1_t.data_ptr<scalar_t>();
                                                        auto y1 = y1_t.data_ptr<scalar_t>();
                                                        auto x2 = x2_t.data_ptr<scalar_t>();
                                                        auto y2 = y2_t.data_ptr<scalar_t>();
                                                        auto areas = areas_t.data_ptr<scalar_t>();
                                                      
                                                        std::vector<int> keep;
                                                        std::vector<std::vector<int> > matched;
                                                      
                                                        for (int64_t _i = 0; _i < ndets; _i++) {
                                                          auto i = order[_i];
                                                          if (suppressed[i] == 1) continue;
                                                          keep.push_back(i);
                                                          std::vector<int> v_i;
                                                          auto ix1 = x1[i];
                                                          auto iy1 = y1[i];
                                                          auto ix2 = x2[i];
                                                          auto iy2 = y2[i];
                                                          auto iarea = areas[i];
                                                      
                                                          for (int64_t _j = _i + 1; _j < ndets; _j++) {
                                                            auto j = order[_j];
                                                            if (suppressed[j] == 1) continue;
                                                            auto xx1 = std::max(ix1, x1[j]);
                                                            auto yy1 = std::max(iy1, y1[j]);
                                                            auto xx2 = std::min(ix2, x2[j]);
                                                            auto yy2 = std::min(iy2, y2[j]);
                                                      
                                                            auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1);
                                                            auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1);
                                                            auto inter = w * h;
                                                            auto ovr = inter / (iarea + areas[j] - inter);
                                                            if (ovr >= threshold) {
                                                              suppressed[j] = 1;
                                                              v_i.push_back(j);
                                                            }
                                                          }
                                                          matched.push_back(v_i);
                                                        }
                                                        for (size_t i = 0; i < keep.size(); i++)
                                                          matched[i].insert(matched[i].begin(), keep[i]);
                                                        return matched;
}

std::vector<std::vector<int> > nms_match_cpu(const at::Tensor& dets,
                                             const float threshold) {
                                                std::vector<std::vector<int> > result;
                                                // result = nms_match_cpu_kernel<scalar_t>(dets, threshold);
                                                AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms_match", [&] {
                                                  result = nms_match_cpu_kernel<scalar_t>(dets, threshold);
                                                });
                                                return result;
}