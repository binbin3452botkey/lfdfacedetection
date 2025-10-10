#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <vector>


#define DIVUP(a, b) (((a) + (b) - 1) / (b))

constexpr int threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(const float *a, const float *b) {
  const float left = max(a[0], b[0]);
  const float right = min(a[2], b[2]);
  const float top = max(a[1], b[1]);
  const float bottom = min(a[3], b[3]);
  
  const float width = max(right - left, 0.f);
  const float height = max(bottom - top, 0.f);
  const float interS = width * height;
  
  const float Sa = (a[2] - a[0]) * (a[3] - a[1]);
  const float Sb = (b[2] - b[0]) * (b[3] - b[1]);
  
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(
    const int n_boxes,
    const float nms_overlap_thresh,
    const float *dev_boxes,
    unsigned long long *dev_mask) {
  
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  
  const int row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 5];
  
  if (threadIdx.x < col_size) {
    const int offset = (threadsPerBlock * col_start + threadIdx.x) * 5;
    block_boxes[threadIdx.x * 5 + 0] = dev_boxes[offset + 0];
    block_boxes[threadIdx.x * 5 + 1] = dev_boxes[offset + 1];
    block_boxes[threadIdx.x * 5 + 2] = dev_boxes[offset + 2];
    block_boxes[threadIdx.x * 5 + 3] = dev_boxes[offset + 3];
    block_boxes[threadIdx.x * 5 + 4] = dev_boxes[offset + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    
    unsigned long long t = 0;
    const int start = (row_start == col_start) ? threadIdx.x + 1 : 0;
    
    for (int i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

at::Tensor nms_cuda_forward(
    const at::Tensor boxes,
    float nms_overlap_thresh) {
  
  at::DeviceGuard guard(boxes.device());
  TORCH_CHECK(boxes.is_cuda(), "boxes must be a CUDA tensor");

  auto scores = boxes.select(1, 4);
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t);

  const int boxes_num = boxes.size(0);
  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  auto boxes_dev = boxes_sorted.contiguous().data_ptr<float>();
  
  size_t mask_size = boxes_num * col_blocks * sizeof(unsigned long long);
  auto mask_dev = static_cast<unsigned long long*>(
      c10::cuda::CUDACachingAllocator::raw_alloc(mask_size));

  const dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
                    DIVUP(boxes_num, threadsPerBlock));
  const dim3 threads(threadsPerBlock);
  
  nms_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      boxes_num, nms_overlap_thresh, boxes_dev, mask_dev);
  
  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  AT_CUDA_CHECK(cudaMemcpyAsync(
      mask_host.data(),
      mask_dev,
      mask_size,
      cudaMemcpyDeviceToHost,
      c10::cuda::getCurrentCUDAStream()));
  
  AT_CUDA_CHECK(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream()));

  std::vector<unsigned long long> remv(col_blocks, 0);
  auto keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
  auto keep_out = keep.data_ptr<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    const int nblock = i / threadsPerBlock;
    const int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      auto p = mask_host.data() + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  c10::cuda::CUDACachingAllocator::raw_delete(mask_dev);
  
  return order_t.index({
      keep.narrow(0, 0, num_to_keep).to(
          order_t.device(), keep.scalar_type())});
}
