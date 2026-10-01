/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Config.h"

#if GSPLAT_BUILD_BBSPLAT

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAStream.h>
#    include <cooperative_groups.h>
#    include <cooperative_groups/reduce.h>
#    include <cuda/std/functional>

#    include "Common.h"
#    include "Rasterization.h"

namespace gsplat
{
namespace cg = cooperative_groups;

// One warp per texture: each lane scans a strided subset of the S x S texels
// (coalesced across the warp), tracks the column / row span of the texels with
// opacity * alpha >= ALPHA_THRESHOLD, and the warp reduces the spans. See
// bbsplat_uv_rects in _wrapper.py for the derivation of the rectangle.
template<typename scalar_t>
__global__ void bbsplat_uv_rects_kernel(
    const int64_t M,
    const int32_t S,
    const scalar_t *__restrict__ texture_alphas, // [M, S, S]
    const scalar_t *__restrict__ opacities,      // [M] optional
    float *__restrict__ uv_rects                 // [M, 4]
)
{
    auto warp       = cg::tiled_partition<32>(cg::this_thread_block());
    const int64_t m = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if(m >= M)
    {
        return;
    }

    // Same threshold as the PyTorch reference, including the slack that keeps
    // texels which round up to ALPHA_THRESHOLD in the rasterizer.
    float thr = ALPHA_THRESHOLD * (1.0f - 1e-4f);
    if(opacities != nullptr)
    {
        thr /= fmaxf((float)opacities[m], 1e-30f);
    }

    const int32_t n_texels = S * S;
    const scalar_t *tex    = texture_alphas + m * n_texels;
    int32_t x_lo = S, x_hi = -1, y_lo = S, y_hi = -1;
    for(int32_t i = warp.thread_rank(); i < n_texels; i += 32)
    {
        if((float)tex[i] >= thr)
        {
            const int32_t y = i / S;
            const int32_t x = i - y * S;
            x_lo            = min(x_lo, x);
            x_hi            = max(x_hi, x);
            y_lo            = min(y_lo, y);
            y_hi            = max(y_hi, y);
        }
    }
    x_lo = cg::reduce(warp, x_lo, cg::less<int32_t>());
    x_hi = cg::reduce(warp, x_hi, cg::greater<int32_t>());
    y_lo = cg::reduce(warp, y_lo, cg::less<int32_t>());
    y_hi = cg::reduce(warp, y_hi, cg::greater<int32_t>());

    if(warp.thread_rank() == 0)
    {
        float4 rect;
        if(x_hi < 0)
        {
            // no texel reaches the threshold: empty rectangle (culled)
            rect = {1.0f, -1.0f, 1.0f, -1.0f};
        }
        else
        {
            // texel x sits at u = 2x / (S - 1) - 1; its bilinear support is (x - 1, x + 1)
            const float scale = 2.0f / (float)(S - 1);
            rect              = {
                (float)(x_lo - 1) * scale - 1.0f,
                (float)(x_hi + 1) * scale - 1.0f,
                (float)(y_lo - 1) * scale - 1.0f,
                (float)(y_hi + 1) * scale - 1.0f,
            };
        }
        reinterpret_cast<float4 *>(uv_rects)[m] = rect;
    }
}

void launch_bbsplat_uv_rects_kernel(
    const at::Tensor texture_alphas,          // [M, S, S]
    const at::optional<at::Tensor> opacities, // [M]
    at::Tensor uv_rects                       // [M, 4] float32
)
{
    const int64_t M = texture_alphas.size(0);
    const int32_t S = texture_alphas.size(-1);
    if(M == 0)
    {
        return;
    }

    constexpr unsigned int threads = 256; // 8 textures per block
    const unsigned int blocks      = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(M, threads / 32));
    auto stream                    = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(
        texture_alphas.scalar_type(),
        "bbsplat_uv_rects_kernel",
        [&]()
        {
            bbsplat_uv_rects_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                M,
                S,
                texture_alphas.const_data_ptr<scalar_t>(),
                opacities.has_value() ? opacities.value().const_data_ptr<scalar_t>() : nullptr,
                uv_rects.data_ptr<float>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
}
} // namespace gsplat

#endif // GSPLAT_BUILD_BBSPLAT
