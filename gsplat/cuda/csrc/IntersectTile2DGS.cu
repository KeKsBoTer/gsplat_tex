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

#if GSPLAT_BUILD_2DGS_PROJECTION

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAStream.h>
#    include <cooperative_groups.h>
#    include <cuda/std/functional>

#    include "Common.h"
#    include "Footprint2DGS.cuh"
#    include "Intersect.h"
#    include "MathUtils.h"

namespace gsplat
{
namespace cg = cooperative_groups;
using namespace footprint2dgs;

// Exact tile intersection for 2DGS / BBSplat primitives; see Footprint2DGS.cuh.

template<typename scalar_t>
__global__ void intersect_tile_2dgs_kernel(
    const int64_t count,
    const bool packed,
    const uint32_t N,
    const int64_t *__restrict__ image_ids,       // [nnz] optional
    const scalar_t *__restrict__ means2d,        // [..., N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,           // [..., N, 2] or [nnz, 2]
    const scalar_t *__restrict__ depths,         // [..., N] or [nnz]
    const scalar_t *__restrict__ ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const float *__restrict__ opacities,         // [..., N] or [nnz] (2DGS only)
    const float *__restrict__ uv_rects,          // [..., N, 4] or [nnz, 4] (BBSplat only)
    const int64_t *__restrict__ cum_tiles_per_gauss,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss,
    int64_t *__restrict__ isect_ids,
    int32_t *__restrict__ flatten_ids
)
{
    const int64_t idx     = cg::this_grid().thread_rank();
    const bool first_pass = cum_tiles_per_gauss == nullptr;
    if(idx >= count)
    {
        return;
    }

    const float2 radius = {(float)radii[idx * 2], (float)radii[idx * 2 + 1]};
    Footprint f;
    bool visible = radius.x > 0.f && radius.y > 0.f;
    if(visible)
    {
        float H[9];
#    pragma unroll
        for(int k = 0; k < 9; ++k)
        {
            H[k] = (float)ray_transforms[idx * 9 + k];
        }
        const float2 mean2d = {(float)means2d[idx * 2], (float)means2d[idx * 2 + 1]};
        visible             = make_footprint(
            H,
            mean2d,
            radius,
            opacities != nullptr ? opacities[idx] : 1.f,
            uv_rects != nullptr ? uv_rects + idx * 4 : nullptr,
            f
        );
    }
    if(!visible)
    {
        if(first_pass)
        {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    const float tile_size_f = (float)tile_size;
    const int2 rect_x       = tile_range(f.bbox_min.x, f.bbox_max.x, tile_size_f, tile_width);
    const int2 rect_y       = tile_range(f.bbox_min.y, f.bbox_max.y, tile_size_f, tile_height);

    // Walk the strips along the shorter side of the tile rectangle.
    const bool is_y       = rect_y.y - rect_y.x < rect_x.y - rect_x.x;
    const int2 outer      = is_y ? rect_y : rect_x;
    const int2 inner      = is_y ? rect_x : rect_y;
    const int32_t n_inner = is_y ? tile_width : tile_height;
    if(is_y)
    {
        transpose(f);
    }

    int64_t iid_enc      = 0;
    int64_t depth_id_enc = 0;
    int64_t cur_idx      = 0;
    if(!first_pass)
    {
        const int64_t iid = packed ? image_ids[idx] : idx / N;
        iid_enc           = iid << (32 + tile_n_bits);
        // Same depth key as intersect_tile_kernel (non-negative float bits).
        depth_id_enc      = __float_as_uint(static_cast<float>(depths[idx]));
        cur_idx           = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    }

    int32_t n = 0;
    for(int32_t a = outer.x; a < outer.y; ++a)
    {
        float vmin, vmax;
        if(!strip_extent(f, a * tile_size_f, (a + 1) * tile_size_f, vmin, vmax))
        {
            continue;
        }
        int2 range = tile_range(vmin, vmax, tile_size_f, n_inner);
        range      = {max(range.x, inner.x), min(range.y, inner.y)};
        if(first_pass)
        {
            n += max(range.y - range.x, 0);
            continue;
        }
        for(int32_t b = range.x; b < range.y; ++b)
        {
            const int64_t tile_id = is_y ? (int64_t)a * tile_width + b : (int64_t)b * tile_width + a;
            isect_ids[cur_idx]    = iid_enc | (tile_id << 32) | depth_id_enc;
            flatten_ids[cur_idx]  = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
    if(first_pass)
    {
        tiles_per_gauss[idx] = n;
    }
}

void launch_intersect_tile_2dgs_kernel(
    // inputs
    const at::Tensor means2d,                 // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                  // [..., N] or [nnz]
    const at::Tensor ray_transforms,          // [..., N, 3, 3] or [nnz, 3, 3]
    const at::optional<at::Tensor> opacities, // [..., N] or [nnz]
    const at::optional<at::Tensor> uv_rects,  // [..., N, 4] or [nnz, 4]
    const at::optional<at::Tensor> image_ids, // [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [..., N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [..., N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids      // [n_isects]
)
{
    const bool packed        = means2d.dim() == 2;
    const uint32_t N         = packed ? 0 : means2d.size(-2);
    const int64_t n_elements = means2d.numel() / 2;
    if(n_elements == 0)
    {
        return;
    }

    constexpr unsigned int threads = 256;
    const unsigned int blocks      = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));
    auto stream                    = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(
        means2d.scalar_type(),
        "intersect_tile_2dgs_kernel",
        [&]()
        {
            intersect_tile_2dgs_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                n_elements,
                packed,
                N,
                image_ids.has_value() ? image_ids.value().const_data_ptr<int64_t>() : nullptr,
                means2d.const_data_ptr<scalar_t>(),
                radii.const_data_ptr<int32_t>(),
                depths.const_data_ptr<scalar_t>(),
                ray_transforms.const_data_ptr<scalar_t>(),
                opacities.has_value() ? opacities.value().const_data_ptr<float>() : nullptr,
                uv_rects.has_value() ? uv_rects.value().const_data_ptr<float>() : nullptr,
                cum_tiles_per_gauss.has_value() ? cum_tiles_per_gauss.value().const_data_ptr<int64_t>() : nullptr,
                tile_size,
                tile_width,
                tile_height,
                tile_n_bits,
                tiles_per_gauss.has_value() ? tiles_per_gauss.value().data_ptr<int32_t>() : nullptr,
                isect_ids.has_value() ? isect_ids.value().data_ptr<int64_t>() : nullptr,
                flatten_ids.has_value() ? flatten_ids.value().data_ptr<int32_t>() : nullptr
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
}
} // namespace gsplat

#endif // GSPLAT_BUILD_2DGS_PROJECTION
