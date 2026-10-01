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

#    include "BBSplatTexture.cuh"
#    include "Common.h"
#    include "Rasterization.h"
#    include "Dispatch.h"

namespace gsplat
{
using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;

namespace cg = cooperative_groups;

// BBSplat forward. Identical to rasterize_to_pixels_2dgs_fwd_kernel (same
// ray-splat intersection, compositing, normals, distortion and median depth)
// except for the per-pixel kernel:
//   alpha = min(MAX_ALPHA, opacity * texture_alpha(u, v))
//   color = colors + texture_color(u, v)   (texture added to the first
//                                           texture_channels channels)
// where (u, v) is the ray-splat intersection in splat-local coordinates and
// textures are sampled bilinearly (see BBSplatTexture.cuh). There is no
// Gaussian falloff and no screen-space low-pass filter.
template<uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_bbsplat_fwd_kernel(
    const uint32_t I,
    const int64_t n_isects,
    const scalar_t *__restrict__ ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const scalar_t *__restrict__ colors,         // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,      // [..., N] or [nnz]
    const scalar_t *__restrict__ normals,        // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ texture_alphas, // [M, S, S]
    const scalar_t *__restrict__ texture_colors, // [M, S, S, TC] (optional)
    const int32_t *__restrict__ texture_ids,     // [..., N] or [nnz], index into M
    const int32_t texture_size,                  // S
    const int32_t texture_channels,              // TC <= CDIM
    const scalar_t *__restrict__ backgrounds,    // [..., CDIM]
    const bool *__restrict__ masks,              // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int64_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // outputs
    scalar_t *__restrict__ render_colors,  // [..., image_height, image_width, CDIM]
    scalar_t *__restrict__ render_alphas,  // [..., image_height, image_width, 1]
    scalar_t *__restrict__ render_normals, // [..., image_height, image_width, 3]
    scalar_t *__restrict__ render_distort, // [..., image_height, image_width, 1]
    scalar_t *__restrict__ render_median,  // [..., image_height, image_width, 1]
    int32_t *__restrict__ last_ids,        // [..., image_height, image_width]
    int32_t *__restrict__ median_ids,      // [..., image_height, image_width]
    scalar_t *__restrict__ impacts         // [..., N] or [nnz] (optional): sum of alpha * T
)
{
    auto block       = cg::this_thread_block();
    int32_t image_id = block.group_index().x;
    int32_t tile_id  = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i       = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j       = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets   += image_id * tile_height * tile_width;
    render_colors  += image_id * image_height * image_width * CDIM;
    render_alphas  += image_id * image_height * image_width;
    last_ids       += image_id * image_height * image_width;
    render_normals += image_id * image_height * image_width * 3;
    render_distort += image_id * image_height * image_width;
    render_median  += image_id * image_height * image_width;
    median_ids     += image_id * image_height * image_width;

    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }
    if(masks != nullptr)
    {
        masks += image_id * tile_height * tile_width;
    }

    const float px       = (float)j + 0.5f;
    const float py       = (float)i + 0.5f;
    const int32_t pix_id = i * image_width + j;

    const bool inside = (i < image_height && j < image_width);
    bool done         = !inside;

    // Masked tile: write background / zeros so no output is left uninitialized.
    if(masks != nullptr && !masks[tile_id])
    {
        if(inside)
        {
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                render_colors[pix_id * CDIM + k] = backgrounds == nullptr ? 0.0f : backgrounds[k];
            }
            render_alphas[pix_id] = 0.0f;
            for(uint32_t k = 0; k < 3; ++k)
            {
                render_normals[pix_id * 3 + k] = 0.0f;
            }
            render_distort[pix_id] = 0.0f;
            render_median[pix_id]  = 0.0f;
            last_ids[pix_id]       = 0;
            median_ids[pix_id]     = 0;
        }
        return;
    }

    const int64_t range_start = tile_offsets[tile_id];
    const int64_t range_end
        = (image_id == I - 1) && (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const int64_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // Shared memory layout:
    // | gaussian id | texture id | opacity | u_M | v_M | w_M |
    extern __shared__ int s[];
    int32_t *id_batch     = (int32_t *)s;                                         // [block_size]
    int32_t *tex_id_batch = &id_batch[block_size];                                // [block_size]
    float *opac_batch     = reinterpret_cast<float *>(&tex_id_batch[block_size]); // [block_size]
    vec3 *u_Ms_batch      = reinterpret_cast<vec3 *>(&opac_batch[block_size]);    // [block_size]
    vec3 *v_Ms_batch      = reinterpret_cast<vec3 *>(&u_Ms_batch[block_size]);    // [block_size]
    vec3 *w_Ms_batch      = reinterpret_cast<vec3 *>(&v_Ms_batch[block_size]);    // [block_size]

    const int64_t tex_texels = (int64_t)texture_size * texture_size;

    float T                          = 1.0f;
    int32_t last_intersection_offset = 0;
    const uint32_t tr                = block.thread_rank();

    float distort         = 0.f;
    float accum_vis_depth = 0.f;

    float median_depth                 = 0.f;
    int32_t median_intersection_offset = 0;

    float pix_out[CDIM] = {0.f};
    float normal_out[3] = {0.f};
    for(int64_t b = 0; b < num_batches; ++b)
    {
        if(__syncthreads_count(done) >= block_size)
        {
            break;
        }

        const int64_t batch_offset = block_size * b;
        const int64_t batch_start  = range_start + batch_offset;
        const int64_t idx          = batch_start + tr;
        if(idx < range_end)
        {
            int32_t g        = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr]     = g;
            tex_id_batch[tr] = texture_ids[g];
            opac_batch[tr]   = opacities[g];
            u_Ms_batch[tr]   = {ray_transforms[g * 9], ray_transforms[g * 9 + 1], ray_transforms[g * 9 + 2]};
            v_Ms_batch[tr]   = {ray_transforms[g * 9 + 3], ray_transforms[g * 9 + 4], ray_transforms[g * 9 + 5]};
            w_Ms_batch[tr]   = {ray_transforms[g * 9 + 6], ray_transforms[g * 9 + 7], ray_transforms[g * 9 + 8]};
        }
        block.sync();

        const int64_t remaining   = range_end - batch_start;
        const uint32_t batch_size = static_cast<uint32_t>(remaining < block_size ? remaining : block_size);
        for(uint32_t t = 0; (t < batch_size) && !done; ++t)
        {
            // ray-splat intersection in splat-local (u, v), see the 2DGS kernel
            const vec3 h_u       = px * w_Ms_batch[t] - u_Ms_batch[t];
            const vec3 h_v       = py * w_Ms_batch[t] - v_Ms_batch[t];
            const vec3 ray_cross = glm::cross(h_u, h_v);
            if(ray_cross.z == 0.0)
            {
                continue;
            }
            const vec2 s = vec2(ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z);

            BBSplatTexCoord tc;
            if(!bbsplat_tex_coord(s.x, s.y, texture_size, tc))
            {
                continue;
            }
            const int64_t tex_base = tex_id_batch[t] * tex_texels;

            float tex_alpha = 0.f;
            bbsplat_for_each_texel(
                tc,
                texture_size,
                [&](int32_t texel, float w, float, float) { tex_alpha += w * texture_alphas[tex_base + texel]; }
            );

            const float alpha = min(MAX_ALPHA, opac_batch[t] * tex_alpha);
            if(alpha < ALPHA_THRESHOLD)
            {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if(next_T <= TRANSMITTANCE_THRESHOLD)
            {
                done = true;
                break;
            }

            int32_t g          = id_batch[t];
            const float vis    = alpha * T;
            const float *c_ptr = colors + g * CDIM;
#    pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                pix_out[k] += c_ptr[k] * vis;
            }
            if(texture_colors != nullptr)
            {
                bbsplat_for_each_texel(
                    tc,
                    texture_size,
                    [&](int32_t texel, float w, float, float)
                    {
                        const float *tex_ptr = texture_colors + (tex_base + texel) * texture_channels;
                        for(int32_t k = 0; k < texture_channels; ++k)
                        {
                            pix_out[k] += w * tex_ptr[k] * vis;
                        }
                    }
                );
            }

            if(impacts != nullptr)
            {
                atomicAdd(impacts + g, vis);
            }

            const float *n_ptr = normals + g * 3;
#    pragma unroll
            for(uint32_t k = 0; k < 3; ++k)
            {
                normal_out[k] += n_ptr[k] * vis;
            }

            // The last channel of colors is depth (untextured when
            // texture_channels < CDIM).
            const float depth = c_ptr[CDIM - 1];
            if(render_distort != nullptr)
            {
                const float distort_bi_0  = vis * depth * (1.0f - T);
                const float distort_bi_1  = vis * accum_vis_depth;
                distort                  += 2.0f * (distort_bi_0 - distort_bi_1);
                accum_vis_depth          += vis * depth;
            }

            if(T > 0.5)
            {
                median_depth               = depth;
                median_intersection_offset = static_cast<int32_t>(batch_offset + t);
            }

            last_intersection_offset = static_cast<int32_t>(batch_offset + t);

            T = next_T;
        }
    }
    if(inside)
    {
        render_alphas[pix_id] = 1.0f - T;
#    pragma unroll
        for(uint32_t k = 0; k < CDIM; ++k)
        {
            render_colors[pix_id * CDIM + k] = backgrounds == nullptr ? pix_out[k] : (pix_out[k] + T * backgrounds[k]);
        }
#    pragma unroll
        for(uint32_t k = 0; k < 3; ++k)
        {
            render_normals[pix_id * 3 + k] = normal_out[k];
        }
        last_ids[pix_id] = last_intersection_offset;
        if(render_distort != nullptr)
        {
            render_distort[pix_id] = distort;
        }
        render_median[pix_id] = median_depth;
        median_ids[pix_id]    = median_intersection_offset;
    }
}

void launch_rasterize_to_pixels_bbsplat_fwd_kernel(
    const at::Tensor ray_transforms,               // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,                       // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,                    // [..., N]  or [nnz]
    const at::Tensor normals,                      // [..., N, 3] or [nnz, 3]
    const at::Tensor texture_alphas,               // [M, S, S]
    const at::optional<at::Tensor> texture_colors, // [M, S, S, TC]
    const at::Tensor texture_ids,                  // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds,    // [..., channels]
    const at::optional<at::Tensor> masks,          // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor renders,              // [..., image_height, image_width, channels]
    at::Tensor alphas,               // [..., image_height, image_width, 1]
    at::Tensor render_normals,       // [..., image_height, image_width, 3]
    at::Tensor render_distort,       // [..., image_height, image_width, 1]
    at::Tensor render_median,        // [..., image_height, image_width, 1]
    at::Tensor last_ids,             // [..., image_height, image_width]
    at::Tensor median_ids,           // [..., image_height, image_width]
    at::optional<at::Tensor> impacts // [..., N] or [nnz]
)
{
    uint32_t I           = alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width  = tile_offsets.size(-1);
    int64_t n_isects     = flatten_ids.size(0);

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid    = {I, tile_height, tile_width};

    int64_t shmem_size
        = tile_size
        * tile_size
        * (sizeof(int32_t) + sizeof(int32_t) + sizeof(float) + sizeof(vec3) + sizeof(vec3) + sizeof(vec3));

    const int32_t channels = colors.size(-1);
    TORCH_CHECK_VALUE(
        SupportedChannels::contains(channels),
        "Unsupported number of color channels: ",
        channels,
        ". To add support, rebuild gsplat with this channel count included "
        "in -DGSPLAT_NUM_CHANNELS=... (see gsplat/cuda/csrc/Config.h)."
    );

    const int32_t texture_size     = texture_alphas.size(-1);
    const int32_t texture_channels = texture_colors.has_value() ? texture_colors.value().size(-1) : 0;

    auto launch_kernel = [&]<typename ChannelsT>()
    {
        constexpr uint32_t CDIM = ChannelsT::value;

        if(cudaFuncSetAttribute(
               rasterize_to_pixels_bbsplat_fwd_kernel<CDIM, float>,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               shmem_size
           )
           != cudaSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ", shmem_size, " bytes), try lowering tile_size."
            );
        }

        rasterize_to_pixels_bbsplat_fwd_kernel<CDIM, float>
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                I,
                n_isects,
                ray_transforms.const_data_ptr<float>(),
                colors.const_data_ptr<float>(),
                opacities.const_data_ptr<float>(),
                normals.const_data_ptr<float>(),
                texture_alphas.const_data_ptr<float>(),
                texture_colors.has_value() ? texture_colors.value().const_data_ptr<float>() : nullptr,
                texture_ids.const_data_ptr<int32_t>(),
                texture_size,
                texture_channels,
                backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
                masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.const_data_ptr<int64_t>(),
                flatten_ids.const_data_ptr<int32_t>(),
                renders.data_ptr<float>(),
                alphas.data_ptr<float>(),
                render_normals.data_ptr<float>(),
                render_distort.data_ptr<float>(),
                render_median.data_ptr<float>(),
                last_ids.data_ptr<int32_t>(),
                median_ids.data_ptr<int32_t>(),
                impacts.has_value() ? impacts.value().data_ptr<float>() : nullptr
            );
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}
} // namespace gsplat

#endif
