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
#    include <ATen/cuda/Atomic.cuh>
#    include <c10/cuda/CUDAStream.h>
#    include <cooperative_groups.h>

#    include "BBSplatTexture.cuh"
#    include "Common.h"
#    include "Rasterization.h"
#    include "Utils.cuh"
#    include "Dispatch.h"

namespace gsplat
{
using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;

namespace cg = cooperative_groups;

// Backward of rasterize_to_pixels_bbsplat_fwd_kernel. Follows the 2DGS
// backward (back-to-front replay with T recovered from the final
// transmittance); the per-pixel kernel gradient differs:
//   d alpha / d s   = opacity * d texture_alpha / d s
//   d color / d s   = d texture_color / d s
// and both textures receive bilinearly splatted gradients via atomics.
template<uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_bbsplat_bwd_kernel(
    const uint32_t I,
    const int64_t n_isects,
    // fwd inputs
    const scalar_t *__restrict__ ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const scalar_t *__restrict__ colors,         // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ normals,        // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ opacities,      // [..., N] or [nnz]
    const scalar_t *__restrict__ texture_alphas, // [M, S, S]
    const scalar_t *__restrict__ texture_colors, // [M, S, S, TC] (optional)
    const int32_t *__restrict__ texture_ids,     // [..., N] or [nnz]
    const int32_t texture_size,
    const int32_t texture_channels,
    const scalar_t *__restrict__ backgrounds, // [..., CDIM]
    const bool *__restrict__ masks,           // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int64_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const scalar_t *__restrict__ render_colors, // [..., image_height, image_width, CDIM]
    const scalar_t *__restrict__ render_alphas, // [..., image_height, image_width, 1]
    const int32_t *__restrict__ last_ids,       // [..., image_height, image_width]
    const int32_t *__restrict__ median_ids,     // [..., image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors,  // [..., image_height, image_width, CDIM]
    const scalar_t *__restrict__ v_render_alphas,  // [..., image_height, image_width, 1]
    const scalar_t *__restrict__ v_render_normals, // [..., image_height, image_width, 3]
    const scalar_t *__restrict__ v_render_distort, // [..., image_height, image_width, 1]
    const scalar_t *__restrict__ v_render_median,  // [..., image_height, image_width, 1]
    // grad inputs
    scalar_t *__restrict__ v_ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    scalar_t *__restrict__ v_colors,         // [..., N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities,      // [..., N] or [nnz]
    scalar_t *__restrict__ v_normals,        // [..., N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_densify,        // [..., N, 2] or [nnz, 2]
    scalar_t *__restrict__ v_texture_alphas, // [M, S, S]
    scalar_t *__restrict__ v_texture_colors  // [M, S, S, TC] (optional)
)
{
    auto block        = cg::this_thread_block();
    uint32_t image_id = block.group_index().x;
    uint32_t tile_id  = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i        = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j        = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets  += image_id * tile_height * tile_width;
    render_alphas += image_id * image_height * image_width;
    render_colors += image_id * image_height * image_width * CDIM;

    last_ids   += image_id * image_height * image_width;
    median_ids += image_id * image_height * image_width;

    v_render_colors  += image_id * image_height * image_width * CDIM;
    v_render_alphas  += image_id * image_height * image_width;
    v_render_normals += image_id * image_height * image_width * 3;
    v_render_median  += image_id * image_height * image_width;

    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }
    if(masks != nullptr)
    {
        masks += image_id * tile_height * tile_width;
    }
    if(v_render_distort != nullptr)
    {
        v_render_distort += image_id * image_height * image_width;
    }

    if(masks != nullptr && !masks[tile_id])
    {
        return;
    }

    const float px       = (float)j + 0.5f;
    const float py       = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    const bool inside = (i < image_height && j < image_width);

    const int64_t range_start = tile_offsets[tile_id];
    const int64_t range_end
        = (image_id == I - 1) && (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size       = block.size();
    const int64_t num_intersections = range_end - range_start;
    const int64_t num_batches       = (num_intersections + block_size - 1) / block_size;

    // Shared memory layout:
    // | gaussian id | texture id | opacity | u_M | v_M | w_M | rgb | normal |
    extern __shared__ int s[];
    int32_t *id_batch     = (int32_t *)s;
    int32_t *tex_id_batch = &id_batch[block_size];
    float *opac_batch     = reinterpret_cast<float *>(&tex_id_batch[block_size]);
    vec3 *u_Ms_batch      = reinterpret_cast<vec3 *>(&opac_batch[block_size]);
    vec3 *v_Ms_batch      = reinterpret_cast<vec3 *>(&u_Ms_batch[block_size]);
    vec3 *w_Ms_batch      = reinterpret_cast<vec3 *>(&v_Ms_batch[block_size]);
    float *rgbs_batch     = (float *)&w_Ms_batch[block_size]; // [block_size * CDIM]
    float *normals_batch  = &rgbs_batch[block_size * CDIM];   // [block_size * 3]

    const int64_t tex_texels = (int64_t)texture_size * texture_size;
    const float tex_scale    = bbsplat_tex_scale(texture_size);

    // this is the T AFTER the last gaussian in this pixel
    const float T_final = 1.0f - render_alphas[pix_id];
    float T             = T_final;

    // the contribution from gaussians behind the current one
    float buffer[CDIM]      = {0.f};
    float buffer_normals[3] = {0.f};

    const int32_t bin_final  = inside ? last_ids[pix_id] : 0;
    const int32_t median_idx = inside ? median_ids[pix_id] : 0;

    float v_render_c[CDIM];
#    pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];
    float v_render_n[3];
#    pragma unroll
    for(uint32_t k = 0; k < 3; ++k)
    {
        v_render_n[k] = v_render_normals[pix_id * 3 + k];
    }

    float v_distort = 0.f;
    float accum_d, accum_w;
    float accum_d_buffer, accum_w_buffer, distort_buffer;
    if(v_render_distort != nullptr)
    {
        v_distort      = v_render_distort[pix_id];
        // last channel of render_colors is accumulated depth
        accum_d_buffer = render_colors[pix_id * CDIM + CDIM - 1];
        accum_d        = accum_d_buffer;
        accum_w_buffer = render_alphas[pix_id];
        accum_w        = accum_w_buffer;
        distort_buffer = 0.f;
    }

    const float v_median = v_render_median[pix_id];

    const uint32_t tr              = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final   = cg::reduce(warp, bin_final, cg::greater<int>());

    for(int64_t b = 0; b < num_batches; ++b)
    {
        block.sync();

        // fetch gaussians back to front
        const int64_t batch_end_offset = num_intersections - 1 - block_size * b;
        const int64_t remaining        = batch_end_offset + 1;
        const uint32_t batch_size      = static_cast<uint32_t>(remaining < block_size ? remaining : block_size);
        const int64_t idx              = range_start + batch_end_offset - tr;
        if(idx >= range_start)
        {
            int32_t g        = flatten_ids[idx];
            id_batch[tr]     = g;
            tex_id_batch[tr] = texture_ids[g];
            opac_batch[tr]   = opacities[g];
            u_Ms_batch[tr]   = {ray_transforms[g * 9], ray_transforms[g * 9 + 1], ray_transforms[g * 9 + 2]};
            v_Ms_batch[tr]   = {ray_transforms[g * 9 + 3], ray_transforms[g * 9 + 4], ray_transforms[g * 9 + 5]};
            w_Ms_batch[tr]   = {ray_transforms[g * 9 + 6], ray_transforms[g * 9 + 7], ray_transforms[g * 9 + 8]};
#    pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
#    pragma unroll
            for(uint32_t k = 0; k < 3; ++k)
            {
                normals_batch[tr * 3 + k] = normals[g * 3 + k];
            }
        }
        block.sync();

        const int64_t first_t64 = batch_end_offset > warp_bin_final ? batch_end_offset - warp_bin_final : 0;
        const uint32_t first_t  = first_t64 < batch_size ? static_cast<uint32_t>(first_t64) : batch_size;
        for(uint32_t t = first_t; t < batch_size; ++t)
        {
            bool valid = inside;
            if(batch_end_offset - t > bin_final)
            {
                valid = false;
            }

            // Replay the forward pass for the t-th primitive.
            float alpha     = 0.f;
            float opac      = 0.f;
            float tex_alpha = 0.f;
            vec2 s;
            vec3 h_u, h_v, ray_cross, w_M;
            BBSplatTexCoord tc;
            int64_t tex_base = 0;
            if(valid)
            {
                opac      = opac_batch[t];
                w_M       = w_Ms_batch[t];
                h_u       = px * w_M - u_Ms_batch[t];
                h_v       = py * w_M - v_Ms_batch[t];
                ray_cross = glm::cross(h_u, h_v);
                if(ray_cross.z == 0.0)
                {
                    valid = false;
                }
                else
                {
                    s = {ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z};
                    if(!bbsplat_tex_coord(s.x, s.y, texture_size, tc))
                    {
                        valid = false;
                    }
                    else
                    {
                        tex_base = tex_id_batch[t] * tex_texels;
                        bbsplat_for_each_texel(
                            tc,
                            texture_size,
                            [&](int32_t texel, float w, float, float)
                            { tex_alpha += w * texture_alphas[tex_base + texel]; }
                        );
                        alpha = min(MAX_ALPHA, opac * tex_alpha);
                        if(alpha < ALPHA_THRESHOLD)
                        {
                            valid = false;
                        }
                    }
                }
            }

            if(!warp.any(valid))
            {
                continue;
            }

            float v_rgb_local[CDIM] = {0.f};
            float v_normal_local[3] = {0.f};
            vec3 v_u_M_local        = {0.f, 0.f, 0.f};
            vec3 v_v_M_local        = {0.f, 0.f, 0.f};
            vec3 v_w_M_local        = {0.f, 0.f, 0.f};
            vec2 v_densify_local    = {0.f, 0.f};
            float v_opacity_local   = 0.f;

            if(valid)
            {
                // per-pixel color: colors + texture_color(s)
                float c_full[CDIM];
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    c_full[k] = rgbs_batch[t * CDIM + k];
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
                                c_full[k] += w * tex_ptr[k];
                            }
                        }
                    );
                }

                if(batch_end_offset - t == median_idx)
                {
                    v_rgb_local[CDIM - 1] += v_median;
                }

                const float ra   = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
                T               *= ra;
                const float fac  = alpha * T;
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    v_rgb_local[k] += fac * v_render_c[k];
                }

                float v_alpha = 0.f;
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    v_alpha += (c_full[k] * T - buffer[k] * ra) * v_render_c[k];
                }
#    pragma unroll
                for(uint32_t k = 0; k < 3; ++k)
                {
                    v_normal_local[k] = fac * v_render_n[k];
                }
                for(uint32_t k = 0; k < 3; ++k)
                {
                    v_alpha += (normals_batch[t * 3 + k] * T - buffer_normals[k] * ra) * v_render_n[k];
                }

                v_alpha += T_final * ra * v_render_a;

                if(backgrounds != nullptr)
                {
                    float accum = 0.f;
#    pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                    {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if(v_render_distort != nullptr)
                {
                    // last channel of colors is depth
                    const float depth = rgbs_batch[t * CDIM + CDIM - 1];
                    const float dl_dw
                        = 2.0f * (2.0f * (depth * accum_w_buffer - accum_d_buffer) + (accum_d - depth * accum_w));
                    v_alpha               += (dl_dw * T - distort_buffer * ra) * v_distort;
                    accum_d_buffer        -= fac * depth;
                    accum_w_buffer        -= fac;
                    distort_buffer        += dl_dw * fac;
                    v_rgb_local[CDIM - 1] += 2.0f * fac * (2.0f - 2.0f * T - accum_w + fac) * v_distort;
                }

                // gradient w.r.t. the intersection point s (texture space
                // derivatives are scaled by d(texture coord) / ds)
                vec2 v_s = {0.f, 0.f};

                if(texture_colors != nullptr)
                {
                    bbsplat_for_each_texel(
                        tc,
                        texture_size,
                        [&](int32_t texel, float w, float dw_dx, float dw_dy)
                        {
                            const int64_t off    = (tex_base + texel) * texture_channels;
                            const float *tex_ptr = texture_colors + off;
                            float v_w            = 0.f;
                            for(int32_t k = 0; k < texture_channels; ++k)
                            {
                                const float v_c = fac * v_render_c[k];
                                gpuAtomicAdd(v_texture_colors + off + k, w * v_c);
                                v_w += v_c * tex_ptr[k];
                            }
                            v_s.x += v_w * dw_dx * tex_scale;
                            v_s.y += v_w * dw_dy * tex_scale;
                        }
                    );
                }

                // alpha is clamped to MAX_ALPHA (zero gradient) above this
                if(opac * tex_alpha <= MAX_ALPHA)
                {
                    const float v_tex_alpha = opac * v_alpha;
                    v_opacity_local         = tex_alpha * v_alpha;
                    bbsplat_for_each_texel(
                        tc,
                        texture_size,
                        [&](int32_t texel, float w, float dw_dx, float dw_dy)
                        {
                            gpuAtomicAdd(v_texture_alphas + tex_base + texel, w * v_tex_alpha);
                            const float v_w  = v_tex_alpha * texture_alphas[tex_base + texel];
                            v_s.x           += v_w * dw_dx * tex_scale;
                            v_s.y           += v_w * dw_dy * tex_scale;
                        }
                    );
                }

                // backward through the ray-splat intersection (see the 2DGS
                // backward kernel)
                const float v_sx_pz    = v_s.x / ray_cross.z;
                const float v_sy_pz    = v_s.y / ray_cross.z;
                const vec3 v_ray_cross = {v_sx_pz, v_sy_pz, -(v_sx_pz * s.x + v_sy_pz * s.y)};
                const vec3 v_h_u       = glm::cross(h_v, v_ray_cross);
                const vec3 v_h_v       = glm::cross(v_ray_cross, h_u);
                v_u_M_local            = -v_h_u;
                v_v_M_local            = -v_h_v;
                v_w_M_local            = px * v_h_u + py * v_h_v;

                // densification gradient: depth-scaled ray-transform gradient
                v_densify_local = {v_u_M_local.z * w_M.z, v_v_M_local.z * w_M.z};

#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    buffer[k] += c_full[k] * fac;
                }
#    pragma unroll
                for(uint32_t k = 0; k < 3; ++k)
                {
                    buffer_normals[k] += normals_batch[t * 3 + k] * fac;
                }
            }

            warpSum<CDIM>(v_rgb_local, warp);
            warpSum<3>(v_normal_local, warp);
            warpSum(v_u_M_local, warp);
            warpSum(v_v_M_local, warp);
            warpSum(v_w_M_local, warp);
            warpSum(v_densify_local, warp);
            warpSum(v_opacity_local, warp);
            int32_t g = id_batch[t];

            if(warp.thread_rank() == 0)
            {
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_normal_ptr = (float *)(v_normals) + 3 * g;
#    pragma unroll
                for(uint32_t k = 0; k < 3; ++k)
                {
                    gpuAtomicAdd(v_normal_ptr + k, v_normal_local[k]);
                }

                float *v_ray_transforms_ptr = (float *)(v_ray_transforms) + 9 * g;
                gpuAtomicAdd(v_ray_transforms_ptr, v_u_M_local.x);
                gpuAtomicAdd(v_ray_transforms_ptr + 1, v_u_M_local.y);
                gpuAtomicAdd(v_ray_transforms_ptr + 2, v_u_M_local.z);
                gpuAtomicAdd(v_ray_transforms_ptr + 3, v_v_M_local.x);
                gpuAtomicAdd(v_ray_transforms_ptr + 4, v_v_M_local.y);
                gpuAtomicAdd(v_ray_transforms_ptr + 5, v_v_M_local.z);
                gpuAtomicAdd(v_ray_transforms_ptr + 6, v_w_M_local.x);
                gpuAtomicAdd(v_ray_transforms_ptr + 7, v_w_M_local.y);
                gpuAtomicAdd(v_ray_transforms_ptr + 8, v_w_M_local.z);

                float *v_densify_ptr = (float *)(v_densify) + 2 * g;
                gpuAtomicAdd(v_densify_ptr + 0, v_densify_local.x);
                gpuAtomicAdd(v_densify_ptr + 1, v_densify_local.y);

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

void launch_rasterize_to_pixels_bbsplat_bwd_kernel(
    const at::Tensor ray_transforms,               // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,                       // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,                    // [..., N] or [nnz]
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
    // forward outputs
    const at::Tensor render_colors, // [..., image_height, image_width, channels]
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    const at::Tensor median_ids,    // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors,  // [..., image_height, image_width, channels]
    const at::Tensor v_render_alphas,  // [..., image_height, image_width, 1]
    const at::Tensor v_render_normals, // [..., image_height, image_width, 3]
    const at::Tensor v_render_distort, // [..., image_height, image_width, 1]
    const at::Tensor v_render_median,  // [..., image_height, image_width, 1]
    // outputs
    at::Tensor v_ray_transforms,              // [..., N, 3, 3] or [nnz, 3, 3]
    at::Tensor v_colors,                      // [..., N, channels] or [nnz, channels]
    at::Tensor v_opacities,                   // [..., N] or [nnz]
    at::Tensor v_normals,                     // [..., N, 3] or [nnz, 3]
    at::Tensor v_densify,                     // [..., N, 2] or [nnz, 2]
    at::Tensor v_texture_alphas,              // [M, S, S]
    at::optional<at::Tensor> v_texture_colors // [M, S, S, TC]
)
{
    uint32_t I           = render_alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width  = tile_offsets.size(-1);
    int64_t n_isects     = flatten_ids.size(0);

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid    = {I, tile_height, tile_width};

    if(n_isects == 0)
    {
        return;
    }

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

        int64_t shmem_size = tile_size
                           * tile_size
                           * (sizeof(int32_t)
                              + sizeof(int32_t)
                              + sizeof(float)
                              + sizeof(vec3)
                              + sizeof(vec3)
                              + sizeof(vec3)
                              + sizeof(float) * CDIM
                              + sizeof(float) * 3);

        if(cudaFuncSetAttribute(
               rasterize_to_pixels_bbsplat_bwd_kernel<CDIM, float>,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               shmem_size
           )
           != cudaSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ", shmem_size, " bytes), try lowering tile_size."
            );
        }

        rasterize_to_pixels_bbsplat_bwd_kernel<CDIM, float>
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                I,
                n_isects,
                ray_transforms.const_data_ptr<float>(),
                colors.const_data_ptr<float>(),
                normals.const_data_ptr<float>(),
                opacities.const_data_ptr<float>(),
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
                render_colors.const_data_ptr<float>(),
                render_alphas.const_data_ptr<float>(),
                last_ids.const_data_ptr<int32_t>(),
                median_ids.const_data_ptr<int32_t>(),
                v_render_colors.const_data_ptr<float>(),
                v_render_alphas.const_data_ptr<float>(),
                v_render_normals.const_data_ptr<float>(),
                v_render_distort.const_data_ptr<float>(),
                v_render_median.const_data_ptr<float>(),
                v_ray_transforms.data_ptr<float>(),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                v_normals.data_ptr<float>(),
                v_densify.data_ptr<float>(),
                v_texture_alphas.data_ptr<float>(),
                v_texture_colors.has_value() ? v_texture_colors.value().data_ptr<float>() : nullptr
            );
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}
} // namespace gsplat

#endif
