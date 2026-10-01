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

#pragma once

#include <cstdint>

namespace gsplat
{
// Bilinear texture lookup for BBSplat (billboard splatting) primitives.
//
// Each splat owns an S x S texture that covers its local plane coordinates
// (u, v) in [-1, 1]^2, i.e. the square spanned by +-scale_x * t_u and
// +-scale_y * t_v around the splat center. Sampling follows
// torch.nn.functional.grid_sample(align_corners=True, padding_mode="zeros"):
// texel centers of the outer row/column sit exactly at u, v = +-1, u maps to
// the column and v to the row, and texels outside the grid read as zero, so
// the splat fades to zero within one texel beyond its border.
struct BBSplatTexCoord
{
    int32_t x0; // column of the top-left texel of the 2x2 footprint
    int32_t y0; // row of the top-left texel of the 2x2 footprint
    float fx;   // fractional offset in x, in [0, 1)
    float fy;   // fractional offset in y, in [0, 1)
};

// Returns false when (u, v) lies so far outside the texture that the whole
// 2x2 footprint is out of bounds (the sample is exactly zero). This also
// guards the float->int conversion for rays at grazing angles, where u, v can
// be arbitrarily large.
inline __device__ bool bbsplat_tex_coord(const float u, const float v, const int32_t S, BBSplatTexCoord &tc)
{
    const float x = 0.5f * (u + 1.0f) * (float)(S - 1);
    const float y = 0.5f * (v + 1.0f) * (float)(S - 1);
    if(!(x > -1.0f && x < (float)S && y > -1.0f && y < (float)S))
    {
        return false;
    }
    const float x0 = floorf(x);
    const float y0 = floorf(y);
    tc             = {(int32_t)x0, (int32_t)y0, x - x0, y - y0};
    return true;
}

// d(texture x) / du == d(texture y) / dv
inline __device__ float bbsplat_tex_scale(const int32_t S)
{
    return 0.5f * (float)(S - 1);
}

// Calls fn(texel, w, dw_dx, dw_dy) for each in-bounds texel of the 2x2
// bilinear footprint, where texel = row * S + col, w is the bilinear weight
// and dw_dx / dw_dy are its derivatives w.r.t. the texture-space coordinates.
template<typename Fn>
inline __device__ void bbsplat_for_each_texel(const BBSplatTexCoord &tc, const int32_t S, Fn &&fn)
{
#pragma unroll
    for(int32_t dy = 0; dy < 2; ++dy)
    {
#pragma unroll
        for(int32_t dx = 0; dx < 2; ++dx)
        {
            const int32_t xi = tc.x0 + dx;
            const int32_t yi = tc.y0 + dy;
            if(xi < 0 || xi >= S || yi < 0 || yi >= S)
            {
                continue;
            }
            const float wx = dx ? tc.fx : 1.0f - tc.fx;
            const float wy = dy ? tc.fy : 1.0f - tc.fy;
            fn(yi * S + xi, wx * wy, (dx ? 1.0f : -1.0f) * wy, (dy ? 1.0f : -1.0f) * wx);
        }
    }
}
} // namespace gsplat
