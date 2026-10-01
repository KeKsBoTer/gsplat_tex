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

#include "Common.h"
#include "Rasterization.h" // FILTER_INV_SQUARE_2DGS

namespace gsplat
{
// ============================================================
// Exact tile intersection for 2DGS / BBSplat primitives.
//
// Both primitives live on a plane with local coordinates q = (u, v, 1), and
// the ray transform H (rows u_M, v_M, w_M, see the 2DGS rasterizer) maps q to
// homogeneous pixel coordinates. Whenever the footprint lies in front of the
// camera (w_M . q > 0 on all of it), H is a projective map that sends:
//
// - 2DGS: the disk u^2 + v^2 <= t (alpha >= ALPHA_THRESHOLD) to an ellipse.
//   The rasterizer also accepts pixels within the screen-space low-pass disk
//   FILTER_INV_SQUARE_2DGS * |p - mean2d|^2 <= t, so the footprint is the
//   union of the ellipse and that disk.
// - BBSplat: the texture rectangle [u0, u1] x [v0, v1] to a convex quad.
//
// Each footprint is convex (or a union of two convex sets for 2DGS), so it
// overlaps a tile iff its extent along a tile column strip overlaps the tile.
// The kernel walks the strips of the shorter bounding-box side and emits the
// tiles in each strip's extent. When the footprint crosses the camera plane
// its image is unbounded, and the kernel falls back to the projection's
// axis-aligned box (means2d +- radii).
// ============================================================

//
// The projection (Projection2DGS*.cu) uses the same footprint to cull and to
// size radii, and the tile intersection (IntersectTile2DGS.cu) to assign tiles.
namespace footprint2dgs
{
    enum class FootprintKind
    {
        Box,
        Ellipse,
        Quad
    };

    struct Footprint
    {
        FootprintKind kind;
        float2 bbox_min;
        float2 bbox_max;
        // Ellipse: {c + d : d^T S^-1 d <= 1}, plus the disk |p - m|^2 <= r2.
        float2 c;
        float sxx, sxy, syy;
        float2 m;
        float r2;
        // Quad: vertices in boundary order.
        float2 p[4];
    };

    inline __device__ float2 swap_xy(const float2 a)
    {
        return {a.y, a.x};
    }

    // Swaps x and y of the footprint so that the strip walk below always
    // iterates over x.
    inline __device__ void transpose(Footprint &f)
    {
        f.bbox_min      = swap_xy(f.bbox_min);
        f.bbox_max      = swap_xy(f.bbox_max);
        f.c             = swap_xy(f.c);
        f.m             = swap_xy(f.m);
        const float sxx = f.sxx;
        f.sxx           = f.syy;
        f.syy           = sxx;
#pragma unroll
        for(int k = 0; k < 4; ++k)
        {
            f.p[k] = swap_xy(f.p[k]);
        }
    }

    // y extent of the footprint within the strip x0 <= x <= x1. Returns false
    // if the footprint does not reach into the strip.
    inline __device__ bool strip_extent(const Footprint &f, const float x0, const float x1, float &ymin, float &ymax)
    {
        ymin = INFINITY;
        ymax = -INFINITY;
        if(f.kind == FootprintKind::Box)
        {
            ymin = f.bbox_min.y;
            ymax = f.bbox_max.y;
        }
        else if(f.kind == FootprintKind::Ellipse)
        {
            // Ellipse: at x = c.x + h, y spans mu(h) +- sqrt(var * (1 - h^2 / sxx))
            // with mu(h) = c.y + sxy / sxx * h and var = det(S) / sxx (the
            // conditional of a Gaussian with covariance S). The upper boundary
            // is concave in h and peaks at h_top = sxy / sqrt(syy), the lower
            // one is convex with its minimum at -h_top, so their extrema over
            // the strip are attained at those points clamped to the strip.
            const float ex = sqrtf(f.sxx);
            const float a  = fmaxf(x0 - f.c.x, -ex);
            const float b  = fminf(x1 - f.c.x, ex);
            if(a <= b)
            {
                const float slope = f.sxy / f.sxx;
                const float var   = fmaxf(f.sxx * f.syy - f.sxy * f.sxy, 0.f) / f.sxx;
                const float h_top = f.sxy / sqrtf(f.syy);
                const float h_hi  = fminf(fmaxf(h_top, a), b);
                const float h_lo  = fminf(fmaxf(-h_top, a), b);
                const float hw_hi = sqrtf(fmaxf(var * (1.f - h_hi * h_hi / f.sxx), 0.f));
                const float hw_lo = sqrtf(fmaxf(var * (1.f - h_lo * h_lo / f.sxx), 0.f));
                ymax              = f.c.y + slope * h_hi + hw_hi;
                ymin              = f.c.y + slope * h_lo - hw_lo;
            }
            // Low-pass disk: the strip point closest to the center decides.
            if(f.r2 > 0.f)
            {
                const float dx = fminf(fmaxf(f.m.x, x0), x1) - f.m.x;
                const float rr = f.r2 - dx * dx;
                if(rr >= 0.f)
                {
                    const float hw = sqrtf(rr);
                    ymin           = fminf(ymin, f.m.y - hw);
                    ymax           = fmaxf(ymax, f.m.y + hw);
                }
            }
        }
        else
        {
            // Convex polygon: its intersection with the strip is spanned by
            // the vertices inside the strip and the edge crossings of x0, x1.
#pragma unroll
            for(int k = 0; k < 4; ++k)
            {
                const float2 pa = f.p[k];
                const float2 pb = f.p[(k + 1) & 3];
                if(pa.x >= x0 && pa.x <= x1)
                {
                    ymin = fminf(ymin, pa.y);
                    ymax = fmaxf(ymax, pa.y);
                }
                const float lo = fminf(pa.x, pb.x);
                const float hi = fmaxf(pa.x, pb.x);
                if(lo < hi)
                {
                    const float dydx = (pb.y - pa.y) / (pb.x - pa.x);
                    if(lo < x0 && x0 < hi)
                    {
                        const float y = pa.y + (x0 - pa.x) * dydx;
                        ymin          = fminf(ymin, y);
                        ymax          = fmaxf(ymax, y);
                    }
                    if(lo < x1 && x1 < hi)
                    {
                        const float y = pa.y + (x1 - pa.x) * dydx;
                        ymin          = fminf(ymin, y);
                        ymax          = fmaxf(ymax, y);
                    }
                }
            }
        }
        return ymin <= ymax;
    }

    // Tile index range [lo, hi) covering [vmin, vmax], clamped to [0, n].
    inline __device__ int2 tile_range(const float vmin, const float vmax, const float tile_size, const int32_t n)
    {
        // Clamp in float first: the footprint of a near-degenerate primitive
        // can reach far beyond the int range.
        const float lo = fminf(fmaxf(floorf(vmin / tile_size), 0.f), (float)n);
        const float hi = fminf(fmaxf(floorf(vmax / tile_size) + 1.f, 0.f), (float)n);
        return {(int32_t)lo, (int32_t)hi};
    }

    // Builds the footprint of the 2DGS ellipse (+ low-pass disk) or of the
    // BBSplat quad. Returns false if nothing can pass ALPHA_THRESHOLD.
    inline __device__ bool make_footprint(
        const float H[9],    // ray transform, rows u_M, v_M, w_M
        const float2 mean2d, // projected center (low-pass disk center)
        const float2 radius, // projection radii (box fallback)
        const float opacity,
        const float *uv_rect, // [u0, u1, v0, v1], nullptr for 2DGS
        Footprint &f
    )
    {
        f.kind     = FootprintKind::Box;
        f.bbox_min = {mean2d.x - radius.x, mean2d.y - radius.y};
        f.bbox_max = {mean2d.x + radius.x, mean2d.y + radius.y};
        f.r2       = 0.f;

        // Normalize by the center depth (w_M . (0, 0, 1) > near_plane) so the
        // products below stay well inside the float range.
        const float inv_z = 1.f / H[8];
        float h[9];
#pragma unroll
        for(int k = 0; k < 9; ++k)
        {
            h[k] = H[k] * inv_z;
        }

        if(uv_rect == nullptr)
        {
            if(opacity < ALPHA_THRESHOLD)
            {
                return false;
            }
            // Opacity-aware level t of u^2 + v^2 (and of the low-pass term),
            // capped at the projection's radius budget like AccuTile.
            const float t = fminf(GAUSSIAN_EXTEND * GAUSSIAN_EXTEND, 2.f * __logf(opacity / ALPHA_THRESHOLD));

            // Dual conic Q* = H diag(t, t, -1) H^T of the image ellipse. It is
            // an ellipse iff the disk lies in front of the camera, i.e.
            // Q*_33 = t (w_u^2 + w_v^2) - w_1^2 < 0. Near zero the ellipse
            // degenerates into an unbounded conic, so keep the box there.
            const float q33 = t * (h[6] * h[6] + h[7] * h[7]) - 1.f;
            if(q33 < -1e-3f)
            {
                const float q13     = t * (h[0] * h[6] + h[1] * h[7]) - h[2];
                const float q23     = t * (h[3] * h[6] + h[4] * h[7]) - h[5];
                // S = (q_i3 q_j3 - q_ij q33) / q33^2, expanded with Cauchy-Binet
                // into 2x2 minors of H (rows i and w, columns k and l). This
                // avoids the cancellation of the equivalent c c^T - Q*_2x2 / q33.
                const float mx01    = h[0] * h[7] - h[1] * h[6];
                const float mx02    = h[0] * h[8] - h[2] * h[6];
                const float mx12    = h[1] * h[8] - h[2] * h[7];
                const float my01    = h[3] * h[7] - h[4] * h[6];
                const float my02    = h[3] * h[8] - h[5] * h[6];
                const float my12    = h[4] * h[8] - h[5] * h[7];
                const float inv     = 1.f / (q33 * q33);
                // A tiny isotropic dilation keeps S positive definite for
                // edge-on surfels (it only grows the ellipse).
                constexpr float eps = 1e-4f;
                f.sxx               = (t * (mx02 * mx02 + mx12 * mx12) - t * t * mx01 * mx01) * inv + eps;
                f.syy               = (t * (my02 * my02 + my12 * my12) - t * t * my01 * my01) * inv + eps;
                f.sxy               = (t * (mx02 * my02 + mx12 * my12) - t * t * mx01 * my01) * inv;
                f.c                 = {q13 / q33, q23 / q33};
                f.m                 = mean2d;
                f.r2                = t / FILTER_INV_SQUARE_2DGS;
                const float r       = sqrtf(f.r2);
                const float ex      = sqrtf(f.sxx);
                const float ey      = sqrtf(f.syy);
                if(isfinite(f.c.x) && isfinite(f.c.y) && isfinite(ex) && isfinite(ey) && f.sxx > 0.f && f.syy > 0.f)
                {
                    f.kind     = FootprintKind::Ellipse;
                    f.bbox_min = {fminf(f.c.x - ex, f.m.x - r), fminf(f.c.y - ey, f.m.y - r)};
                    f.bbox_max = {fmaxf(f.c.x + ex, f.m.x + r), fmaxf(f.c.y + ey, f.m.y + r)};
                }
            }
            return true;
        }

        const float u0 = uv_rect[0], u1 = uv_rect[1], v0 = uv_rect[2], v1 = uv_rect[3];
        if(!(u0 <= u1 && v0 <= v1))
        {
            // empty rectangle: no texel can reach ALPHA_THRESHOLD
            return false;
        }
        const float us[4] = {u0, u1, u1, u0};
        const float vs[4] = {v0, v0, v1, v1};
        float2 bmin       = {INFINITY, INFINITY};
        float2 bmax       = {-INFINITY, -INFINITY};
#pragma unroll
        for(int k = 0; k < 4; ++k)
        {
            const float w = h[6] * us[k] + h[7] * vs[k] + h[8];
            // Corners at or behind ~0.1% of the center depth make the image
            // unbounded (or numerically meaningless): keep the box.
            if(!(w > 1e-3f))
            {
                return true;
            }
            const float2 p = {
                (h[0] * us[k] + h[1] * vs[k] + h[2]) / w,
                (h[3] * us[k] + h[4] * vs[k] + h[5]) / w,
            };
            f.p[k] = p;
            bmin   = {fminf(bmin.x, p.x), fminf(bmin.y, p.y)};
            bmax   = {fmaxf(bmax.x, p.x), fmaxf(bmax.y, p.y)};
        }
        if(isfinite(bmin.x) && isfinite(bmin.y) && isfinite(bmax.x) && isfinite(bmax.y))
        {
            f.kind     = FootprintKind::Quad;
            f.bbox_min = bmin;
            f.bbox_max = bmax;
        }
        return true;
    }

    // Half extents of the smallest box centered at mean2d that contains the
    // footprint (the radii convention of the projection outputs).
    inline __device__ float2 radius_around(const Footprint &f, const float2 mean2d)
    {
        return {
            fmaxf(f.bbox_max.x - mean2d.x, mean2d.x - f.bbox_min.x),
            fmaxf(f.bbox_max.y - mean2d.y, mean2d.y - f.bbox_min.y),
        };
    }

    // Cull box and radii of the 2DGS projection. Starts from the legacy box
    // mean2d +- radius; with an opacity (2DGS) or uv_rect (BBSplat) it is
    // replaced by the exact footprint. Returns false if nothing can pass
    // ALPHA_THRESHOLD.
    inline __device__ bool projection_bounds(
        const float H[9],     // ray transform, rows u_M, v_M, w_M
        const float2 mean2d,  // projected center
        const float *opacity, // nullptr unless 2DGS opacity-aware culling
        const float *uv_rect, // [u0, u1, v0, v1], nullptr unless BBSplat
        float2 &radius,       // in: legacy radii, out: radii around mean2d
        float2 &bbox_min,
        float2 &bbox_max
    )
    {
        bbox_min = {mean2d.x - radius.x, mean2d.y - radius.y};
        bbox_max = {mean2d.x + radius.x, mean2d.y + radius.y};
        if(opacity == nullptr && uv_rect == nullptr)
        {
            return true;
        }
        Footprint f;
        if(!make_footprint(H, mean2d, radius, opacity != nullptr ? *opacity : 1.f, uv_rect, f))
        {
            return false;
        }
        if(f.kind != FootprintKind::Box)
        {
            const float2 r = radius_around(f, mean2d);
            radius         = {ceilf(r.x), ceilf(r.y)};
            bbox_min       = f.bbox_min;
            bbox_max       = f.bbox_max;
        }
        return true;
    }
} // namespace footprint2dgs
} // namespace gsplat
