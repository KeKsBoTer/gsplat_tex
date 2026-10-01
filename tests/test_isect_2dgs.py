# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exact 2DGS / BBSplat tile intersection (isect_tiles_2dgs)."""

import math

import pytest
import torch

import gsplat
from gsplat.cuda._constants import ALPHA_THRESHOLD

device = torch.device("cuda:0")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device"),
    pytest.mark.skipif(
        not (gsplat.has_2dgs() or gsplat.has_bbsplat()),
        reason="2DGS projection wasn't built",
    ),
]

TILE = 16


def _scene(N: int, C: int, W: int, H: int, seed: int = 0, min_depth: float = 0.4):
    """Surfels at a wide range of depths and tilts, some close to the camera.

    With the default `min_depth` some footprints cross the camera plane, which
    exercises the radii-box fallback.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    means = torch.randn(N, 3, device=device, generator=g) * torch.tensor(
        [0.8, 0.6, 1.0], device=device
    ) + torch.tensor([0.0, 0.0, 2.5], device=device)
    means[:, 2] = means[:, 2].clamp_min(min_depth)
    quats = torch.nn.functional.normalize(
        torch.randn(N, 4, device=device, generator=g), dim=-1
    )
    scales = torch.rand(N, 3, device=device, generator=g) * 0.3 + 0.02
    viewmats = torch.eye(4, device=device).repeat(C, 1, 1)
    viewmats[:, 0, 3] = torch.linspace(-0.3, 0.3, C, device=device)
    Ks = torch.tensor(
        [[W, 0.0, W / 2], [0.0, W, H / 2], [0.0, 0.0, 1.0]], device=device
    ).repeat(C, 1, 1)
    opacities = torch.rand(N, device=device, generator=g)
    return means, quats, scales, viewmats, Ks, opacities


def _textures(N: int, S: int, seed: int = 0):
    """Mostly transparent alpha textures with a random opaque blob."""
    g = torch.Generator(device=device).manual_seed(seed)
    lin = torch.linspace(-1.0, 1.0, S, device=device)
    v, u = torch.meshgrid(lin, lin, indexing="ij")
    center = torch.rand(N, 2, device=device, generator=g) * 1.2 - 0.6
    radius = torch.rand(N, device=device, generator=g) * 0.6 + 0.1
    d2 = (u - center[:, 0, None, None]) ** 2 + (v - center[:, 1, None, None]) ** 2
    tex = torch.sigmoid((radius[:, None, None] ** 2 - d2) * 40.0)
    # a few fully opaque and fully transparent textures
    tex[: N // 10] = 1.0
    tex[N // 10 : N // 5] = 0.0
    return tex


def _tile_masks(isect_ids, flatten_ids, n, tile_width, tile_height):
    n_tiles = tile_width * tile_height
    tile_ids = (isect_ids >> 32) & ((1 << (n_tiles - 1).bit_length()) - 1)
    masks = torch.zeros(n, n_tiles, dtype=torch.int32, device=device)
    masks.index_put_(
        (flatten_ids.long(), tile_ids.long()),
        torch.ones_like(flatten_ids),
        accumulate=True,
    )
    return masks.reshape(n, tile_height, tile_width)


def _project(N, C, W, H, seed, packed=False, min_depth=0.4):
    from gsplat.cuda._wrapper import fully_fused_projection_2dgs

    means, quats, scales, viewmats, Ks, opacities = _scene(N, C, W, H, seed, min_depth)
    proj = fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, W, H, packed=packed
    )
    return proj, opacities


@pytest.mark.parametrize("mode", ["2dgs", "bbsplat"])
@pytest.mark.parametrize("seed", [0, 1])
def test_isect_tiles_2dgs_matches_reference(mode, seed):
    from gsplat.cuda._torch_impl_2dgs import _isect_tiles_2dgs
    from gsplat.cuda._wrapper import bbsplat_uv_rects, isect_tiles, isect_tiles_2dgs

    N, C, W, H, S = 200, 2, 96, 80, 8
    tw, th = math.ceil(W / TILE), math.ceil(H / TILE)
    (radii, means2d, depths, ray_transforms, _), opacities = _project(N, C, W, H, seed)
    kwargs = {}
    if mode == "2dgs":
        kwargs["opacities"] = opacities.expand(C, N).contiguous()
    else:
        rects = bbsplat_uv_rects(_textures(N, S, seed), opacities)
        kwargs["uv_rects"] = rects.expand(C, N, 4).contiguous()

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_2dgs(
        means2d, radii, depths, ray_transforms, TILE, tw, th, **kwargs
    )
    kern = _tile_masks(isect_ids, flatten_ids, C * N, tw, th)
    assert kern.max() <= 1, "duplicate (primitive, tile) intersections"
    assert torch.equal(kern.sum(dim=(1, 2)), tiles_per_gauss.reshape(-1))
    kern = kern.bool()

    ref, exact = _isect_tiles_2dgs(
        means2d, radii, ray_transforms, TILE, tw, th, **kwargs
    )
    ref, exact = ref.reshape(C * N, th, tw), exact.reshape(-1)
    assert exact.sum() > 0.5 * (radii > 0).all(-1).sum()

    # Conservative: every tile containing a covered sample is assigned.
    missing = ref & ~kern
    assert not missing[exact].any(), f"{missing[exact].sum()} missing tiles"
    # Tight: assigned tiles are (almost) all touched by the footprint. The
    # reference samples at 1/4 px, so slivers thinner than that may differ.
    extra = (kern & ~ref)[exact].sum().item()
    assert extra <= 0.02 * kern[exact].sum().item() + 2, extra

    # Never worse than the radii box in total.
    n_box = isect_tiles(means2d, radii, depths, TILE, tw, th)[0].sum().item()
    assert tiles_per_gauss.sum().item() <= n_box


@pytest.mark.parametrize("mode", ["2dgs", "bbsplat"])
def test_isect_tiles_2dgs_packed_matches_dense(mode):
    from gsplat.cuda._wrapper import bbsplat_uv_rects, isect_tiles_2dgs

    N, C, W, H, S = 150, 3, 64, 64, 6
    tw, th = math.ceil(W / TILE), math.ceil(H / TILE)
    (radii, means2d, depths, ray_transforms, _), opacities = _project(N, C, W, H, 3)
    rects = bbsplat_uv_rects(_textures(N, S, 3), opacities)

    def extra(gids):
        if mode == "2dgs":
            return {"opacities": opacities[gids]}
        return {"uv_rects": rects[gids]}

    dense_gids = torch.arange(N, device=device).expand(C, N)
    _, isect_ids, flatten_ids = isect_tiles_2dgs(
        means2d, radii, depths, ray_transforms, TILE, tw, th, **extra(dense_gids)
    )
    dense = _tile_masks(isect_ids, flatten_ids, C * N, tw, th)

    (
        batch_ids,
        camera_ids,
        gaussian_ids,
        _,
        p_radii,
        p_means2d,
        p_depths,
        p_ray_transforms,
        _,
    ) = _project(N, C, W, H, 3, packed=True)[0]
    image_ids = batch_ids * C + camera_ids
    _, p_isect_ids, p_flatten_ids = isect_tiles_2dgs(
        p_means2d,
        p_radii,
        p_depths,
        p_ray_transforms,
        TILE,
        tw,
        th,
        packed=True,
        n_images=C,
        image_ids=image_ids,
        **extra(gaussian_ids.long()),
    )
    packed = _tile_masks(p_isect_ids, p_flatten_ids, len(gaussian_ids), tw, th)
    dense_rows = dense[image_ids.long() * N + gaussian_ids.long()]
    torch.testing.assert_close(packed, dense_rows)
    assert dense.sum() == packed.sum()
    # image ids of the packed keys match their primitives
    n_tile_bits = (tw * th - 1).bit_length()
    assert torch.equal(
        p_isect_ids >> (32 + n_tile_bits), image_ids[p_flatten_ids.long()].long()
    )


def _all_tiles(means2d, radii, depths, tw, th):
    """Intersections with every tile, for primitives the projection keeps."""
    from gsplat.cuda._wrapper import isect_tiles

    big = torch.where(radii > 0, torch.full_like(radii, 1 << 20), 0)
    return isect_tiles(means2d, big, depths, TILE, tw, th)


@pytest.mark.skipif(not gsplat.has_2dgs(), reason="2DGS wasn't built")
def test_rasterize_2dgs_exact_isect_is_lossless():
    """Rendering with the exact tiles equals rendering with all tiles."""
    from gsplat.cuda._wrapper import (
        isect_offset_encode,
        isect_tiles_2dgs,
        rasterize_to_pixels_2dgs,
    )

    N, C, W, H = 300, 2, 96, 80
    tw, th = math.ceil(W / TILE), math.ceil(H / TILE)
    # Keep every footprint in front of the camera: the radii-box fallback
    # for footprints crossing the camera plane is not conservative.
    (radii, means2d, depths, ray_transforms, normals), opacities = _project(
        N, C, W, H, 5, min_depth=1.5
    )
    opacities = opacities.expand(C, N).contiguous()
    colors = torch.rand(C, N, 3, device=device)
    densify = torch.zeros_like(means2d)

    def render(isect):
        _, isect_ids, flatten_ids = isect
        offsets = isect_offset_encode(isect_ids, C, tw, th).reshape(C, th, tw)
        return rasterize_to_pixels_2dgs(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            W,
            H,
            TILE,
            offsets,
            flatten_ids,
        )

    exact_isect = isect_tiles_2dgs(
        means2d, radii, depths, ray_transforms, TILE, tw, th, opacities=opacities
    )
    all_isect = _all_tiles(means2d, radii, depths, tw, th)
    assert exact_isect[1].numel() < 0.5 * all_isect[1].numel()
    for a, b in zip(render(exact_isect)[:3], render(all_isect)[:3]):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not gsplat.has_bbsplat(), reason="BBSplat wasn't built")
def test_rasterize_bbsplat_exact_isect_is_lossless():
    """Rendering with the exact tiles equals rendering with all tiles."""
    from gsplat.cuda._wrapper import (
        bbsplat_uv_rects,
        isect_offset_encode,
        isect_tiles_2dgs,
        rasterize_to_pixels_bbsplat,
    )

    N, C, W, H, S = 300, 2, 96, 80, 8
    tw, th = math.ceil(W / TILE), math.ceil(H / TILE)
    (radii, means2d, depths, ray_transforms, normals), opacities = _project(
        N, C, W, H, 6, min_depth=1.5
    )
    texture_alphas = _textures(N, S, 6)
    rects = bbsplat_uv_rects(texture_alphas, opacities)
    colors = torch.rand(C, N, 3, device=device)
    densify = torch.zeros_like(means2d)
    texture_ids = torch.arange(N, device=device).expand(C, N).contiguous()

    def render(isect):
        _, isect_ids, flatten_ids = isect
        offsets = isect_offset_encode(isect_ids, C, tw, th).reshape(C, th, tw)
        return rasterize_to_pixels_bbsplat(
            ray_transforms,
            colors,
            opacities.expand(C, N).contiguous(),
            normals,
            densify,
            texture_alphas,
            None,
            texture_ids,
            W,
            H,
            TILE,
            offsets,
            flatten_ids,
        )

    exact_isect = isect_tiles_2dgs(
        means2d,
        radii,
        depths,
        ray_transforms,
        TILE,
        tw,
        th,
        uv_rects=rects.expand(C, N, 4).contiguous(),
    )
    all_isect = _all_tiles(means2d, radii, depths, tw, th)
    assert exact_isect[1].numel() < 0.5 * all_isect[1].numel()
    for a, b in zip(render(exact_isect)[:3], render(all_isect)[:3]):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_bbsplat_uv_rects_bound_alpha():
    """Bilinear samples outside the rectangle stay below ALPHA_THRESHOLD."""
    from gsplat.cuda._wrapper import bbsplat_uv_rects

    N, S = 64, 7
    tex = _textures(N, S, 7)
    opacities = torch.rand(N, device=device)
    rects = bbsplat_uv_rects(tex, opacities)

    # grid_sample(align_corners=True, zeros) matches the rasterizer's lookup
    lin = torch.linspace(-1.4, 1.4, 281, device=device)
    v, u = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([u, v], dim=-1).expand(N, -1, -1, -1)
    alpha = (
        torch.nn.functional.grid_sample(
            tex[:, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]
        * opacities[:, None, None]
    )
    inside = (
        (u >= rects[:, 0, None, None])
        & (u <= rects[:, 1, None, None])
        & (v >= rects[:, 2, None, None])
        & (v <= rects[:, 3, None, None])
    )
    assert not (alpha >= ALPHA_THRESHOLD)[~inside].any()
    # empty textures get empty rectangles, opaque ones the full extended square
    assert (rects[N // 10 : N // 5, 0] > rects[N // 10 : N // 5, 1]).all()
    full = opacities[: N // 10] >= ALPHA_THRESHOLD
    ext = 1.0 + 2.0 / (S - 1)
    torch.testing.assert_close(
        rects[: N // 10][full],
        torch.tensor([-ext, ext, -ext, ext], device=device).expand(int(full.sum()), 4),
    )


@pytest.mark.skipif(not gsplat.has_bbsplat(), reason="BBSplat wasn't built")
@pytest.mark.parametrize("S", [2, 7, 16, 33])
@pytest.mark.parametrize("with_opacities", [False, True])
def test_bbsplat_uv_rects_matches_reference(S, with_opacities):
    from gsplat.cuda._torch_impl_2dgs import _bbsplat_uv_rects
    from gsplat.cuda._wrapper import bbsplat_uv_rects

    N = 500
    tex = _textures(N, S, S).reshape(2, N // 2, S, S)
    opacities = (
        torch.rand(2, N // 2, device=device) * 1.2 - 0.1 if with_opacities else None
    )
    rects = bbsplat_uv_rects(tex, opacities)
    assert rects.shape == (2, N // 2, 4) and rects.dtype == torch.float32
    torch.testing.assert_close(rects, _bbsplat_uv_rects(tex, opacities))


@pytest.mark.parametrize("mode", ["2dgs", "bbsplat"])
@pytest.mark.parametrize("packed", [False, True])
def test_projection_2dgs_exact_culling(mode, packed):
    """With opacities / uv_rects the projection culls by the exact footprint and
    its radii box still contains that footprint."""
    from gsplat.cuda._torch_impl_2dgs import _isect_tiles_2dgs
    from gsplat.cuda._wrapper import (
        bbsplat_uv_rects,
        fully_fused_projection_2dgs,
        isect_tiles,
    )

    N, C, W, H, S = 300, 2, 96, 80, 8
    tw, th = math.ceil(W / TILE), math.ceil(H / TILE)
    means, quats, scales, viewmats, Ks, opacities = _scene(N, C, W, H, 8)
    opacities[: N // 10] = 0.5 * ALPHA_THRESHOLD  # invisible
    if mode == "2dgs":
        cull = {"opacities": opacities}
    else:
        cull = {"uv_rects": bbsplat_uv_rects(_textures(N, S, 8), opacities)}

    project = lambda **kw: fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, W, H, **kw
    )
    radii_old, means2d_old, _, ray_transforms_old, _ = project()
    radii, means2d, depths, ray_transforms, _ = project(**cull)
    kept, kept_old = (radii > 0).all(-1), (radii_old > 0).all(-1)
    both = kept & kept_old
    torch.testing.assert_close(means2d[both], means2d_old[both])
    torch.testing.assert_close(ray_transforms[both], ray_transforms_old[both])
    assert (kept_old & ~kept).sum() >= C * N // 10  # at least the invisible ones

    if packed:
        _, c, g, _, radii_p, means2d_p, _, ray_transforms_p, _ = project(
            packed=True, **cull
        )
        assert torch.equal(kept.nonzero(), torch.stack([c, g], dim=-1))
        torch.testing.assert_close(radii_p, radii[c, g])
        torch.testing.assert_close(means2d_p, means2d[c, g])
        torch.testing.assert_close(ray_transforms_p, ray_transforms[c, g])
        return

    per_cam = {k: v.expand(C, *v.shape) for k, v in cull.items()}

    def coverage(m2d, r, rt):
        ref, in_front = _isect_tiles_2dgs(m2d, r, rt, TILE, tw, th, **per_cam)
        return ref.reshape(C * N, th, tw), in_front.reshape(-1)

    # Culled primitives cover nothing in the image.
    ref_old, front_old = coverage(means2d_old, radii_old, ray_transforms_old)
    culled = (kept_old & ~kept).reshape(-1) & front_old
    assert not ref_old[culled].any()

    # Kept primitives' radii box around means2d contains the footprint, and
    # those the legacy box wrongly culled do reach into the image.
    ref, front = coverage(means2d, radii, ray_transforms)
    _, ids, fids = isect_tiles(means2d, radii, depths, TILE, tw, th)
    box = _tile_masks(ids, fids, C * N, tw, th).bool()
    assert not (ref & ~box)[front].any()
    rescued = (kept & ~kept_old).reshape(-1) & front
    assert ref[rescued].flatten(1).any(dim=1).all()


@pytest.mark.skipif(not gsplat.has_2dgs(), reason="2DGS wasn't built")
def test_rasterization_2dgs_is_lossless():
    """The full pipeline (exact culling + tiles) renders like the legacy
    projection with every tile assigned."""
    from gsplat import rasterization_2dgs
    from gsplat.cuda._wrapper import (
        fully_fused_projection_2dgs,
        isect_offset_encode,
        rasterize_to_pixels_2dgs,
    )

    N, C, W, H = 300, 2, 96, 80
    tw, th = math.ceil(W / TILE), math.ceil(H / TILE)
    means, quats, scales, viewmats, Ks, opacities = _scene(N, C, W, H, 9, 1.5)
    opacities[: N // 10] = 0.5 * ALPHA_THRESHOLD
    colors = torch.rand(C, N, 3, device=device)

    render, alpha, *_ = rasterization_2dgs(
        means, quats, scales, opacities, colors, viewmats, Ks, W, H
    )

    radii, means2d, depths, ray_transforms, normals = fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, W, H
    )
    _, isect_ids, flatten_ids = _all_tiles(means2d, radii, depths, tw, th)
    offsets = isect_offset_encode(isect_ids, C, tw, th).reshape(C, th, tw)
    ref_render, ref_alpha, _ = rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        colors,
        opacities.expand(C, N).contiguous(),
        normals,
        torch.zeros_like(means2d),
        W,
        H,
        TILE,
        offsets,
        flatten_ids,
    )[:3]
    torch.testing.assert_close(render, ref_render, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(alpha, ref_alpha, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not gsplat.has_2dgs(), reason="2DGS wasn't built")
def test_rasterization_2dgs_packed_matches_dense():
    """Packed mode culls the same primitives (packed rasterization_2dgs only
    takes SH colors)."""
    from gsplat import rasterization_2dgs

    N, C, W, H = 300, 2, 96, 80
    means, quats, scales, viewmats, Ks, opacities = _scene(N, C, W, H, 11, 1.5)
    opacities[: N // 10] = 0.5 * ALPHA_THRESHOLD
    sh = torch.randn(N, 4, 3, device=device) * 0.3
    outs = [
        rasterization_2dgs(
            means,
            quats,
            scales,
            opacities,
            sh,
            viewmats,
            Ks,
            W,
            H,
            sh_degree=1,
            packed=packed,
        )
        for packed in (False, True)
    ]
    for a, b in zip(outs[0][:2], outs[1][:2]):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    dense_kept = (outs[0][-1]["radii"] > 0).all(-1).sum()
    assert len(outs[1][-1]["gaussian_ids"]) == dense_kept


@pytest.mark.skipif(not gsplat.has_bbsplat(), reason="BBSplat wasn't built")
@pytest.mark.parametrize("packed", [False, True])
def test_rasterization_bbsplat_is_lossless(packed):
    """The full pipeline (exact culling + tiles) renders like the legacy
    projection with every tile assigned."""
    from gsplat import rasterization_bbsplat
    from gsplat.cuda._wrapper import (
        fully_fused_projection_2dgs,
        isect_offset_encode,
        rasterize_to_pixels_bbsplat,
    )

    N, C, W, H, S = 300, 2, 96, 80, 8
    tw, th = math.ceil(W / TILE), math.ceil(H / TILE)
    means, quats, scales, viewmats, Ks, opacities = _scene(N, C, W, H, 10, 1.5)
    texture_alphas = _textures(N, S, 10)
    colors = torch.rand(N, 3, device=device)

    render, alpha, *_ = rasterization_bbsplat(
        means,
        quats,
        scales,
        opacities,
        colors,
        texture_alphas,
        None,
        viewmats,
        Ks,
        W,
        H,
        packed=packed,
    )

    radii, means2d, depths, ray_transforms, normals = fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, W, H
    )
    _, isect_ids, flatten_ids = _all_tiles(means2d, radii, depths, tw, th)
    offsets = isect_offset_encode(isect_ids, C, tw, th).reshape(C, th, tw)
    ref_render, ref_alpha = rasterize_to_pixels_bbsplat(
        ray_transforms,
        colors.expand(C, N, 3).contiguous(),
        opacities.expand(C, N).contiguous(),
        normals,
        torch.zeros_like(means2d),
        texture_alphas,
        None,
        torch.arange(N, device=device).expand(C, N).contiguous(),
        W,
        H,
        TILE,
        offsets,
        flatten_ids,
    )[:2]
    torch.testing.assert_close(render, ref_render, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(alpha, ref_alpha, atol=1e-6, rtol=1e-6)
