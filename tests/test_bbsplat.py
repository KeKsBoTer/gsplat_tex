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

import math

import pytest
import torch
from typing_extensions import Tuple

import gsplat

device = torch.device("cuda:0")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device"),
    pytest.mark.skipif(not gsplat.has_bbsplat(), reason="BBSplat support wasn't built"),
]


def _scene(N: int, C: int, S: int, W: int, H: int, seed: int = 0):
    g = torch.Generator(device=device).manual_seed(seed)
    means = torch.randn(N, 3, device=device, generator=g) * torch.tensor(
        [0.8, 0.6, 0.5], device=device
    ) + torch.tensor([0.0, 0.0, 3.0], device=device)
    quats = torch.nn.functional.normalize(
        torch.randn(N, 4, device=device, generator=g), dim=-1
    )
    scales = torch.rand(N, 3, device=device, generator=g) * 0.2 + 0.05
    viewmats = torch.eye(4, device=device).repeat(C, 1, 1)
    viewmats[:, 0, 3] = torch.linspace(-0.2, 0.2, C, device=device)
    Ks = torch.tensor(
        [[W, 0.0, W / 2], [0.0, W, H / 2], [0.0, 0.0, 1.0]], device=device
    ).repeat(C, 1, 1)
    texture_alphas = torch.rand(N, S, S, device=device, generator=g)
    texture_colors = torch.rand(N, S, S, 3, device=device, generator=g) - 0.5
    return means, quats, scales, viewmats, Ks, texture_alphas, texture_colors


@pytest.mark.parametrize("batch_dims", [(), (2,)])
def test_rasterize_to_pixels_bbsplat(batch_dims: Tuple[int, ...]):
    from gsplat.cuda._torch_impl_2dgs import _rasterize_to_pixels_bbsplat
    from gsplat.cuda._wrapper import (
        fully_fused_projection_2dgs,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels_bbsplat,
    )

    N, C, S, W, H = 60, 2, 8, 48, 40
    means, quats, scales, viewmats, Ks, tex_a, tex_c = _scene(N, C, S, W, H)
    B = math.prod(batch_dims)
    I = B * C

    def bexp(x):
        return x.expand(batch_dims + x.shape)

    radii, means2d, depths, ray_transforms, normals = fully_fused_projection_2dgs(
        bexp(means), bexp(quats), bexp(scales), bexp(viewmats), bexp(Ks), W, H
    )
    tile_size = 16
    tile_width = math.ceil(W / tile_size)
    tile_height = math.ceil(H / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    # 3 color channels + depth; textures only touch the color channels
    colors = torch.cat(
        [torch.rand(batch_dims + (C, N, 3), device=device), depths[..., None]], dim=-1
    )
    opacities = torch.rand(batch_dims + (C, N), device=device) * 0.5 + 0.5
    backgrounds = torch.rand(batch_dims + (C, 4), device=device)
    densify = torch.zeros_like(means2d)
    # each batch element owns its textures
    texture_alphas = tex_a.repeat(B, 1, 1)
    texture_colors = tex_c.repeat(B, 1, 1, 1)
    texture_ids = torch.broadcast_to(
        torch.arange(B * N, device=device).reshape(B, 1, N), (B, C, N)
    ).reshape(batch_dims + (C, N))

    inputs = [
        ray_transforms,
        colors,
        opacities,
        normals,
        backgrounds,
        texture_alphas,
        texture_colors,
    ]
    for x in inputs:
        x.requires_grad_(True)

    (
        render_colors,
        render_alphas,
        render_normals,
        _,
        _,
        impacts,
    ) = rasterize_to_pixels_bbsplat(
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        texture_alphas,
        texture_colors,
        texture_ids,
        W,
        H,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        compute_impact=True,
    )
    _render_colors, _render_alphas, _render_normals = _rasterize_to_pixels_bbsplat(
        ray_transforms,
        colors,
        opacities,
        normals,
        texture_alphas,
        texture_colors,
        texture_ids,
        radii,
        depths,
        W,
        H,
        backgrounds=backgrounds,
    )

    assert render_alphas.max() > 0.5, "test scene renders (almost) nothing"
    # sum over splats of their total blending weight == sum over pixels of alpha
    assert impacts.shape == opacities.shape
    torch.testing.assert_close(
        impacts.sum(dim=-1), render_alphas.sum(dim=(-3, -2, -1)), rtol=1e-4, atol=1e-2
    )
    torch.testing.assert_close(render_colors, _render_colors, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(render_alphas, _render_alphas, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(render_normals, _render_normals, atol=1e-4, rtol=1e-4)

    v_colors_out = torch.rand_like(render_colors)
    v_alphas_out = torch.rand_like(render_alphas)
    v_normals_out = torch.rand_like(render_normals)

    def loss(c, a, n):
        return (
            (c * v_colors_out).sum()
            + (a * v_alphas_out).sum()
            + (n * v_normals_out).sum()
        )

    grads = torch.autograd.grad(
        loss(render_colors, render_alphas, render_normals), inputs
    )
    _grads = torch.autograd.grad(
        loss(_render_colors, _render_alphas, _render_normals), inputs
    )
    names = [
        "ray_transforms",
        "colors",
        "opacities",
        "normals",
        "backgrounds",
        "texture_alphas",
        "texture_colors",
    ]
    for name, g, _g in zip(names, grads, _grads):
        assert _g.abs().max() > 0, f"{name}: reference gradient is all zero"
        scale = _g.abs().max().item()
        torch.testing.assert_close(
            g, _g, atol=1e-3 * scale, rtol=1e-3, msg=lambda m: f"{name}: {m}"
        )


def test_rasterize_to_pixels_bbsplat_footprint():
    """A fronto-parallel splat with an opaque texture covers exactly its square."""
    from gsplat import rasterization_bbsplat

    W = H = 64
    S = 65
    z, scale = 2.0, 0.25
    means = torch.tensor([[0.0, 0.0, z]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[scale, scale, 1.0]], device=device)
    colors = torch.zeros(1, 3, device=device)
    texture_alphas = torch.ones(1, S, S, device=device)
    # color texture encodes u in red and v in green
    lin = torch.linspace(-1.0, 1.0, S, device=device)
    texture_colors = torch.stack(
        torch.meshgrid(lin, lin, indexing="xy") + (torch.zeros(S, S, device=device),),
        dim=-1,
    )[None]
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor(
        [[W, 0.0, W / 2], [0.0, W, H / 2], [0.0, 0.0, 1.0]], device=device
    )[None]

    render_colors, render_alphas, *_ = rasterization_bbsplat(
        means,
        quats,
        scales,
        None,
        colors,
        texture_alphas,
        texture_colors,
        viewmats,
        Ks,
        W,
        H,
    )
    alpha = render_alphas[0, ..., 0]
    half = W * scale / z  # 8 px
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device) + 0.5 - H / 2,
        torch.arange(W, device=device) + 0.5 - W / 2,
        indexing="ij",
    )
    inside = (xs.abs() < half - 0.5) & (ys.abs() < half - 0.5)
    outside = (xs.abs() > half + 0.5) | (ys.abs() > half + 0.5)
    assert torch.allclose(alpha[inside], torch.full_like(alpha[inside], 0.99))
    assert (alpha[outside] == 0).all()

    # u increases with image x, v with image y
    rgb = render_colors[0] / render_alphas[0].clamp_min(1e-6)
    torch.testing.assert_close(
        rgb[..., 0][inside], (xs / half)[inside], atol=1e-3, rtol=0
    )
    torch.testing.assert_close(
        rgb[..., 1][inside], (ys / half)[inside], atol=1e-3, rtol=0
    )


@pytest.mark.parametrize("sh_degree", [None, 2])
def test_rasterization_bbsplat_packed_matches_dense(sh_degree):
    from gsplat import rasterization_bbsplat

    N, C, S, W, H = 80, 3, 6, 64, 48
    means, quats, scales, viewmats, Ks, tex_a, tex_c = _scene(N, C, S, W, H, seed=1)
    opacities = torch.rand(N, device=device) * 0.5 + 0.5
    if sh_degree is None:
        colors = torch.rand(N, 3, device=device)
    else:
        colors = torch.randn(N, (sh_degree + 1) ** 2, 3, device=device) * 0.3
    backgrounds = torch.rand(C, 3, device=device)

    outs = {}
    for packed in (False, True):
        params = [
            x.clone().requires_grad_(True)
            for x in (means, quats, scales, opacities, colors, tex_a, tex_c)
        ]
        (
            render_colors,
            render_alphas,
            render_normals,
            surf_normals,
            render_distort,
            render_median,
            meta,
        ) = rasterization_bbsplat(
            *params[:5],
            params[5],
            params[6],
            viewmats,
            Ks,
            W,
            H,
            sh_degree=sh_degree,
            packed=packed,
            backgrounds=backgrounds,
            render_mode="RGB+ED",
            distloss=True,
        )
        assert render_colors.shape == (C, H, W, 4)
        assert render_alphas.shape == (C, H, W, 1)
        assert render_normals.shape == surf_normals.shape == (C, H, W, 3)
        loss = (
            render_colors.square().sum()
            + render_normals.sum()
            + render_distort.sum()
            + render_alphas.sum()
        )
        loss.backward()
        assert meta["gradient_2dgs"].grad is not None
        outs[packed] = (
            [render_colors, render_alphas, render_normals, render_distort],
            [p.grad for p in params],
        )

    for a, b in zip(outs[False][0], outs[True][0]):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    names = ["means", "quats", "scales", "opacities", "colors", "tex_a", "tex_c"]
    for name, a, b in zip(names, outs[False][1], outs[True][1]):
        assert a.abs().max() > 0, f"{name}: zero gradient"
        torch.testing.assert_close(
            a,
            b,
            atol=1e-3 * a.abs().max().item(),
            rtol=1e-3,
            msg=lambda m: f"{name}: {m}",
        )
