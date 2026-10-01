# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
from typing import Optional, Tuple

import torch
from torch import Tensor

from gsplat.cuda._math import _quat_scale_to_matrix
from gsplat.cuda._constants import (
    ALPHA_THRESHOLD,
    FILTER_INV_SQUARE_2DGS,
    GAUSSIAN_EXTEND,
    MAX_ALPHA,
    TRANSMITTANCE_THRESHOLD,
)


def _fully_fused_projection_2dgs(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps: float = 0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection_2dgs()`

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape

    R_cw = viewmats[..., :3, :3]  # [..., C, 3, 3]
    t_cw = viewmats[..., :3, 3]  # [..., C, 3]
    means_c = (
        torch.einsum("...cij,...nj->...cni", R_cw, means) + t_cw[..., None, :]
    )  # [..., C, N, 3]
    RS_wl = _quat_scale_to_matrix(quats, scales)
    RS_cl = torch.einsum("...cij,...njk->...cnik", R_cw, RS_wl)  # [..., C, N, 3, 3]

    # compute normals
    normals = RS_cl[..., 2]  # [..., C, N, 3]
    cos = -normals.reshape((-1, 1, 3)) @ means_c.reshape((-1, 3, 1))
    cos = cos.reshape(batch_dims + (C, N, 1))
    multiplier = torch.where(cos > 0, torch.tensor(1.0), torch.tensor(-1.0))
    normals *= multiplier

    # ray transform matrix, omitting the z rotation
    T_cl = torch.cat([RS_cl[..., :2], means_c[..., None]], dim=-1)  # [..., C, N, 3, 3]
    T_sl = torch.einsum(
        "...cij,...cnjk->...cnik", Ks[..., :3, :3], T_cl
    )  # [..., C, N, 3, 3]
    # in paper notation M = (WH)^T
    # later h_u = M @ h_x, h_v = M @ h_y
    M = torch.transpose(T_sl, -1, -2)  # [..., C, N, 3, 3]

    # compute the AABB of gaussian
    test = torch.tensor([1.0, 1.0, -1.0], device=means.device).expand(
        batch_dims + (1, 1, 3)
    )
    d = (M[..., 2] * M[..., 2] * test).sum(dim=-1, keepdim=True)  # [..., C, N, 1]
    valid = torch.abs(d) > eps
    f = torch.where(valid, test / d, torch.zeros_like(test)).unsqueeze(
        -1
    )  # [..., C, N, 3, 1]
    means2d = (M[..., :2] * M[..., 2:3] * f).sum(dim=-2)  # [..., C, N, 2]
    extents = torch.sqrt(
        (means2d**2 - (M[..., :2] * M[..., :2] * f).sum(dim=-2)).clamp_min(1e-4)
    )  # [..., C, N, 2]

    depths = means_c[..., 2]  # [..., C, N]
    radius = torch.ceil(3.33 * extents)  # [..., C, N, 2]

    valid = valid.squeeze(-1) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < width)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < height)
    )
    radius[~inside] = 0.0
    radii = radius.int()
    M = torch.transpose(M, -1, -2)  # [..., C, N, 3, 3]
    return radii, means2d, depths, M, normals


def accumulate_2dgs(
    means2d: Tensor,  # [..., N, 2]
    ray_transforms: Tensor,  # [..., N, 3, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,  # [..., N, channels]
    normals: Tensor,  # [..., N, 3]
    gaussian_ids: Tensor,  # [M]
    pixel_ids: Tensor,  # [M]
    image_ids: Tensor,  # [M]
    image_width: int,
    image_height: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Alpha compositing for 2DGS.

    .. warning::
        This function requires the nerfacc package to be installed. Please install it using the following command pip install nerfacc.

    Args:
        means2d: Gaussian means in 2D. [C, N, 2]
        ray_transforms: transformation matrices that transform rays in pixel space into splat's local frame. [C, N, 3, 3]
        opacities: Per-view Gaussian opacities (for example, when antialiasing is enabled, Gaussian in
            each view would efficiently have different opacity). [C, N]
        colors: Per-view Gaussian colors. Supports N-D features. [C, N, channels]
        normals: Per-view Gaussian normals. [C, N, 3]
        gaussian_ids: Collection of Gaussian indices to be rasterized. A flattened list of shape [M].
        pixel_ids: Collection of pixel indices (row-major) to be rasterized. A flattened list of shape [M].
        image_ids: Collection of image indices to be rasterized. A flattened list of shape [M].
        image_width: Image width.
        image_height: Image height.

    Returns:
        A tuple:

        - **renders**: Accumulated colors. [..., image_height, image_width, channels]
        - **alphas**: Accumulated opacities. [..., image_height, image_width, 1]
        - **normals**: Accumulated normals. [..., image_height, image_width, 3]
    """

    try:
        from nerfacc import accumulate_along_rays, render_weight_from_alpha
    except ImportError:
        raise ImportError("Please install nerfacc package: pip install nerfacc")

    image_dims = means2d.shape[:-2]
    I = math.prod(image_dims)
    N = means2d.shape[-2]
    channels = colors.shape[-1]
    assert means2d.shape == image_dims + (N, 2), means2d.shape
    assert ray_transforms.shape == image_dims + (N, 3, 3), ray_transforms.shape
    assert opacities.shape == image_dims + (N,), opacities.shape
    assert colors.shape == image_dims + (N, channels), colors.shape
    assert normals.shape == image_dims + (N, 3), normals.shape

    means2d = means2d.reshape(I, N, 2)
    ray_transforms = ray_transforms.reshape(I, N, 3, 3)
    opacities = opacities.reshape(I, N)
    colors = colors.reshape(I, N, channels)
    normals = normals.reshape(I, N, 3)

    pixel_ids_x = pixel_ids % image_width + 0.5
    pixel_ids_y = pixel_ids // image_width + 0.5
    pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1)  # [M, 2]
    deltas = pixel_coords - means2d[image_ids, gaussian_ids]  # [M, 2]

    M = ray_transforms[image_ids, gaussian_ids]  # [M, 3, 3]

    h_u = -M[..., 0, :3] + M[..., 2, :3] * pixel_ids_x[..., None]  # [M, 3]
    h_v = -M[..., 1, :3] + M[..., 2, :3] * pixel_ids_y[..., None]  # [M, 3]
    tmp = torch.cross(h_u, h_v, dim=-1)
    us = tmp[..., 0] / tmp[..., 2]
    vs = tmp[..., 1] / tmp[..., 2]
    sigmas_3d = us**2 + vs**2  # [M]
    sigmas_2d = 2 * (deltas[..., 0] ** 2 + deltas[..., 1] ** 2)
    sigmas = 0.5 * torch.minimum(sigmas_3d, sigmas_2d)  # [M]

    alphas = torch.clamp_max(
        opacities[image_ids, gaussian_ids] * torch.exp(-sigmas), MAX_ALPHA
    )

    indices = image_ids * image_height * image_width + pixel_ids
    total_pixels = I * image_height * image_width

    weights, trans = render_weight_from_alpha(
        alphas, ray_indices=indices, n_rays=total_pixels
    )
    renders = accumulate_along_rays(
        weights,
        colors[image_ids, gaussian_ids],
        ray_indices=indices,
        n_rays=total_pixels,
    ).reshape(image_dims + (image_height, image_width, channels))
    alphas = accumulate_along_rays(
        weights, None, ray_indices=indices, n_rays=total_pixels
    ).reshape(image_dims + (image_height, image_width, 1))
    renders_normal = accumulate_along_rays(
        weights,
        normals[image_ids, gaussian_ids],
        ray_indices=indices,
        n_rays=total_pixels,
    ).reshape(image_dims + (image_height, image_width, 3))

    return renders, alphas, renders_normal


def _rasterize_to_pixels_2dgs(
    means2d: Tensor,  # [..., N, 2]
    ray_transforms: Tensor,  # [..., N, 3, 3]
    colors: Tensor,  # [..., N, channels]
    normals: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., channels]
    batch_per_iter: int = 100,
):
    """Pytorch implementation of `gsplat.cuda._wrapper.rasterize_to_pixels_2dgs()`.

    This function rasterizes 2D Gaussians to pixels in a Pytorch-friendly way. It
    iteratively accumulates the renderings within each batch of Gaussians. The
    interations are controlled by `batch_per_iter`.

    .. note::
        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.

    .. note::

        This function relies on Pytorch's autograd for the backpropagation. It is much slower
        than our fully fused rasterization implementation and comsumes much more GPU memory.
        But it could serve as a playground for new ideas or debugging, as no backward
        implementation is needed.

    .. warning::

        This function requires the `nerfacc` package to be installed. Please install it
        using the following command `pip install nerfacc`.
    """
    from ._wrapper import rasterize_to_indices_in_range_2dgs

    image_dims = means2d.shape[:-2]
    channels = colors.shape[-1]
    N = means2d.shape[-2]
    tile_height = isect_offsets.shape[-2]
    tile_width = isect_offsets.shape[-1]

    assert means2d.shape == image_dims + (N, 2), means2d.shape
    assert ray_transforms.shape == image_dims + (N, 3, 3), ray_transforms.shape
    assert colors.shape == image_dims + (N, channels), colors.shape
    assert normals.shape == image_dims + (N, 3), normals.shape
    assert opacities.shape == image_dims + (N,), opacities.shape
    assert isect_offsets.shape == image_dims + (
        tile_height,
        tile_width,
    ), isect_offsets.shape
    n_isects = len(flatten_ids)
    device = means2d.device

    render_colors = torch.zeros(
        image_dims + (image_height, image_width, channels), device=device
    )
    render_alphas = torch.zeros(
        image_dims + (image_height, image_width, 1), device=device
    )
    render_normals = torch.zeros(
        image_dims + (image_height, image_width, 3), device=device
    )

    # Split Gaussians into batches and iteratively accumulate the renderings
    block_size = tile_size * tile_size
    isect_offsets_fl = torch.cat(
        [isect_offsets.flatten(), torch.tensor([n_isects], device=device)]
    )
    max_range = (isect_offsets_fl[1:] - isect_offsets_fl[:-1]).max().item()
    num_batches = (max_range + block_size - 1) // block_size
    for step in range(0, num_batches, batch_per_iter):
        transmittances = 1.0 - render_alphas[..., 0]

        # Find the M intersections between pixels and gaussians.
        # Each intersection corresponds to a tuple (gs_id, pixel_id, image_id)
        gs_ids, pixel_ids, image_ids = rasterize_to_indices_in_range_2dgs(
            step,
            step + batch_per_iter,
            transmittances,
            means2d,
            ray_transforms,
            opacities,
            image_width,
            image_height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )  # [M], [M]
        if len(gs_ids) == 0:
            break

        # Accumulate the renderings within this batch of Gaussians.
        renders_step, accs_step, renders_normal_step = accumulate_2dgs(
            means2d,
            ray_transforms,
            opacities,
            colors,
            normals,
            gs_ids,
            pixel_ids,
            image_ids,
            image_width,
            image_height,
        )
        render_colors = render_colors + renders_step * transmittances[..., None]
        render_alphas = render_alphas + accs_step * transmittances[..., None]
        render_normals = (
            render_normals + renders_normal_step * transmittances[..., None]
        )

    render_alphas = render_alphas
    if backgrounds is not None:
        render_colors = render_colors + backgrounds[..., None, None, :] * (
            1.0 - render_alphas
        )

    return render_colors, render_alphas, render_normals


def _rasterize_to_pixels_bbsplat(
    ray_transforms: Tensor,  # [..., N, 3, 3]
    colors: Tensor,  # [..., N, channels]
    opacities: Tensor,  # [..., N]
    normals: Tensor,  # [..., N, 3]
    texture_alphas: Tensor,  # [M, S, S]
    texture_colors: Optional[Tensor],  # [M, S, S, TC]
    texture_ids: Tensor,  # [..., N]
    radii: Tensor,  # [..., N, 2]
    depths: Tensor,  # [..., N]
    image_width: int,
    image_height: int,
    backgrounds: Optional[Tensor] = None,  # [..., channels]
) -> Tuple[Tensor, Tensor, Tensor]:
    """PyTorch implementation of `gsplat.cuda._wrapper.rasterize_to_pixels_bbsplat()`.

    Composites every visible splat (``radii > 0``) over the full image, front to
    back by ``depths``, instead of using tile intersections. This matches the
    CUDA op as long as the projected tile bounds cover each splat's textured
    footprint. Textures are sampled with ``F.grid_sample(align_corners=True)``,
    which is the convention the CUDA kernel implements. It is slow and meant for
    testing on small images, and it relies on autograd for the backward pass.
    """
    import torch.nn.functional as F

    image_dims = ray_transforms.shape[:-3]
    N = ray_transforms.shape[-3]
    I = math.prod(image_dims)
    channels = colors.shape[-1]
    device = ray_transforms.device

    ray_transforms = ray_transforms.reshape(I, N, 3, 3)
    colors = colors.reshape(I, N, channels)
    opacities = opacities.reshape(I, N)
    normals = normals.reshape(I, N, 3)
    texture_ids = texture_ids.reshape(I, N)
    radii = radii.reshape(I, N, 2)
    depths = depths.reshape(I, N)

    ys, xs = torch.meshgrid(
        torch.arange(image_height, device=device, dtype=torch.float32) + 0.5,
        torch.arange(image_width, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )  # [H, W]
    pix = torch.stack([xs, ys], dim=-1)  # [H, W, 2]

    def sample(texture: Tensor, grid: Tensor) -> Tensor:
        # texture [S, S, K] -> [H, W, K]
        out = F.grid_sample(
            texture.permute(2, 0, 1)[None],
            grid[None],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return out[0].permute(1, 2, 0)

    out_colors, out_alphas, out_normals = [], [], []
    for i in range(I):
        T = torch.ones(image_height, image_width, 1, device=device)
        done = torch.zeros(
            image_height, image_width, 1, dtype=torch.bool, device=device
        )
        render_c = torch.zeros(image_height, image_width, channels, device=device)
        render_n = torch.zeros(image_height, image_width, 3, device=device)

        visible = (radii[i] > 0).all(dim=-1)
        order = torch.argsort(depths[i], stable=True)
        for g in order[visible[order]].tolist():
            M = ray_transforms[i, g]
            h_u = pix[..., :1] * M[2] - M[0]  # [H, W, 3]
            h_v = pix[..., 1:] * M[2] - M[1]
            cross = torch.cross(h_u, h_v, dim=-1)
            uv = cross[..., :2] / cross[..., 2:]  # [H, W, 2]
            # keep grid_sample away from inf/nan; |uv| > 2 samples zero anyway
            uv = torch.nan_to_num(uv, nan=1e6, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)

            tid = int(texture_ids[i, g])
            tex_alpha = sample(texture_alphas[tid][..., None], uv)  # [H, W, 1]
            alpha = torch.clamp_max(opacities[i, g] * tex_alpha, MAX_ALPHA)
            skip = alpha < 1.0 / 255.0
            done = done | (~skip & (T * (1.0 - alpha) <= TRANSMITTANCE_THRESHOLD))
            alpha = torch.where(skip | done, torch.zeros_like(alpha), alpha)

            c = colors[i, g].expand(image_height, image_width, channels)
            if texture_colors is not None:
                tex_c = sample(texture_colors[tid], uv)
                c = c + F.pad(tex_c, (0, channels - tex_c.shape[-1]))
            vis = alpha * T
            render_c = render_c + vis * c
            render_n = render_n + vis * normals[i, g]
            T = T * (1.0 - alpha)

        out_colors.append(render_c)
        out_alphas.append(1.0 - T)
        out_normals.append(render_n)

    render_colors = torch.stack(out_colors).reshape(
        image_dims + (image_height, image_width, channels)
    )
    render_alphas = torch.stack(out_alphas).reshape(
        image_dims + (image_height, image_width, 1)
    )
    render_normals = torch.stack(out_normals).reshape(
        image_dims + (image_height, image_width, 3)
    )
    if backgrounds is not None:
        render_colors = render_colors + backgrounds[..., None, None, :] * (
            1.0 - render_alphas
        )
    return render_colors, render_alphas, render_normals


def _isect_tiles_2dgs(
    means2d: Tensor,  # [..., N, 2]
    radii: Tensor,  # [..., N, 2]
    ray_transforms: Tensor,  # [..., N, 3, 3]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    opacities: Optional[Tensor] = None,  # [..., N]
    uv_rects: Optional[Tensor] = None,  # [..., N, 4]
    samples_per_pixel: int = 4,
    chunk: int = 64,
) -> Tuple[Tensor, Tensor]:
    """Pytorch reference for `gsplat.cuda._wrapper.isect_tiles_2dgs()`.

    Evaluates each primitive's footprint the way the rasterizers do (ray-splat
    intersection, plus the low-pass disk for 2DGS) on a grid with
    `samples_per_pixel` samples per pixel that includes the tile borders, and
    marks every tile containing a covered sample.

    Returns:
        A tuple:

        - **Tile masks**. Bool [..., N, tile_height, tile_width].
        - **In front**. Bool [..., N]: whether the footprint lies in front of the
          camera, i.e. the kernel tests the exact footprint rather than the
          radii box fallback.
    """
    assert (opacities is None) != (uv_rects is None)
    lead = means2d.shape[:-1]
    device = means2d.device
    M = math.prod(lead)
    H = ray_transforms.reshape(M, 3, 3).double()
    H = H / H[:, 2:3, 2:3]
    mean2d = means2d.reshape(M, 2).double()
    visible = (radii.reshape(M, 2) > 0).all(dim=-1)

    if opacities is not None:
        opac = opacities.reshape(M).double()
        visible &= opac >= ALPHA_THRESHOLD
        t = torch.clamp(
            2.0 * torch.log(opac.clamp_min(1e-30) / ALPHA_THRESHOLD),
            max=GAUSSIAN_EXTEND**2,
        )
        in_front = t * (H[:, 2, 0] ** 2 + H[:, 2, 1] ** 2) < 1.0 - 1e-3
    else:
        rects = uv_rects.reshape(M, 4).double()
        visible &= (rects[:, 0] <= rects[:, 1]) & (rects[:, 2] <= rects[:, 3])
        us = rects[:, [0, 1, 1, 0]]
        vs = rects[:, [2, 2, 3, 3]]
        w = H[:, 2, 0:1] * us + H[:, 2, 1:2] * vs + H[:, 2, 2:3]
        in_front = (w > 1e-3).all(dim=-1)

    T = tile_size * samples_per_pixel
    xs = torch.arange(tile_width * T + 1, device=device, dtype=torch.float64)
    ys = torch.arange(tile_height * T + 1, device=device, dtype=torch.float64)
    xs = xs / samples_per_pixel
    ys = ys / samples_per_pixel
    py, px = torch.meshgrid(ys, xs, indexing="ij")  # [Hs, Ws]

    masks = torch.zeros(M, tile_height, tile_width, dtype=torch.bool, device=device)
    for start in range(0, M, chunk):
        sl = slice(start, min(start + chunk, M))
        h = H[sl, :, None, None, :]  # [m, 3, 1, 1, 3]
        h_u = px[..., None] * h[:, 2] - h[:, 0]  # [m, Hs, Ws, 3]
        h_v = py[..., None] * h[:, 2] - h[:, 1]
        cross = torch.linalg.cross(h_u, h_v)
        s = cross[..., :2] / cross[..., 2:3]
        if opacities is not None:
            tt = t[sl, None, None]
            d = torch.stack([px, py], dim=-1) - mean2d[sl, None, None, :]
            covered = ((s * s).sum(dim=-1) <= tt) | (
                FILTER_INV_SQUARE_2DGS * (d * d).sum(dim=-1) <= tt
            )
        else:
            r = rects[sl, None, None, :]
            covered = (
                (s[..., 0] >= r[..., 0])
                & (s[..., 0] <= r[..., 1])
                & (s[..., 1] >= r[..., 2])
                & (s[..., 1] <= r[..., 3])
            )
        covered &= visible[sl, None, None]
        # A sample on a tile border counts for both adjacent tiles.
        for i in range(tile_height):
            rows = covered[:, i * T : (i + 1) * T + 1]
            for j in range(tile_width):
                masks[sl, i, j] = rows[:, :, j * T : (j + 1) * T + 1].any(dim=(1, 2))
    return (
        masks.reshape(lead + (tile_height, tile_width)),
        (in_front & visible).reshape(lead),
    )


def _bbsplat_uv_rects(
    texture_alphas: Tensor,  # [..., S, S]
    opacities: Optional[Tensor] = None,  # [...]
) -> Tensor:
    """Pytorch implementation of `gsplat.cuda._wrapper.bbsplat_uv_rects()`."""
    S = texture_alphas.shape[-1]
    assert S >= 2, "BBSplat textures need S >= 2"
    # The bound must not drop texels that round up to ALPHA_THRESHOLD in the
    # rasterizer, hence the small slack.
    thr = ALPHA_THRESHOLD * (1.0 - 1e-4)
    if opacities is not None:
        thr = thr / opacities.clamp_min(1e-30)[..., None]
    cols = texture_alphas.amax(dim=-2) >= thr  # [..., S] (u)
    rows = texture_alphas.amax(dim=-1) >= thr  # [..., S] (v)

    idx = torch.arange(S, device=texture_alphas.device, dtype=torch.float32)
    big = float(S)

    def span(m: Tensor) -> Tuple[Tensor, Tensor]:
        lo = torch.where(m, idx, big).amin(dim=-1)
        hi = torch.where(m, idx, -big).amax(dim=-1)
        # texel x sits at u = 2x / (S - 1) - 1 and its bilinear support is (x - 1, x + 1)
        scale = 2.0 / (S - 1)
        return (lo - 1.0) * scale - 1.0, (hi + 1.0) * scale - 1.0

    u0, u1 = span(cols)
    v0, v1 = span(rows)
    rects = torch.stack([u0, u1, v0, v1], dim=-1)
    empty = ~cols.any(dim=-1)
    return torch.where(
        empty[..., None], rects.new_tensor([1.0, -1.0, 1.0, -1.0]), rects
    ).float()
