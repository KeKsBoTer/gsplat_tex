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

"""BBSplat trainer (BillBoard Splatting, Svitov et al. 2025).

Follows the reference implementation (github.com/david-svitov/BBSplat, train.py and
scene/gaussian_model.py): 2D billboards with learned per-primitive alpha and RGB
textures, MCMC-style relocation driven by the mean texture alpha, impact-weighted
texture regularizers, textures trained in a window of iterations, and geometry
frozen for the last iterations. Structure mirrors simple_trainer_2dgs.py.

Example (Mip-NeRF 360 settings from the reference scripts/train_all.sh):

    python simple_trainer_bbsplat.py --data_dir data/360_v2/counter --data_factor 2 \\
        --result_dir results/bbsplat_counter --cap_max 160000 --max_init_points 150000 \\
        --add_sky_box
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path
from gsplat.losses import l1_loss, ssim_loss
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization_bbsplat
from gsplat.strategy.ops import _update_param_with_optimizer
from gsplat_viewer_2dgs import GsplatRenderTabState, GsplatViewer
from nerfview import CameraState, RenderTabState, apply_float_colormap
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import knn, rgb_to_sh, set_random_seed


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt file. If provided, it will skip training and run evaluation only.
    ckpt: Optional[str] = None

    # Path to the COLMAP dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/bbsplat"
    # Every N images there is a test image
    test_every: int = 8
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Use a white instead of a black background
    white_background: bool = False

    # Port for the viewer server
    port: int = 8080
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 32_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000, 32_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000, 32_000])

    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000

    # Initialization: at most this many SfM points, chosen by farthest point sampling
    max_init_points: int = 140_000
    # Add points on a sphere around the scene for distant content (sky)
    add_sky_box: bool = False
    # Number of sky box points
    sky_box_points: int = 10_000
    # Texture resolution (S x S texels per billboard)
    texture_size: int = 16
    # Optional grayscale image for the initial alpha texture. The reference uses
    # assets/alpha_init_gaussian_small.png from the BBSplat repo. By default a
    # Gaussian blob fitted to that image is generated (max error < 0.01).
    alpha_init_path: Optional[str] = None
    # Std of the generated initial alpha blob, in texels of a 16x16 texture
    alpha_init_sigma: float = 2.7

    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.2
    # Far plane clipping distance
    far_plane: float = 1e10

    # Learning rates (reference: arguments/__init__.py)
    means_lr: float = 1.6e-4
    means_lr_final: float = 1.6e-6
    # Steps of the exponential means lr decay; geometry is frozen afterwards
    means_lr_max_steps: int = 30_000
    sh0_lr: float = 5e-3
    shN_lr: float = 5e-3 / 20
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    texture_alpha_lr: float = 1e-3
    texture_color_lr: float = 2.5e-3
    # Textures are only optimized in [texture_start_iter, texture_stop_iter)
    texture_start_iter: int = 500
    texture_stop_iter: int = 30_000

    # Normal consistency loss weight (reference default 0; 0.05 for DTU)
    normal_lambda: float = 0.0
    # Iteration to start normal consistency regularization
    normal_start_iter: int = 7_000
    # Distortion loss weight (reference default 0; 100 for DTU). Note that gsplat's
    # distortion is the L1 variant, so weights do not transfer one to one.
    dist_lambda: float = 0.0
    # Iteration to start distortion regularization
    dist_start_iter: int = 3_000
    # Depth used for the pseudo surface normals ("expected" = reference depth_ratio 0)
    depth_mode: Literal["expected", "median"] = "expected"

    # Texture regularization: pushes texture colors down and alpha textures back to
    # their initialization, weighted by (max_impact - impact) so that billboards with
    # little blending weight are regularized more.
    texture_color_reg: float = 1e-4
    texture_alpha_reg: float = 1e-4
    max_impact: float = 100.0
    # Regularization of the mean alpha texture value (MCMC opacity regularization)
    alpha_reg: float = 0.01

    # Relocation / growth (MCMC sampler of the reference)
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    refine_every: int = 100
    # Billboards with mean texture alpha below this are relocated
    dead_alpha: float = 0.005
    # Maximum number of billboards
    cap_max: int = 160_000
    # Grow the number of billboards by this fraction per refinement
    grow_rate: float = 0.05
    # Position noise (reference noise_lr). With the reference's fixed opacity of 1
    # the noise scale is ~1e-43, i.e. effectively disabled; kept for parity.
    noise_lr: float = 5e5

    # Use packed mode for rasterization
    packed: bool = False
    # Update only billboards inside the current view (radii > 0) with gsplat's fused
    # Adam kernel (Taming-3DGS "visible Adam"; no bias correction, like
    # simple_trainer.py --visible_adam). The reference uses dense torch Adam; set
    # --no-selective-adam for that.
    selective_adam: bool = True

    # Dump information to tensorboard every this steps
    tb_every: int = 100

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.means_lr_max_steps = int(self.means_lr_max_steps * factor)
        self.texture_start_iter = int(self.texture_start_iter * factor)
        self.texture_stop_iter = int(self.texture_stop_iter * factor)
        self.normal_start_iter = int(self.normal_start_iter * factor)
        self.dist_start_iter = int(self.dist_start_iter * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.refine_every = max(1, int(self.refine_every * factor))


class SigmoidWithMean(torch.autograd.Function):
    """sigmoid(x) and its per-billboard mean, with a single fused backward.

    The gradient of a per-billboard mean is constant over that billboard's texels,
    so it is folded into the sigmoid backward instead of being broadcast to a
    full-size tensor and accumulated by autograd.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tuple[Tensor, Tensor]:
        ctx.set_materialize_grads(False)
        s = torch.sigmoid(x)
        ctx.save_for_backward(s)
        return s, s.flatten(1).mean(1)

    @staticmethod
    def backward(ctx, v_s: Optional[Tensor], v_mean: Optional[Tensor]):
        (s,) = ctx.saved_tensors
        if v_mean is not None:
            v_mean = (v_mean / s[0].numel()).view(-1, *[1] * (s.dim() - 1))
            v_s = v_mean.expand_as(s) if v_s is None else v_s + v_mean
        return None if v_s is None else torch.ops.aten.sigmoid_backward(v_s, s)


def farthest_point_sampling(points: Tensor, k: int) -> Tensor:
    """Indices of `k` farthest points, starting at index 0 (like pytorch3d)."""
    n = points.shape[0]
    idxs = torch.empty(k, dtype=torch.long, device=points.device)
    dists = torch.full((n,), float("inf"), device=points.device)
    last = torch.zeros((), dtype=torch.long, device=points.device)
    for i in range(k):
        idxs[i] = last
        dists = torch.minimum(dists, (points - points[last]).square().sum(-1))
        last = torch.argmax(dists)
    return idxs


def fibonacci_sphere(n: int) -> Tensor:
    i = torch.arange(n, dtype=torch.float64)
    y = 1 - i / (n - 1) * 2
    r = torch.sqrt(1 - y * y)
    theta = math.pi * (math.sqrt(5.0) - 1.0) * i
    return torch.stack([torch.cos(theta) * r, y, torch.sin(theta) * r], -1).float()


def initial_alpha_texture(cfg: Config) -> Tensor:
    """[S, S] alpha texture in [0, 1] before the reference's 0.5 scaling."""
    if cfg.alpha_init_path is not None:
        image = imageio.imread(cfg.alpha_init_path)
        if image.ndim == 3:
            image = image[..., 0]
        return torch.from_numpy(image.astype(np.float32) / 255.0)
    S = cfg.texture_size
    sigma = cfg.alpha_init_sigma * S / 16.0
    x = torch.arange(S, dtype=torch.float32) - (S - 1) / 2
    return torch.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma**2))


def create_splats_with_optimizers(
    cfg: Config, parser: Parser, scene_scale: float, device: str = "cuda"
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    points = torch.from_numpy(parser.points).float()
    rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    if len(points) >= cfg.max_init_points:
        idxs = farthest_point_sampling(points.to(device), cfg.max_init_points).cpu()
        points, rgbs = points[idxs], rgbs[idxs]
    if cfg.add_sky_box:
        radius = points.abs().max()
        sky = fibonacci_sphere(cfg.sky_box_points) * radius
        points = torch.cat([sky, points])
        rgbs = torch.cat([torch.ones_like(sky), rgbs])

    N = points.shape[0]
    # Billboard half size = average distance of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1).clamp_min(1e-7)
    scales = torch.log(torch.sqrt(dist2_avg)).unsqueeze(-1).repeat(1, 2)  # [N, 2]
    quats = torch.rand((N, 4))  # [N, 4]

    colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = rgb_to_sh(rgbs) - 0.1

    alpha_init = initial_alpha_texture(cfg)  # [S, S]
    S = alpha_init.shape[-1]
    texture_alpha = torch.logit(0.5 * alpha_init).expand(N, S, S).clone()
    texture_color = torch.full((N, S, S, 3), math.log(0.1 / 0.9))  # sigmoid -> 0.1

    params = [
        # name, value, lr (texture lrs start at 0 and are switched on later)
        ("means", points, cfg.means_lr * scene_scale),
        ("scales", scales, cfg.scales_lr),
        ("quats", quats, cfg.quats_lr),
        ("sh0", colors[:, :1, :], cfg.sh0_lr),
        ("shN", colors[:, 1:, :], cfg.shN_lr),
        ("texture_alpha", texture_alpha, 0.0),
        ("texture_color", texture_color, 0.0),
    ]
    splats = torch.nn.ParameterDict(
        {n: torch.nn.Parameter(v) for n, v, _ in params}
    ).to(device)
    if cfg.selective_adam:
        optimizers = {
            name: SelectiveAdam(
                [{"params": splats[name], "lr": lr}], eps=1e-15, betas=(0.9, 0.999)
            )
            for name, _, lr in params
        }
    else:
        optimizers = {
            name: torch.optim.Adam([{"params": splats[name], "lr": lr}], eps=1e-15)
            for name, _, lr in params
        }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(self.parser, split="train")
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        self.splats, self.optimizers = create_splats_with_optimizers(
            cfg, self.parser, self.scene_scale, self.device
        )
        print("Model initialized. Number of billboards:", len(self.splats["means"]))
        self.texture_alpha_init = torch.sigmoid(
            self.splats["texture_alpha"][:1]
        ).detach()

        bg = 1.0 if cfg.white_background else 0.0
        self.background = torch.full((1, 3), bg, device=self.device)

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        textures: Optional[Tuple[Tensor, Tensor]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Tensor, Tensor, Dict]:
        """`textures` are the activated (alpha, color) textures; computed if None."""
        splats = self.splats
        if textures is None:
            textures = self.activated_textures()[:2]
        N = len(splats["means"])
        # Billboards are planar; a unit third scale keeps the rendered normals unit length.
        scales = torch.cat(
            [torch.exp(splats["scales"]), torch.ones(N, 1, device=self.device)], -1
        )
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]
        kwargs.setdefault("backgrounds", self.background.expand(len(Ks), -1))
        return rasterization_bbsplat(
            means=splats["means"],
            quats=splats["quats"],
            scales=scales,
            opacities=None,  # reference BBSplat: alpha comes from the texture only
            colors=colors,
            texture_alphas=textures[0],
            texture_colors=textures[1],
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            depth_mode=self.cfg.depth_mode,
            **kwargs,
        )

    def activated_textures(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """(alpha, color) textures and their per-billboard means."""
        alpha, alpha_mean = SigmoidWithMean.apply(self.splats["texture_alpha"])
        color, color_mean = SigmoidWithMean.apply(self.splats["texture_color"])
        return alpha, color, alpha_mean, color_mean

    # ------------------------------------------------------------------
    # Learning rate schedules
    # ------------------------------------------------------------------
    def _set_lr(self, name: str, lr: float):
        for group in self.optimizers[name].param_groups:
            group["lr"] = lr

    def update_learning_rates(self, iteration: int):
        cfg = self.cfg
        if iteration > cfg.means_lr_max_steps:
            # Freeze everything but the higher order SH (reference:
            # deactivate_gaussians_training).
            for name in self.optimizers:
                if name != "shN":
                    self._set_lr(name, 0.0)
            return 0.0
        # Exponential means lr decay (reference get_expon_lr_func)
        t = min(max(iteration / cfg.means_lr_max_steps, 0.0), 1.0)
        means_lr = math.exp(
            math.log(cfg.means_lr) * (1 - t) + math.log(cfg.means_lr_final) * t
        )
        means_lr *= self.scene_scale
        self._set_lr("means", means_lr)
        textures_on = cfg.texture_start_iter <= iteration < cfg.texture_stop_iter
        self._set_lr("texture_alpha", cfg.texture_alpha_lr if textures_on else 0.0)
        self._set_lr("texture_color", cfg.texture_color_lr if textures_on else 0.0)
        return means_lr

    # ------------------------------------------------------------------
    # Relocation and growth (reference: relocate_gs / add_new_gs)
    # ------------------------------------------------------------------
    def _mean_alpha(self) -> Tensor:
        return torch.sigmoid(self.splats["texture_alpha"]).flatten(1).mean(1)

    def _split_alpha(self, sampled: Tensor) -> Tensor:
        """Alpha textures for a source billboard copied n times: 1 - (1 - a)^(1/(n+1))."""
        alpha = torch.sigmoid(self.splats["texture_alpha"][sampled])
        ratio = torch.bincount(sampled, minlength=len(self.splats["means"]))
        n = (ratio[sampled] + 1).float()[:, None, None]
        new_alpha = 1.0 - torch.pow(1.0 - alpha, 1.0 / n)
        return torch.logit(new_alpha.clamp(1e-6, 1 - 1e-6))

    @staticmethod
    def _sample(probs: Tensor, n: int) -> Tensor:
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        return torch.multinomial(probs, n, replacement=True)

    @torch.no_grad()
    def relocate(self) -> int:
        mean_alpha = self._mean_alpha()
        dead = mean_alpha <= self.cfg.dead_alpha
        dead_idx = dead.nonzero(as_tuple=True)[0]
        alive_idx = (~dead).nonzero(as_tuple=True)[0]
        if len(dead_idx) == 0 or len(alive_idx) == 0:
            return 0
        sampled = alive_idx[self._sample(mean_alpha[alive_idx], len(dead_idx))]
        new_alpha = self._split_alpha(sampled)

        def param_fn(name: str, p: Tensor) -> Tensor:
            if name == "texture_alpha":
                p[sampled] = new_alpha
            p[dead_idx] = p[sampled]
            return torch.nn.Parameter(p, requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            v[sampled] = 0
            return v

        _update_param_with_optimizer(
            param_fn, optimizer_fn, self.splats, self.optimizers
        )
        return len(dead_idx)

    @torch.no_grad()
    def grow(self) -> int:
        n_cur = len(self.splats["means"])
        n_new = min(self.cfg.cap_max, int((1.0 + self.cfg.grow_rate) * n_cur)) - n_cur
        if n_new <= 0:
            return 0
        sampled = self._sample(self._mean_alpha(), n_new)
        new_alpha = self._split_alpha(sampled)

        def param_fn(name: str, p: Tensor) -> Tensor:
            if name == "texture_alpha":
                p[sampled] = new_alpha
            return torch.nn.Parameter(
                torch.cat([p, p[sampled]]), requires_grad=p.requires_grad
            )

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            v[sampled] = 0
            return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), device=v.device)])

        _update_param_with_optimizer(
            param_fn, optimizer_fn, self.splats, self.optimizers
        )
        return n_new

    @torch.no_grad()
    def inject_noise(self, means_lr: float):
        """Reference MCMC position noise with the opacity fixed to 1."""

        def op_sigmoid(x, k=100.0, x0=0.995):
            return 1.0 / (1.0 + math.exp(-k * (x - x0)))

        scale = op_sigmoid(1.0 - 1.0) * self.cfg.noise_lr * means_lr
        if scale < 1e-20:  # always the case with the reference settings
            return
        quats = F.normalize(self.splats["quats"], dim=-1)
        w, x, y, z = quats.unbind(-1)
        R = torch.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
            -1,
        ).reshape(-1, 3, 3)
        s = torch.exp(self.splats["scales"])
        L = R[..., :2] * s[:, None, :]  # [N, 3, 2], third scale is 0
        noise = torch.randn_like(self.splats["means"]) * scale
        noise = (L @ (L.transpose(1, 2) @ noise[..., None]))[..., 0]
        self.splats["means"].add_(noise)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        device = self.device

        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        global_tic = time.time()
        pbar = tqdm.tqdm(range(max_steps))
        for step in pbar:
            iteration = step + 1  # the reference counts iterations from 1
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            height, width = pixels.shape[1:3]

            means_lr = self.update_learning_rates(iteration)
            sh_degree_to_use = min(iteration // cfg.sh_degree_interval, cfg.sh_degree)
            # Activate the textures once; the regularizers below reuse them.
            (
                texture_alpha,
                texture_color,
                alpha_mean,
                color_mean,
            ) = self.activated_textures()

            (
                renders,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                _,
                info,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                textures=(texture_alpha, texture_color),
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                distloss=cfg.dist_lambda > 0,
                compute_impact=True,
            )
            colors = renders[..., :3]

            # photometric loss
            l1loss = l1_loss(colors, pixels).mean()
            ssimloss = ssim_loss(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
            loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)

            # geometric regularization
            normal_lambda = (
                cfg.normal_lambda if iteration > cfg.normal_start_iter else 0.0
            )
            dist_lambda = cfg.dist_lambda if iteration > cfg.dist_start_iter else 0.0
            normalloss = torch.zeros((), device=device)
            if normal_lambda > 0:
                surf = normals_from_depth * alphas.detach()
                normal_error = 1 - (normals * surf).sum(dim=-1)
                normalloss = normal_lambda * normal_error.mean()
            distloss = dist_lambda * render_distort.mean()

            # texture regularization weighted by the per-billboard impact
            impacts = info["impacts"]
            if cfg.packed:
                impact = torch.zeros(len(self.splats["means"]), device=device)
                impact.index_add_(0, info["gaussian_ids"].long(), impacts)
            else:
                impact = impacts.reshape(-1)
            visible = impact > 0
            weights = cfg.max_impact - impact[visible].clamp(0, cfg.max_impact)
            # Uses the per-billboard texture means [N];
            # mean(alpha - init) = mean(alpha) - mean(init).
            n_visible = visible.sum().clamp_min(1)  # avoid NaN for empty views
            texreg = (color_mean[visible] * weights).sum() / n_visible
            texreg = texreg * cfg.texture_color_reg
            alpha_dev = alpha_mean[visible] - self.texture_alpha_init.mean()
            texreg_alpha = (alpha_dev * weights).abs().sum() / n_visible
            texreg = texreg + texreg_alpha * cfg.texture_alpha_reg

            total = loss + normalloss + distloss + texreg
            total = total + cfg.alpha_reg * alpha_mean.mean()
            total.backward()

            desc = (
                f"loss={loss.item():.3f}| sh degree={sh_degree_to_use}| "
                f"#BB={len(self.splats['means'])}"
            )
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/normalloss", normalloss.item(), step)
                self.writer.add_scalar("train/distloss", distloss.item(), step)
                self.writer.add_scalar("train/texreg", texreg.item(), step)
                self.writer.add_scalar("train/num_BB", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                self.writer.flush()

            # Relocation / growth before the optimizer step, as in the reference.
            # The replaced parameters carry no gradient, so this step skips updates.
            if (
                cfg.refine_start_iter < iteration < cfg.refine_stop_iter
                and iteration % cfg.refine_every == 0
            ):
                n_relocated = self.relocate()
                n_added = self.grow()
                self.writer.add_scalar("train/num_relocated", n_relocated, step)
                self.writer.add_scalar("train/num_added", n_added, step)
                torch.cuda.empty_cache()

            if iteration < max_steps:
                visibility = None
                if cfg.selective_adam:
                    if cfg.packed:
                        visibility = torch.zeros(
                            len(self.splats["means"]), dtype=torch.bool, device=device
                        )
                        visibility[info["gaussian_ids"].long()] = True
                    else:
                        visibility = (info["radii"] > 0).all(-1).any(0)
                for optimizer in self.optimizers.values():
                    if visibility is None:
                        optimizer.step()
                    elif len(visibility) == len(self.splats["means"]):
                        # (after relocation/growth the new parameters have no
                        # gradient and nothing is updated this step)
                        optimizer.step(visibility)
                    optimizer.zero_grad(set_to_none=True)
                if iteration <= cfg.means_lr_max_steps:
                    self.inject_noise(means_lr)

            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_BB": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {"step": step, "splats": self.splats.state_dict()},
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            renders, _, normals, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )
            colors = torch.clamp(renders[..., :3], 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )
            normals = (normals * 0.5 + 0.5).squeeze(0).clamp(0, 1).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_normal_{step}.png",
                (normals * 255).astype(np.uint8),
            )

            pixels = pixels.permute(0, 3, 1, 2)
            colors = colors.permute(0, 3, 1, 2)
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        ellipse_time /= len(valloader)
        stats = {
            "psnr": torch.stack(metrics["psnr"]).mean().item(),
            "ssim": torch.stack(metrics["ssim"]).mean().item(),
            "lpips": torch.stack(metrics["lpips"]).mean().item(),
            "ellipse_time": ellipse_time,
            "num_BB": len(self.splats["means"]),
        }
        print(
            f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, "
            f"LPIPS: {stats['lpips']:.3f} Time: {ellipse_time:.3f}s/image "
            f"Number of billboards: {stats['num_BB']}"
        )
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]
        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)
            depths = renders[0, ..., 3:4]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            writer.append_data((canvas.cpu().numpy() * 255).astype(np.uint8))
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        K = (
            torch.from_numpy(camera_state.get_K((width, height)))
            .float()
            .to(self.device)
        )

        (
            render_colors,
            render_alphas,
            render_normals,
            _,
            _,
            render_median,
            info,
        ) = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            render_mode="RGB+ED",
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
        )
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "depth":
            depth = render_median[0]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "normal":
            renders = (render_normals[0] * 0.5 + 0.5).cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        else:
            renders = render_colors[0, ..., 0:3].clamp(0, 1).cpu().numpy()
        return renders


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
