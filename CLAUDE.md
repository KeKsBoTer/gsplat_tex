# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repo (`gsplat_tex`) is a fork of [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat): a CUDA-accelerated Gaussian splatting rasterization library with PyTorch bindings (3DGS, 2DGS, 3DGUT, LiDAR, sparse rendering).

## Setup and build

```bash
git submodule update --init --recursive    # glm (CUDA kernels) and googletest (C++ tests) are required
BUILD_NO_CUDA=1 pip install -e .[dev]      # skip AOT CUDA build; JIT-compile on first use (preferred when editing CUDA)
pip install -e .[dev]                      # AOT-compile CUDA during install
pip install -r examples/requirements.txt --no-build-isolation   # extra deps for examples/
./bootstrap.sh                             # installs lint/format-code.sh as pre-commit hook
```

JIT builds are incremental and cached under `~/.cache/torch_extensions/py*-cu*/`. To trigger/debug a build directly:

```bash
VERBOSE=1 DEBUG=1 TORCH_CUDA_ARCH_LIST="8.9" python -c "from gsplat.cuda._backend import _C"
```

Build knobs read in `gsplat/cuda/build.py` (env vars): `DEBUG`, `VERBOSE`, `FAST_MATH` (default on), `MAX_JOBS`, `NVCC_FLAGS`, `BUILD_3DGS` / `BUILD_2DGS` / `BUILD_3DGUT` / `BUILD_ADAM` / `BUILD_RELOC` / `BUILD_LOSSES` / `BUILD_BBSPLAT` (feature subsets; `BUILD_BBSPLAT` also pulls in the 2DGS projection), `BUILD_CAMERA_WRAPPERS`, and `NUM_CHANNELS` (comma-separated list of color channel counts the rasterizer kernels are instantiated for). Changing these changes the JIT cache key, so expect a rebuild. Python checks for compiled features via `gsplat.cuda._wrapper.has_3dgs()`, `has_3dgut()`, etc.

## Tests and formatting

```bash
pytest tests/                                   # all tests (most need a CUDA GPU)
pytest tests/test_basic.py                      # one file
pytest tests/test_basic.py::test_projection     # one test
pytest tests/test_basic.py -k "pinhole"         # filter parametrizations
pytest -m "not gradcheck"                       # skip slow double-precision gradchecks
pytest tests/test_cpp.py                        # builds and runs the GoogleTest C++ tests in tests/cpp/
```

`pytest.ini` sets `VERBOSE=1`, `BUILD_CAMERA_WRAPPERS=1` and a wide `NUM_CHANNELS` list, so the extension built under pytest differs from the one built for normal use (separate JIT build, slow first run). Public CI only runs CPU tests (torch CPU, `BUILD_NO_CUDA=1`); GPU tests run on a self-hosted runner, so run GPU tests locally before pushing CUDA changes. `GPU_CI_XFAIL=1` marks known FP-marginal failures listed in the root `conftest.py`.

```bash
lint/format-code.sh                  # format all tracked files (black 22.3.0 for Python, clang-format for C++/CUDA)
lint/format-code.sh --check          # CI check mode
lint/format-code.sh --changed main   # only files changed since a ref
```

The clang-format version is pinned in `config.yaml`; CI fails on any formatting diff.

Source files carry SPDX Apache-2.0 license headers; keep them on new files.

## Architecture

**Main rendering path.** `gsplat.rendering.rasterization()` (`gsplat/rendering.py`) is the main entry point. It takes means/quats/scales/opacities/colors(SH) plus `viewmats`/`Ks` with arbitrary leading batch dims, and dispatches to one of these pipelines:
1. Projection: `fully_fused_projection` (EWA for 3DGS, unscented transform for 3DGUT with non-pinhole cameras and rolling shutter), packed or dense.
2. Optional SH evaluation into colors.
3. Tile intersection (`isect_tiles`, `isect_offset_encode`; optional AccuTile ellipse test, sparse active-tile variants).
4. Per-tile alpha compositing: `rasterize_to_pixels` (3DGS), or `rasterize_to_pixels_eval3d` (3DGUT / LiDAR, evaluated in world space).

`rasterization_2dgs()` is the 2DGS path. `*_inria_wrapper` functions mirror the original Inria implementation for comparison.

**Python ↔ CUDA layering for core ops.** `gsplat/cuda/_wrapper.py` defines the Python ops and `torch.autograd.Function` / custom-op autograd registrations. It calls into `_C`, which `gsplat/cuda/_backend.py` loads either from a prebuilt `gsplat.csrc` or by JIT compiling through `build.py`. Bindings live in `gsplat/cuda/ext.cpp`. Kernels live in `gsplat/cuda/csrc/`, where each op is split into a `Foo.h`/`Foo.cpp` launcher (tensor checks, dispatch) and `FooCUDA.cu` / `*Fwd.cu` / `*Bwd.cu` kernels. Every CUDA op has a pure-PyTorch reference in `gsplat/cuda/_torch_impl*.py`, and tests compare the kernels against it for both forward and gradients. A new op needs: kernel, launcher, binding in `ext.cpp`, wrapper plus autograd in `_wrapper.py`, a torch reference, and a test.

**Shared modules** (`gsplat/geometry`, `gsplat/sensors`, `gsplat/scene`, `gsplat/stage`) follow the pattern in `docs/modules-design.md`: public stateless API in `functional/`; backend dispatch, autograd, and their own separately built CUDA extension in `kernels/` (`kernels/_backend.py`, `kernels/cuda/{build.py,ext.cpp,csrc}`); optional `models/` (nn.Module) and `components/` (non-Module stateful). Each module has a `design.md` with domain-specific rules. Read it before changing the module.

**Training-side pieces.**
- `gsplat/strategy/`: densification strategies (`DefaultStrategy`, `MCMCStrategy`). They mutate params and optimizer state in place via `strategy/ops.py`.
- `gsplat/optimizers/`, `relocation.py`, `losses*.py`, `compression/`: training support code.
- `gsplat/distributed.py`: multi-GPU support.
- `gsplat/experimental/`: the HiGS inference-only renderer (fp16 scene packing, macro-tile rasterization).

**Examples.** `examples/simple_trainer.py` is the reference 3DGS trainer (COLMAP data, tyro CLI, e.g. `python simple_trainer.py default --data_dir ... --result_dir ...`). The 2DGS, AV, and dynamic-surgical trainers, plus the viewers, live alongside it. `examples/datasets/` holds the data loaders. Benchmarks are in `examples/benchmarks/`.

Further docs: `docs/DEV.md` (dev workflow, clangd setup), `docs/3dgut.md`, `docs/batch.md` (batching semantics), `EXPLORATION.md` (trainer flag experiments).
