# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.9.0] - 2026-10-11

### Added

- **Dense 2D regression**: Training, validation, cross-validation, and inference now accept scalar `(N, H, W)` and channels-first multi-property `(N, C, H, W)` targets while preserving the existing `(N, T)` vector workflow
- **Models**: Added `unet_field`, a dense U-Net head that predicts `(N, C, H_out, W_out)` fields and supports output grids that differ in size from the 2D input grid
- **Inference**: Dense predictions can be saved to compressed NPZ; field evaluation reports channel-aware metrics and plots target, prediction, and error maps

### Changed

- **Runtime baseline**: Require PyTorch 2.13+, torchvision 0.28+, and ONNX 1.22+ to exclude older dependency versions affected by published advisories; use a dedicated, audited deployment environment
- **Training/HPO**: Add `--selection_metric loss|mse`; HPO compares normalized MSE across losses, seeds sampling, validates persistent-study signatures, and leases visible GPUs exclusively
- **Recovery**: `--fresh` archives previous training artifacts; incompatible resume metadata is rejected and completed experiments exit without restarting
- **Security**: Legacy NPZ object arrays require explicit `--allow_pickle`; checkpoint and cache metadata still require trusted sources
- **Release quality**: CI checks minimum/latest PyTorch, the full CPU architecture suite, distributed training, source/wheel distributions, and installed entry points before publishing verified artifacts

- **Target scaling**: Dense fields use one mean and standard deviation per property channel, fitted over training samples and both spatial axes; checkpoint metadata preserves the canonical target shape
- **Validation**: Model/target shape compatibility is checked explicitly to prevent vector architectures from silently broadcasting against dense fields
- **HPO**: Target-aware sweeps select `unet_field` and smaller batch candidates for dense maps, and reject incompatible explicit model lists before launching trials
- **Metrics/export**: Public R²/Pearson helpers accept dense fields, and ONNX field exports preserve the requested opset, dynamic batch metadata, and channel-wise de-normalization
- **Discovery/docs**: CLI model listings identify vector versus dense output contracts; configuration, CV, custom-model, loss, constraint, and data-shape guidance now cover dense fields

### Fixed

- Correct Mamba/Vim selective-scan values and gradients for realistic decay, long sequences, odd chunks, and length-one inputs
- Preserve frozen backbone evaluation state, exact head parameter selection, and timm convolution padding/device/dtype during channel adaptation
- Prevent FP16 GRN/log-cosh overflow, `DropPath(1)` NaNs, WaveNet even-kernel shape errors, and default-configuration constructor collisions
- Weight accumulated gradients by sample count, including partial windows; step OneCycle per optimizer update and epoch schedulers before saving resume state
- Propagate training interrupts and inference/export failures; publish ONNX files only after successful validation
- Correct distributed min/max reductions and scalar broadcast precision, unit-invariant correlations, large-value inference metrics, and short multistep schedules
- Validate finite real data, configuration ranges, constraints, cache completion, sparse MATLAB groups, and bounded preprocessing chunk sizes
- Make notebook setup repeatable and fail visibly on subprocess errors; align MATLAB R² aggregation with Python metrics
- Remove the pre-commit hook that recursively deleted cache/environment directories; include tests, configuration, and release tools in source distributions

- Preserve large-offset physical values during dense normalization, constraint evaluation, ONNX de-normalization, and R²/Pearson calculation; use stable streaming validation statistics
- Validate target values, spatial dimensions, channel weights, and prediction shapes before accepting incompatible data or losses
- Preserve normalization when resuming, validate it before rebuilding caches, and include scalers in periodic and emergency checkpoints; old experimental dense normalization requires retraining
- Preserve the published WaveNet optimizer layout while freezing unused parameters. Version Mamba/Vim scan weights and reject legacy unversioned weights rather than silently changing predictions; use the original release for legacy inference/resume or retrain with the corrected scan
- Detect changes at the end of small cached datasets and rebuild dense caches when the normalization version changes
- Forward channel weights, Huber delta, and pretrained settings through CV; reject unsupported CV constraints/gradient accumulation, and keep HPO initialization consistent between execution modes
- Handle singleton U-Net bottlenecks with small channel counts and validate invalid architecture settings before construction
- Canonicalize scalar-map reporting and prediction files, validate channel names, and make architecture tests independent of pretrained downloads
- Restore sync-free training steps: non-finite losses are checked with one collective per optimizer update (not per micro-batch); validation is scored once on rank 0 and the loss is broadcast so plateau schedulers stay identical across ranks
- Score dense test sets in bounded-memory streaming chunks and fit dense target scalers from contiguous reads, avoiding full-size float64 copies and per-sample HDF5/MAT reads

> `unet_field` is a supervised dense inversion model suitable for learned property mapping or FWI initialization. It does not implement full waveform inversion, which still requires a differentiable wave solver and an observed-versus-simulated data-consistency loop.

## [1.8.1] - 2026-06-18

### Added
- **Training**: `--grad_accum_steps` flag for gradient accumulation across mini-batches

### Fixed
- **Training**: Auto-resume picks latest checkpoint (interrupted > periodic > best) with deferred cleanup and shared-memory probe for worker auto-detect
- **Config**: CLI-over-YAML precedence handles `--flag=value` form; `validate_config()` derives valid keys from parser
- **Inference**: Model auto-detection uses longest-prefix matching (fixes multi-underscore IDs)
- **DDP**: Epoch-based schedulers now step only on rank 0 with LR broadcast (cosine `T_max` was consumed N× faster with N GPUs)
- **Training**: Per-batch schedulers correctly skipped during gradient accumulation sub-steps
- **ConvNeXt V1/V2**: Pretrained V2 now loads real V2 weights from `timm` (was V1 torchvision); added ≥32 min spatial size guard; exempt `res_scale` from weight decay; added `res_scale` init=1e-6 for stable from-scratch training; fixed `_init_weights` docstring
- **Mamba/Vim**: Corrected the selective-scan recurrence (off-by-one in the prefix product) — all Mamba/Vim models were computing a non-Mamba operator
- **HPO**: A single failed trial no longer aborts the whole study (`study.optimize(catch=…)`); `--n_jobs` is validated; GPU count detected once instead of per-trial; warns when resuming a persistent study with a changed search space
- **DDP**: Validation loss (early-stopping / best-model selection) is now identical on single- vs multi-GPU when physics constraints are active
- **Data**: MAT memory-mapped loading applies MATLAB output normalization (fixes wrong `out_dim`/crash on `(1, N)` targets); scaler cache invalidates when `val_split`/`seed` change (prevents validation-set leakage)
- **Schedulers**: `exponential` no longer halves the LR every epoch from the shared `--scheduler_factor`; removed the fragile `gamma == 0.99` sentinel
- **Metrics**: Publication scatter R² is computed on the full set (matches the reported metric) with seeded subsampling; guarded `np.corrcoef` against constant-error NaNs
- **Robustness**: Construction-time min-size guards (CNN, U-Net, UniRepLKNet); native YAML lists for `betas`/`milestones`/`loss_weights` no longer crash; `validate_config()` checks `choices`; physics-constraint `log`/`sqrt` domain-protected; timm load failures use exception chaining

## [1.8.0] - 2026-02-23

### Added
- **Models**: 8 new architectures (69 → 71 public, after removing 6):
  - **WaveNet** (small/base/large): Gated dilated convolutional network adapted for 1D waveform regression. Signature `tanh×sigmoid` gated activation, skip connections summed across all dilation layers, same-padding (non-causal). ~1M/4M/15M params. 1D-only.
  - **S4D** (small/base/large): Diagonal Structured State Space Model (S4D-Lin kernel). Computed as FFT convolution — O(L log L), fully vectorized, `torch.compile`-safe, MPS-compatible. HiPPO-LegS initialization. ~0.8M/3.2M/11M params. 1D-only.
  - **EfficientNet-B4**: Medium tier, ~19M params. Torchvision pretrained. 2D-only.
  - **EfficientNet-B7**: Large tier, ~66M params. Torchvision pretrained. 2D-only.

### Removed
- **Models**: 6 variants pruned to clean up redundant parameter-count tiers:
  - `efficientnet_b1` (7.8M) — sandwiched between B0 (5.3M) and B2 (9.1M)
  - `efficientvit_m0` (2.2M), `efficientvit_m2` (3.8M) — M-series clusters 2–4M; keep M1 only
  - `efficientvit_b0` (2.1M) — duplicates M-series range
  - `efficientvit_b3` (46M), `efficientvit_l1` (49M) — near-identical; keep L2 (60M) only
- **Resulting EfficientNet tier**: B0 (5.3M) → B2 (9.1M) → B4 (19M) → B7 (66M)
- **Resulting EfficientViT tier**: M1 (2.6M) → B1 (7.5M) → B2 (21.8M) → L2 (60.5M)

### Fixed
- **DenseNet**: Replaced `MaxPool` with `AvgPool` in stem — workaround for a Triton compiler bug that causes incorrect gradients on large tensors when using `--compile`
- **MaxViT**: Replaced hardcoded `_DIVISOR = 28` with a `_NATIVE_SIZES` map (`224 → 224`, `384 → 384`, `512 → 512`); `img_size` is now passed to `timm.create_model()` so attention windows are pre-configured for the actual input resolution
- **HPC plotting**: Added `matplotlib.use("Agg")` before `pyplot` import in `train.py`, `test.py`, and `utils/metrics.py` — prevents crash on headless compute nodes that have no `$DISPLAY`

## [1.7.1] - 2026-02-14

### Added
- **Cross-validation**: MPS (Apple Silicon GPU) device support — auto-detects CUDA → MPS → CPU
- **HPO**: SQLite study persistence (`--storage`) — interrupted searches resume automatically
- **ConvNeXt**: Stochastic depth (DropPath) via `timm.layers.DropPath` with linearly-increasing rates (was no-op `nn.Identity()`)

### Changed
- **CI**: All workflows now test against Python `3.11`/`3.12`/`3.13` matrix
- **Tests**: Regression tests rewritten to call production code (`hpo.main()`, `export_to_onnx()`, `_save_best_checkpoint()`) instead of replaying logic inline

### Fixed
- **Critical**: Auto-resume duplicated epochs — history now truncated to `start_epoch` on resume
- **Critical**: Cross-validation OOM (SIGKILL:9) — `CVDataset` uses zero-copy `torch.from_numpy()` instead of `torch.tensor()`
- **Critical**: Release workflow race — split matrix job so GitHub Release is created once, not per Python version
- **Critical**: Inference crash on empty datasets — `run_inference()` guards empty predictions; `main()` early-returns before scaler validation
- **Models**: Pretrained timm wrappers (CaFormer, EfficientViT, FastViT, MaxViT) now probe feature dims in `eval()` mode to preserve BatchNorm running stats
- **Cross-validation**: `StratifiedKFold` gracefully falls back to `KFold` when bins have too few samples
- **Training**: Warning suppression scoped to known-noisy libraries (`sklearn`, `timm`, `torchvision`, `scipy`) instead of blanket `FutureWarning`/`DeprecationWarning` filter
- **Metrics**: Relative-error plots and CDF percentile markers now exclude `NaN` from near-zero targets (was mapping to 0%, understating error)
- **Metrics**: 5 plot functions (`plot_correlation_heatmap`, `plot_relative_error`, `plot_error_cdf`, `plot_prediction_vs_index`, `plot_error_boxplot`) now call `_ensure_style_configured()` for consistent styling
- **Cross-validation**: Fold-level `gc.collect()` + `torch.cuda.empty_cache()`; scheduler/optimizer args no longer silently dropped; `pin_memory` conditional on CUDA
- **DDP**: ReduceLROnPlateau broadcasts per-group LRs (preserves multi-group ratios)
- **HPO**: Crashed subprocess trials return `inf`; `--inprocess` forces `n_jobs=1`; empty `nvidia-smi` no longer yields phantom GPU
- **Launcher**: W&B default changed to `"online"` (fixes spurious offline sync messages on local machines)
- **Training**: Mixed-precision log shows actual `accelerator.mixed_precision` value
- **Inference**: Output directory created before ONNX export

### Removed
- **Training**: Dead `_run_train_epoch()` / `_run_validation()` helpers (~130 lines)

## [1.7.0] - 2026-02-05

### Added
- **HPO**: `--medium` search preset (balanced between `--quick` and full)
- **HPO**: `--inprocess` mode for in-process trial execution with pruning support (faster, but no GPU memory isolation)
- **Training**: `train_single_trial()` function for programmatic HPO integration with pruning callbacks
- **Tests**: 368 new lines in `test_integration.py` covering CLI E2E subprocess tests, HPO objective execution, ONNX denormalization accuracy, and Mamba long-sequence stability
- **Utils**: `setup_hpc_cache_dirs()` exported as public API for HPC environments

### Changed
- **Mamba**: Chunked parallel scan for sequences > 512 tokens (numerical stability), warning for sequences > 2048
- **ViT**: `MultiHeadAttention` now uses `F.scaled_dot_product_attention` (PyTorch 2.0+ fused attention)
- **CNN**: Added proper weight initialization (Kaiming for conv, Xavier for linear)
- **Refactoring**: Consolidated `_setup_cache_dir()` into `wavedl.utils.setup_hpc_cache_dirs()`
- **Refactoring**: Consolidated `LayerNormNd` into `_pretrained_utils.py` (removed duplicate from `convnext.py`)
- **Refactoring**: Added `DropPath` and `freeze_backbone()` utilities to `_pretrained_utils.py`
- **Refactoring**: Extracted `_run_train_epoch()` and `_run_validation()` helpers in `train.py`
- **HPO**: Subprocess mode now uses `NopPruner`; in-process mode uses `MedianPruner`
- **HPO**: Conditional args (`huber_delta`, `momentum`) always set with defaults, not `None`

### Fixed
- **Mamba**: Numerical overflow on long sequences (> 512) via chunked scan
- **ConvNeXt V2**: Renamed misleading class names to match architecture
- **Metrics**: Type hint `any` → `Any` in `load_checkpoint()` return type
- **Training**: Removed redundant `MPLCONFIGDIR` setup (already handled by `setup_hpc_cache_dirs()`)
- **ResNet3D**: Input channel adaptation uses shared `_adapt_input_channels()` utility
- **UniRepLKNet**: Input channel adaptation uses shared utility
- **Template**: Fixed docstring placeholder in `_template.py`

## [1.6.3] - 2026-02-05

### Fixed
- **Data**: Explicit `--input_key` now raises `KeyError` if not found (previously silently fell back to auto-detection, risking wrong data load)
- **DDP**: Non-main ranks now timeout after 1 hour (configurable via `WAVEDL_CACHE_TIMEOUT`) instead of waiting indefinitely for cache files
- **DDP**: Cache wait uses `time.monotonic()` for robustness against system clock changes
- **Inference**: Clear `ImportError` with install instructions when `.safetensors` checkpoint exists but library not installed
- **Training**: Scaler now always copied to checkpoint (previously skipped if destination existed, causing stale scaler on retrain)
- **Documentation**: `CONTRIBUTING.md` setup now includes `[dev]` extras for pre-commit and ruff

### Added
- **Tests**: 6 new tests covering explicit key validation, safetensors error handling, and scaler portability

## [1.6.2] - 2026-01-30

### Added
- **CLI**: Unified `wavedl-train` command that works on both local machines and HPC clusters
  - Auto-detects environment (SLURM, PBS, LSF, SGE, Cobalt)
  - HPC: Uses local caching (CWD), offline WandB
  - Local: Uses standard cache locations (`~/.cache`)
  - Fast `--list_models` flag (no accelerate overhead)
  - `wavedl-hpc` kept as backwards-compatible alias

### Changed
- **CLI**: Renamed `hpc.py` → `launcher.py` (clearer purpose for universal launcher)
- **Documentation**: All README examples now use `wavedl-train` instead of `accelerate launch`

## [1.6.1] - 2026-01-30

### Added
- **Models**: 12 new architectures (57 → 69 total):
  - **UniRepLKNet** (tiny/small/base): Large-kernel ConvNet with 31×31 kernels for long-range wave correlations. Dimension-agnostic (1D/2D/3D). Custom implementation, no pretrained weights.
  - **EfficientViT** (m0-m2, b0-b3, l1-l2): Memory-efficient ViT with cascaded group attention. 9 variants from 2.1M to 60.5M params. ImageNet pretrained via timm. 2D only.

### Changed
- **Refactoring**: Consolidated `SpatialShape` type alias into `base.py` (was duplicated in 8 files)
- **Refactoring**: Consolidated GroupNorm helpers (`_get_num_groups`, `_find_group_count`, `_compute_num_groups`) into single `compute_num_groups()` in `base.py`
- **Refactoring**: Renamed `_timm_utils.py` → `_pretrained_utils.py` (now handles both torchvision and timm models)
- **Refactoring**: Extracted pretrained model channel adaptation into shared utilities:
  - `adapt_first_conv_for_single_channel()`: For torchvision models with known paths
  - `find_and_adapt_input_convs()`: For timm models with dynamic layer discovery

### Fixed
- **MaxViT**: Auto-resize input to compatible size (divisible by 28) for arbitrary input dimensions
- **Mamba/Vim**: Replaced O(L) sequential for-loop with vectorized parallel scan (~100x faster, fixes infinite hang with `--compile`)
- **Dependencies**: Added `onnxscript` (required by `torch.onnx.export` in PyTorch 2.1+)
- **HPC Cache**: Pre-download script now uses exact weight versions, preventing redundant downloads

## [1.6.0] - 2026-01-29

### Added
- **Models**: 19 new architectures (38 → 57 total): ConvNeXt V2, Mamba, Vision Mamba, MaxViT, FastViT, CAFormer, PoolFormer
- **Tests**: Expanded architecture tests with freeze_backbone and single-channel input validation

### Changed
- **CLI**: Renamed `--no-pretrained` to `--no_pretrained` for consistency with other flags

### Fixed
- **ConvNeXt**: Added LayerScale (init=1e-6) and fixed LayerNorm to prevent gradient explosion
- **Data**: `_TransposedH5Dataset` now has `ndim` property (fixes MAT v7.3 memmap crash)
- **Data**: Explicit `--output_key` now raises `KeyError` if not found (no silent fallback)
- **Training**: Mixed precision (`--precision bf16/fp16`) now wraps forward pass in `autocast()`

## [1.5.7] - 2026-01-24

### Added
- **Plotting**: Refactored with helper functions for cleaner code
- **Tests**: 178 new unit tests (725 → 903 total)
- **Training**: `--deterministic` and `--cache_validate` flags

### Changed
- **Plotting**: Publication-quality styling with LaTeX fonts
- **Documentation**: Updated README with new SPIE paper link
- **Pretrained Models**: All use modified conv for 1-channel (3× memory savings vs expand)

### Fixed
- **CLI**: `--pretrained` now uses `BooleanOptionalAction` (was no-op)
- **Constraints**: `x[i,j]` auto-squeezes channel for single-channel data
- **Inference**: Channels-last format now raises error with fix guidance
- **Inference**: `load_checkpoint` uses `pretrained=False` (offline-safe)
- **Inference**: `--input_key`/`--output_key` strict validation (exact match required)
- **TCN**: `GroupNorm` divisibility for custom channel counts
- **CLI**: `--import` error handling for missing/invalid files
- **Pretrained**: `freeze_backbone` now freezes adapted stem conv
- **Pretrained**: Swin `features[0][0]` access guarded for torchvision compatibility
- **CI**: LaTeX rendering optional with `_is_latex_available()` check
- **Examples**: Added missing checkpoint files, fixed notebook cell
- **Tests**: Fixed RUF059 lint warnings in `test_data_cv.py`

## [1.5.6] - 2026-01-15

### Added
- **ViT**: `pad_if_needed` parameter in `PatchEmbed` and `ViTBase` for NDE/QUS applications where edge effects matter (pads input to patch-aligned size instead of dropping edge pixels)
- **Training**: `--no_pretrained` flag to train from scratch without ImageNet weights
- **MATLAB**: `WaveDL_ONNX_Inference.m` script for ONNX model inference in MATLAB with automatic data format handling

### Changed
- **API**: `NPZSource.load_mmap()` now returns `LazyDataHandle` (consistent with `HDF5Source` and `MATSource`)
- **Warnings**: Narrowed warning suppression in `train.py` to preserve legitimate torch/numpy warnings about NaN and dtype issues
- **Data**: Cache validation now uses SHA256 content hash instead of mtime (portable across folders, robust against Dropbox/cloud sync)
- **Examples**: Renamed `elastic_cnn_example/` to `elasticity_prediction/` with MobileNetV3 model (was CNN)

### Fixed
- **API**: Removed special-case handling in train.py and data.py for inconsistent `load_mmap()` return types
- **DDP**: ReduceLROnPlateau patience was divided by GPU count (accelerator wrapper caused multi-process stepping)
- **MATLAB ONNX**: Fixed critical image transpose issue - data must be transposed to convert from MATLAB column-major to Python row-major ordering
- **MATLAB ONNX**: Added network initialization step after `importNetworkFromONNX` (required for networks with unknown input formats)

## [1.5.5] - 2026-01-13

### Fixed
- **Inference**: Single-sample MAT files with multiple targets now correctly load as `(1, T)` instead of `(T, 1)`
- **HPO**: Removed read-only site-packages cwd (prevents permission errors when pip-installed)
- **Data**: Cache invalidation now raises RuntimeError if stale files cannot be removed (prevents silent stale data reuse)
- **Data**: NPZ file descriptors now properly closed after loading (prevents leaks in long-running workflows)
- **Metrics**: `plot_qq` handles zero-variance errors gracefully (no more NaN/division-by-zero)
- **Tests**: Integration test for multi-epoch training no longer flaky (removed random-data loss decrease assertion)

## [1.5.4] - 2026-01-11

### Changed
- **Packaging**: Moved dev tools (pytest, ruff, pre-commit) from core deps to `[project.optional-dependencies]`

### Fixed
- **Critical**: Cache invalidation now deletes stale `.dat`/`.pkl` files (prevents silent reuse of old data)
- **Critical**: ReduceLROnPlateau patience was divided by GPU count (scheduler stepped by all processes)
- **Data**: OOM guard in `load_test_data()` now applies to main HDF5/MAT paths (was only in fallback)
- **Swin**: Backbone bias/norm params now get 0.1× LR decay (matches intended fine-tuning behavior)
- **CLI**: Multiple `--import` files now use unique module names (prevents silent overwrites)

## [1.5.3] - 2026-01-10

### Changed
- **HPC**: TORCH_HOME and WandB caches now always use CWD (compute nodes lack internet access)
- **HPC**: Triton/Inductor caches set unconditionally before torch imports (prevents `--compile` permission errors)
- **Training**: Per-GPU Triton/Inductor cache directories prevent multi-process race warnings with `--compile`
- **Validation**: Replaced manual `torch.distributed.gather` with `accelerator.gather_for_metrics` (eliminates GPU memory spike)
- **Config**: `wavedl_version` metadata now dynamically reads from `__version__` instead of hardcoded `"1.0.0"`

### Fixed
- **Cross-validation**: Auto-detect optimal DataLoader workers when `--workers=-1` (matches `train.py` behavior)
- **Test data loading**: Prioritize `input_test`/`output_test` keys over training keys in `load_test_data()`
- **ResNet**: Added GroupNorm divisibility validation (prevents cryptic runtime errors)
- **Tests**: Force `pretrained=False` in architecture tests for offline CI compatibility
- **Documentation**: Updated README custom model signature and HPC environment variable notes
- **Metadata**: Synced CITATION.cff version

## [1.5.2] - 2026-01-08

### Fixed
- **Critical**: NPZ safe_load failed on data access (error occurs when reading arrays, not file open)

## [1.5.1] - 2026-01-07

### Added
- **MPS Inference**: Apple Silicon GPU support for inference (`test.py` auto-detects MPS)
- `--input_channels` flag for explicit channel override in `load_test_data()` (bypasses heuristics)

### Changed
- **NPZ Security**: Pickle now disabled by default, only enabled as fallback for sparse matrices

### Fixed
- **Input-dependent constraints**: Now properly pass inputs to loss function for `x_mean`, `x[...]` expressions
- **DDP validation memory**: Gather validation data only on rank 0 (prevents OOM on multi-GPU setups)
- **Cross-validation**: OneCycleLR now correctly steps per-batch instead of per-epoch
- **ViT patch embedding**: Added warning for non-divisible input shapes (prevents silent data loss)

## [1.5.0] - 2026-01-06

### Added
- **Physics-Constrained Training**: Enforce physical laws during training via penalty terms
  - `--constraint`: Expression constraints (`"y0 > 0"`, `"y0 - y1*y2"`)
  - `--constraint_file`: Custom Python constraint functions
  - `--constraint_weight`: Penalty weights (default: 0.1)
  - `--constraint_reduction`: Reduction mode (`mse` or `mae`)
- Expression syntax with math functions (`sin`, `cos`, `exp`, `log`, `sqrt`, etc.)
- Comparison operators (`>`, `<`, `>=`, `<=`, `==`)
- Input indexing with literal integers (`x[0]`, `x[0,5]`, `x[0,5,10]`)
- Input aggregates (`x_mean`, `x_sum`, `x_max`, `x_min`, `x_std`)
- Automatic denormalization for constraints in physical space
- 21 new unit tests for constraints (704 → 903 total)

### Removed
- `--output_transform` and `--output_bounds` (hard constraints) — redundant with soft constraints

## [1.4.6] - 2026-01-04

### Added
- **HPO**: Auto-detect GPUs and default `--n_jobs` to GPU count (maximizes resource utilization)
- **HPO**: GPU isolation for parallel trials (each trial runs on a dedicated GPU)

### Changed
- **HPC**: Launcher now passes `--multi_gpu` explicitly to suppress accelerate auto-detection warnings
- **Training**: Checkpoints now use `.bin` format (`safe_serialization=False`) for faster saves
- **Training**: Suppressed verbose accelerate checkpoint logging during saves (cleaner output)
- **HPO**: Default `--n_jobs` changed from `1` to `-1` (auto-detect GPUs)

### Fixed
- **HPC**: WandB offline sync instructions only shown when `--wandb` flag is actually used
- **Inference**: `test.py` now checks for `model.bin` in addition to `model.safetensors` and `pytorch_model.bin`
- **HPO**: Relative data paths now converted to absolute (fixes "file not found" in child processes)

## [1.4.5] - 2026-01-04

### Fixed
- **Critical**: `test.py` failed to load checkpoints from `--compile` models (`_orig_mod.` prefix not stripped)

## [1.4.4] - 2026-01-04

### Changed
- Unified HPC cache directory setup across all entry points (`train.py`, `test.py`, `hpc.py`)
- Simplified cache logic: uses CWD fallback only when home is not writable (cleaner for local development)
- Removed `tempfile` dependency from `train.py` and `hpc.py` (uses CWD-based caching instead)

### Fixed
- `torch.compile` model unwrapping during checkpoint save (handles missing `_orig_mod` gracefully)
- E402 lint errors in `test.py` from intentional HPC environment setup imports
- Unit test for HPC environment setup now properly mocks non-writable home directory

## [1.4.3] - 2026-01-03

### Added
- Smart HPC cache directory setup (`_setup_cache_dir`) - auto-detects writable paths for matplotlib/fontconfig

### Changed
- **DDP**: Switched back to `accelerator.gather()` for broader accelerate version compatibility
- Simplified Triton availability check (imports package instead of internal compiler API)

### Fixed
- E402 lint errors from intentional HPC environment setup imports in `train.py`
- Configured per-file-ignores in `pyproject.toml` to allow early `os`/`tempfile` imports
- Added pydantic warning suppression for accelerate's internal Field() usage

## [1.4.2] - 2026-01-03

### Added
- Input-only loading for HDF5/MAT files in `load_test_data()` (inference without ground truth)
- Cache metadata now includes file size and modification time for stale detection

### Changed
- **DDP**: Validation now uses `gather_object` (memory-efficient, collects only on rank 0)
- **HPO**: Reads `training_history.csv` instead of parsing stdout (reliable metric extraction)
- HPO stdout fallback uses regex pattern matching to avoid false positives

### Fixed
- **Critical**: HPO trials always returned `inf` (stdout parsing never matched trainer output)
- **Critical**: DDP validation gathered full tensors to all ranks, risking OOM on large val sets
- HDF5/MAT `load_test_data()` raised KeyError when outputs missing (now optional)
- MAT input-only fallback lacked sparse matrix handling (now uses `MATSource._load_dataset`)

## [1.4.1] - 2026-01-03

### Added
- `validate_input_shape()` method in `BaseModel` for explicit shape contract enforcement
- `--wandb_watch` flag for opt-in gradient watching (reduces overhead by default)
- `--main_process_ip` and `--main_process_port` args in `wavedl-hpc` for multi-node clusters
- Unknown config key detection with helpful warnings for typos

### Changed
- **Performance**: Enabled TF32 precision by default (~2x speedup on Ampere/Hopper GPUs)
- **Performance**: Enabled cuDNN benchmark for auto-tuned convolutions
- **Performance**: Increased DataLoader worker cap from 8 to 16 per GPU
- Improved config validation with type checking before numeric comparisons
- Made `wandb.watch()` opt-in via `--wandb_watch` flag (was always-on)

### Fixed
- **Critical**: `--machine_rank` was hardcoded to 0 in `wavedl-hpc` (multi-node now works correctly)
- `merge_config_with_args()` fragility when required args are added later
- Silent exception swallowing in cross-validation cleanup
- Documentation clarity for `--precision` vs `--mixed_precision` flags

## [1.4.0] - 2026-01-03

### Added
- 6 new model architectures (38 total variants):
  - **EfficientNetV2** (S/M/L) - modern efficient CNNs with pretrained weights
  - **MobileNetV3** (Small/Large) - mobile-optimized with pretrained weights
  - **RegNet** (Y-400MF to Y-8GF) - regularized networks with pretrained weights
  - **ResNet3D-18, MC3-18** - 3D video/volume models
  - **Swin Transformer** (T/S/B) - shifted window attention with pretrained weights
  - **TCN** (small/base/large) - temporal convolutional networks for 1D signals
- New unit tests: `test_cli.py`, `test_config_metrics.py`, `test_data_cv.py`
- 704 total unit tests (up from 422)

### Changed
- Simplified installation: `pip install wavedl` now includes all dependencies
- Removed optional extras `[all]`, `[hpo]`, `[onnx]` - all included by default
- Triton installs automatically on Linux only (via environment marker)
- Skip slow architecture tests in CI for faster builds
- Synced `wavedl-hpc` with original bash script functionality

### Fixed
- E402 lint errors in `train.py` (moved imports to top)
- Suppressed pydantic deprecation warnings

## [1.3.1] - 2026-01-02

### Fixed
- `wavedl-train --list_models` crash with `UnboundLocalError: cannot access local variable 'sys'`

## [1.3.0] - 2026-01-02

### Added
- `wavedl-hpc` command for HPC distributed training (replaces `run_training.sh`)
- `--import` flag for loading custom model modules without wrapper scripts
- PyPI package: `pip install wavedl`

### Changed
- Removed `run_training.sh` (use `wavedl-hpc` instead)
- Made `triton` dependency Linux-only for cross-platform compatibility
- Simplified custom model documentation to 2-step workflow
- Updated all CLI examples to use `wavedl-*` commands

### Fixed
- Pinned `setuptools<77` for PyPI metadata compatibility

## [1.2.0] - 2026-01-02

### Added
- Console entry points: `wavedl-train`, `wavedl-test`, `wavedl-hpo`
- Single version source in `src/wavedl/__init__.py` with dynamic `pyproject.toml` reading
- `--single_channel` flag for explicit channel handling in data loading
- Optuna hyperparameter optimization support (`hpo.py`)

### Changed
- **BREAKING**: Restructured to `src/wavedl/` namespace package layout
  - Use `python -m wavedl.train` instead of `python train.py`
  - Use `from wavedl.models import CNN` instead of `from models import CNN`
- Migrated CI workflows to use `pyproject.toml` for dependencies
- Improved data loading robustness with lazy handles and format detection
- Optimized training loop and consolidated data utilities
- Updated Ruff linter to v0.14.10 for consistent formatting
- Enhanced contributor guidelines with pre-commit setup
- Pinned Ruff version across pre-commit, CI, and local configs
- Moved development setup instructions to CONTRIBUTING.md

### Fixed
- Worker seeding in DataLoader for diverse random augmentations

## [1.1.0] - 2025-12-28

### Added
- GitHub Actions CI/CD for automated testing and linting
- Google Colab demo notebook for easy experimentation
- Pre-commit hooks for code quality enforcement
- GitHub Discussions link for community support

### Fixed
- LaTeX rendering in diagnostic plots
- Badge spacing and display in README

## [1.0.0] - 2025-12-24

### Added
- Initial release of WaveDL framework
- Core CNN, ResNet, and Transformer model architectures
- Multi-format data loading (NPZ, HDF5, MAT)
- Training and evaluation scripts with WandB integration
- Comprehensive diagnostic plotting (10+ plot types)
- ONNX export functionality
- Mixed-precision training support
- Reproducibility features (seeding, deterministic ops)
- Example configurations and training scripts
- MIT License and citation file

[Unreleased]: https://github.com/ductho-le/WaveDL/compare/v1.9.0...HEAD
[1.9.0]: https://github.com/ductho-le/WaveDL/compare/v1.8.1...v1.9.0
[1.8.1]: https://github.com/ductho-le/WaveDL/compare/v1.8.0...v1.8.1
[1.8.0]: https://github.com/ductho-le/WaveDL/compare/v1.7.1...v1.8.0
[1.7.1]: https://github.com/ductho-le/WaveDL/compare/v1.7.0...v1.7.1
[1.7.0]: https://github.com/ductho-le/WaveDL/compare/v1.6.3...v1.7.0
[1.6.3]: https://github.com/ductho-le/WaveDL/compare/v1.6.2...v1.6.3
[1.6.2]: https://github.com/ductho-le/WaveDL/compare/v1.6.1...v1.6.2
[1.6.1]: https://github.com/ductho-le/WaveDL/compare/v1.6.0...v1.6.1
[1.6.0]: https://github.com/ductho-le/WaveDL/compare/v1.5.7...v1.6.0
[1.5.7]: https://github.com/ductho-le/WaveDL/compare/v1.5.6...v1.5.7
[1.5.6]: https://github.com/ductho-le/WaveDL/compare/v1.5.5...v1.5.6
[1.5.5]: https://github.com/ductho-le/WaveDL/compare/v1.5.4...v1.5.5
[1.5.4]: https://github.com/ductho-le/WaveDL/compare/v1.5.3...v1.5.4
[1.5.3]: https://github.com/ductho-le/WaveDL/compare/v1.5.2...v1.5.3
[1.5.2]: https://github.com/ductho-le/WaveDL/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/ductho-le/WaveDL/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/ductho-le/WaveDL/compare/v1.4.6...v1.5.0
[1.4.6]: https://github.com/ductho-le/WaveDL/compare/v1.4.5...v1.4.6
[1.4.5]: https://github.com/ductho-le/WaveDL/compare/v1.4.4...v1.4.5
[1.4.4]: https://github.com/ductho-le/WaveDL/compare/v1.4.3...v1.4.4
[1.4.3]: https://github.com/ductho-le/WaveDL/compare/v1.4.2...v1.4.3
[1.4.2]: https://github.com/ductho-le/WaveDL/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/ductho-le/WaveDL/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/ductho-le/WaveDL/compare/v1.3.1...v1.4.0
[1.3.1]: https://github.com/ductho-le/WaveDL/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/ductho-le/WaveDL/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/ductho-le/WaveDL/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/ductho-le/WaveDL/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ductho-le/WaveDL/releases/tag/v1.0.0
