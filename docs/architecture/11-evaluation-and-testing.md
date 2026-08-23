# 11 — Evaluation and Testing

## 1. What "correct" means here

A reference implementation has two obligations, and this repository tests both:

1. **Encoder and decoder must agree bit-exactly.** Verified by comparing MD5 hashes of the
   encoder-side and decoder-side reconstructions.
2. **Coding performance must not regress.** Verified by BD-rate style comparison against the
   tools-off anchor across the whole test set.

```mermaid
flowchart TB
    ENC["encoder"] -->|"MD5: (Y, U, V)"| L1["log/enc/{img}_{bpp}.txt"]
    DEC["decoder"] -->|"MD5: (Y, U, V)"| L2["log/dec/{img}_{bpp}.txt"]
    L1 --> CMP["compare_md5s.py"]
    L2 --> CMP
    CMP --> R{"identical?"}
    R -->|yes| OK["✓ conformant"]
    R -->|no| FAIL["✗ mismatch — the run fails"]

    ENC --> BIT["bitstream size → bpp"]
    DEC --> METRICS["PSNR / MS-SSIM"]
    BIT --> SUM["summary.txt"]
    METRICS --> SUM
    SUM --> MERGE["merge_summaries / merge_op_results"]
    MERGE --> XLS["Excel workbook<br/>deltas vs anchor"]
```

## 2. The evaluation harness

`CodecEval` in `src/codec/scripts/eval.py` is the engine; `src/reco/scripts/eval.py` supplies the
reconstruction-task specifics (which encoder, which decoder, which comparator).

```mermaid
sequenceDiagram
    participant Eval as CodecEval
    participant Pool as ProcessPoolExecutor
    participant Enc as encoder subprocess
    participant Dec as decoder subprocess
    participant Cmp as compare_md5s

    Eval->>Eval: create_output_dirs()
    Eval->>Eval: build the resolved config → cfg.json
    Eval->>Eval: enumerate images × target_bpps
    Note over Eval: apply the per-image override and the<br/>per-image-per-bpp override, when either exists

    Eval->>Pool: compute_with_threads(gpu_list, configs)
    loop per (image, rate) job
        Pool->>Enc: encode → bit/, rec/, log/enc/
        Pool->>Dec: decode → rec_dec/, log/dec/
        Pool->>Cmp: compare MD5s → log/compare/
    end

    Eval->>Eval: execute_eval() → summary
    Eval->>Eval: remove_result_files() if --store_bit/--store_rec are 0
```

Each worker pins itself to one GPU via `CUDA_VISIBLE_DEVICES`, derived from the process name in
`set_gpu_id()`. `CoderProcess` (`src/codec/scripts/coder_process.py`) keeps the model loaded
across jobs within a worker — `is_model_loaded()` / `is_first_time()` / `is_args_stored()` control
whether the tree is rebuilt, which is what makes a full-set run tractable.

### Naming conventions

Generated filenames are structured, and the decoder parses them back:

| Kind | Pattern | Example |
| --- | --- | --- |
| Bitstream | `<codec>_<name>_<bpp:03d>.bits` | `JAI_00030_TE_050.bits` |
| Reconstruction | `<codec>_<name>_<W>x<H>_<bits>bit_<fmt>_<bpp:03d>` | `JAI_00030_TE_560x888_8bit_sRGB_050.png` |
| Test image | `<id>_TE_<W>x<H>_<bits>bit_<fmt>.png` | `00030_TE_560x888_8bit_sRGB.png` |

`get_correct_bit_name()` and `get_correct_rec_name()` in `src/codec/__init__.py` build them;
`CodecDecoder.parse_bitstream_name()` recovers the image name and target bpp from a bitstream
filename.

## 3. Metrics

`src/codec/metrics/` (version 1.1.0). Two metrics are instantiated by default:

| Metric | Class | Notes |
| --- | --- | --- |
| PSNR | `PSNRMetric` | Per component; `--jvet-psnr` switches to the JVET convention |
| MS-SSIM | `MSSSIMTorch` | Via `pytorch-msssim` |

`requirements.txt` also pins `psnr_hvsm`, `pyrtools` and `scikit-image`, the dependencies of the
wider JPEG AI metric set; `MetricsFabric.metrics_list` is the switchboard that decides which are
actually computed.

Supporting machinery:

| Piece | Purpose |
| --- | --- |
| `DataClass` | Loads PNG or raw YUV, normalises range and bit depth, converts colour |
| `color_conv_matrix('709')`, `yuv_to_rgb`, `rgb_to_yuv` | Colour conversion for metric computation |
| `extract_info(fn)` | Parse `WxH_Nbit_FMT` from a filename |
| `read_yuv` / `write_yuv` | Raw YUV IO with subsampling handling |
| `compute_bpp(filename, shape)` | Bitstream bytes → bits per pixel |
| `store_complexity_info()` | Record kMAC/px and encode/decode times |

A summary row carries: `Reconstruct`, `Original`, `Codec`, `BPP`, the metric columns,
`MAC/pxl`, `DecGPU`, `DecCPU`, `EncGPU`, `EncCPU`.

Complexity comes from `ptflops`: `CodecCoder.setup_ptflops_custom_hooks()` collects
`ptflops_custom_hook()` from every module that defines one — necessary because the custom
integer convolutions are invisible to the default counter.

## 4. Unit tests

```bash
make unittest       # CUDA_VISIBLE_DEVICES="-1" python -m unittest -v
```

Tests live next to what they test:

| File | Covers |
| --- | --- |
| `src/codec/entropy_coding/lib_wrappers/mans/test.py` | me-tANS round-trip: encode then decode must return the input |
| `src/codec/entropy_coding/lib_wrappers/direct/test.py` | Direct bit IO |
| `src/codec/components/contexts/tests.py` | Context model / MCM phases |
| `src/codec/coding_tools/quality_map/test.py` | Quality map generation |
| `src/codec/coding_tools/rdi/test.py` | RDI header round-trip |
| `src/codec/coding_tools/udi/test.py` | UDI substream round-trip |

They run CPU-only by design so they stay deterministic. On failure, CI collects
`sgm_err_arr.json` and `factorized_err_arr.json` — the tensors that made the entropy coder
disagree — and attaches them to the merge request.

## 5. Continuous integration

`.gitlab-ci.yml`, two stages, both `only: merge_requests` and `allow_failure: false`.

```mermaid
flowchart TB
    MR["Merge request"] --> S1["Stage: unit-test"]
    S1 --> U1["conda activate jpeg_ai_vm<br/>pip install -r requirements.txt<br/>dvc remote configure"]
    U1 --> U2["./scripts/build_test_libs.sh"]
    U2 --> U3["python -m src.reco.scripts.eval --only_cpu<br/><i>generates the files the tests need</i>"]
    U3 --> U4["python -m unittest -v"]
    U4 -->|fail| U5["post_comment_gitlab.py<br/>attach ANS error tensors + training logs"]

    MR --> S2["Stage: test performance"]
    S2 --> P1["test-img30-perf<br/>run_reco_all_cfgs.sh on image 30"]
    S2 --> P2["test-img30-toolena<br/>run_tool_perf.py cfg/tool_ena, BOP"]
    P1 --> P3["merge_op_results.py → .xlsm"]
    P2 --> P4["merge_op_results.py --anchor bop:anchor → .xlsm"]
    P3 --> P5["post_comment_gitlab.py<br/>attach workbook + logs"]
    P4 --> P5
```

Both performance jobs run with `CUDA_VISIBLE_DEVICES="-1,-2,-3"` (CPU) on a single image, image
30 (`00030_TE_560x888_8bit_sRGB.png`, 560×888 — the smallest useful test image). The reviewer
gets an Excel workbook attached to the merge request showing the coding-performance delta of the
change. That workbook, not a pass/fail flag, is the actual review artefact.

## 6. Code style

`.pre-commit-config.yaml`, installed by `make setup_env`:

| Hook | Configuration |
| --- | --- |
| `flake8` | `--ignore=W503,W504,E501` |
| `isort` | Default |
| `yapf` | `--style={based_on_style: pep8, column_limit: 99}` |
| `trailing-whitespace`, `end-of-file-fixer`, `mixed-line-ending --fix=lf` | Whitespace hygiene |
| `check-yaml`, `check-merge-conflict` | Sanity |
| `double-quote-string-fixer` | Prefer single quotes |
| `fix-encoding-pragma --remove` | Strip `# -*- coding: utf-8 -*-` |
| `requirements-txt-fixer` | Keep `requirements.txt` sorted |
| `dvc-pre-commit`, `dvc-pre-push`, `dvc-post-checkout` | Keep DVC-tracked files in sync with the working tree |

All hooks exclude `3rdparty|thirdparty`.

## 7. Data and model management with DVC

```mermaid
flowchart LR
    subgraph REPO["Git repository"]
        P1["models/VM_bop/decoder_Y_0.002.pth.dvc"]
        P2["data/test/00030_….png.dvc"]
    end
    subgraph REMOTE["DVC remote 'storage'"]
        C["ssh://storage_reader@jpeg-git.lx.it.pt/cache"]
    end
    subgraph LOCAL["Working tree"]
        F1["models/VM_bop/decoder_Y_0.002.pth"]
        F2["data/test/00030_….png"]
    end

    P1 -.dvc pull.-> F1
    P2 -.dvc pull.-> F2
    C --> F1
    C --> F2
    F1 --> DL["Downloader<br/>dvc fetch + checkout at load time"]
```

Git holds only `.dvc` pointer files; the binaries come from the remote configured in
`.dvc/config`. The `Downloader` in `src/codec/utils/downloader.py` runs `dvc fetch` and
`dvc checkout -f`
on a model's pointers when it is first needed, so DVC re-verifies content hashes and a partial or
stale pull is repaired rather than silently producing wrong output.
`critical_for_file_absence` (inverted by `--skip_loading_error`) decides whether a missing file is
fatal.

For teams behind a slow link, `scripts/sFTP_mirror/` sets up a local mirror of the cache with a
3-hourly cron refresh and generates a `.dvc/config.local` pointing at it.

`dvc.yaml` declares two pipeline stages, so `dvc repro` can drive the whole evaluation:

```yaml
stages:
  VM: python -m src.reco.scripts.eval -coding_type enc_dec -out_dir ./results/VM
  CV: python -m src.cv.scripts.eval data/imagenet_val ./results/CV -coding_type enc_dec
```

Note the `CV` stage references `src/cv`, a computer-vision task package that is **not present in
this release** — like `src/train`, it ships separately.

## 8. Reproducibility

Bit-exactness between encoder and decoder is not automatic on GPU hardware. The codebase enforces
it explicitly:

| Mechanism | Where | Effect |
| --- | --- | --- |
| `@determinism`, `determinism_on_eval` | `common/pytorch_ops.py`, applied to `compress`, `decompress`, `forward` | Force deterministic kernels around the sensitive paths |
| `disable_torch_random()` | `decode_z`, `decode_y` | Remove any RNG dependence |
| `disable_tf32()` | same | Disable TF32 on Ampere+, which would otherwise change results |
| `torch.set_num_threads(1)` | encoder and decoder when `target_device == 'cpu'` | Remove thread-count dependence |
| Integer convolutions | `Conv2di` in the hyper-scale-decoder | Make the σ index exact by construction |
| Feature clipping | `feature_clipping` with per-layer `clip_thres` | Bound activation ranges for fixed-point equivalence |
| Control-point hashes | `TensorOps.get_hash()` logged at named points | Localise a divergence to a specific tensor |

Those control points are worth knowing about when debugging a mismatch. The logs carry hashes at:
`z_hat`, `residual`, `residual_dq_tiles` (per tile), `quantized_residual`, `scale_log_masked_tile`,
`mask_tile`, and `y_hat`/`psi`. Comparing encoder and decoder logs finds the first divergent hash,
which names the stage that broke — far quicker than bisecting a pixel difference.

## 9. Debugging playbook

| Symptom | First things to check |
| --- | --- |
| Encoder/decoder MD5 mismatch | Diff the control-point hashes in `log/enc/` vs `log/dec/`; the first difference names the stage |
| A substream differs | Run with `--cfg cfg/AE/verbose.json` to print per-substream sizes and MD5s |
| Header field looks wrong | `python scripts/bitstream_probe.py stream.bits` prints every named field |
| Entropy coder desync | Check the σ path: is `VM_common_int` being used? Did the me-tANS cache go stale (`--rebuild_ae_cache 1`)? |
| Out of memory | Reduce `numSamplesPerTile` in the tile managers, or enable region partitioning |
| Non-deterministic results | Confirm the `determinism` decorators are on the path you changed, and that TF32 is off |
| Model fails to load | Run `dvc pull` manually and read its output — the `Downloader` only reports "No *.dvc files" / "No information about model" and exits |
| Slow encode | The bitrate matcher in search mode re-runs compression per candidate beta — use `cfg/BRM/use_list.json` or `default.json` |
