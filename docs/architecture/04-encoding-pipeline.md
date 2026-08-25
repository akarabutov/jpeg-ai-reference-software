# 04 — Encoding Pipeline

## 1. Command

```bash
python -m src.reco.coders.encoder <IMAGE.png> <OUTPUT.bits> \
       [-r <REC.png>] [--set_target_bpp <BPP×100>] \
       --cfg cfg/tools_on.json cfg/profiles/base.json
```

## 2. Two phases, one engine

The encoder runs in two clearly separated phases, and it is worth understanding why:

```mermaid
flowchart LR
    A["Phase 1<br/><b>ce.compress(image)</b>"] --> B["Decisions<br/><i>every tensor and every<br/>coding decision</i>"]
    B --> C["Phase 2<br/><b>ce.encode(ec_module, decisions)</b>"]
    C --> D["Bitstream"]
    A --> E["rec_image<br/><i>encoder-side reconstruction</i>"]
```

Phase 1 does all the *thinking* — networks, rate control, RDO, filter decisions — and produces
both a `Decisions` object and the encoder's own reconstruction. Phase 2 does only *writing*: it
serialises `Decisions` into substreams.

The split exists because the bitstream cannot be opened until the number of residual substreams
is known, and that depends on region/tile decisions made during compression. `RecoEncoder.encode_stream`
therefore calls `self.ce.compress(raw_image)` first and only then `self.create_bs(...)`.

## 3. Top-level flow

```mermaid
sequenceDiagram
    autonumber
    participant CLI as encoder.py
    participant Coder as RecoEncoder
    participant CE as CodingEngine
    participant BS as BitstreamStructure
    participant FS as filesystem

    CLI->>Coder: process_encoder()
    Coder->>Coder: init_common_codec()<br/>build the tool tree from --cfg
    Coder->>Coder: load_models(downloader)<br/>resolve paths and load .pth files
    Coder->>Coder: set_target_bpp_idx(bpp_idx)

    Coder->>CE: compress(raw_image)
    CE-->>Coder: rec_image, decisions

    Coder->>BS: create_bs(bin_path)
    Coder->>Coder: init_ec_module()
    Coder->>CE: encode(ec_module, decisions)
    Coder->>BS: close_bs() → fill_substreams + write_substreams
    BS->>FS: OUTPUT.bits

    opt -r given
        Coder->>FS: write reconstruction PNG
    end
    opt --calc_metrics
        Coder->>Coder: compute PSNR / MS-SSIM / … against the original
    end
    Coder->>Coder: print MD5 of each reconstructed component
    Coder->>FS: save profiler results
```

The MD5 print is not cosmetic: `src/reco/scripts/compare_md5s.py` compares the encoder's and
decoder's reconstruction hashes to prove encoder/decoder match. This is the primary correctness
check in the evaluation harness and in CI.

## 4. Phase 1 in detail — `CodingEngine.compress()`

```mermaid
flowchart TB
    IMG["Image (sRGB or YUV)"] --> INIT["init_new_img_recursivly()<br/>reset per-image state in every tool"]
    INIT --> SUB["read s_ver / s_hor from the image<br/>decide c_ver / c_hor (coded subsampling)"]
    SUB --> RC["res_changer.forward_transform()<br/><i>optional intra-resolution downscale</i>"]
    RC --> CP["colour_processing.pre_processing()<br/>colour transform, chroma histogram"]
    CP --> CM["model.compress(img2compress)<br/><b>the neural codec</b>"]
    CM --> DEC["Decisions"]
    CM --> DECOMP["model.decompress(decisions)<br/>reconstruct exactly as the decoder will"]
    DECOMP --> RCB["res_changer.backward_transform()<br/>upscale back"]
    RCB --> PFC["post_filters.compress(rec, original)<br/><i>choose filter parameters using the original</i>"]
    PFC --> PFD["post_filters.decompress(rec)<br/>apply the chosen filters"]
    PFD --> CPP["colour_processing.post_processing()"]
    CPP --> REC["rec_image"]
```

Two things to notice:

- The encoder **runs the decoder**. `model.decompress()` is called inside `compress()` so that
  post-filters see the true reconstruction and so the encoder's output matches the decoder's
  bit-exactly.
- Post-filters have a two-call protocol: `compress(rec, original)` *decides* (comparing against
  the original, and writing its decisions into the tool's header state), `decompress(rec)`
  *applies*. The decoder only ever calls `decompress`.

## 5. Inside the core model

`CcsGvaeMultiTools` selects the active beta model, then `CcsGvaeSGMM.compress()` splits the
picture into the two component branches.

```mermaid
flowchart TB
    IMG["Image → YUV, range-converted"] --> LUMA["luma = component a<br/>(1, 1, H, W)"]
    IMG --> CHROMA["chroma = U, V<br/>at coded subsampling c_ver/c_hor"]

    LUMA --> SUPP["pixel_unshuffle(luma, 2)<br/>4 channels of luma support info"]

    LUMA --> MY["model_y — SepChannelsSGMMTool<br/>chs_input = 1, chs_ls = 160"]
    SUPP --> CAT["concat"]
    CHROMA --> PACK["pack to 8 channels<br/>(420: U×4, V×4)"]
    PACK --> CAT
    CAT --> MUV["model_uv — SepChannelsSGMMTool<br/>chs_input = 2, chs_ls = 96<br/>downsample_factor = 2"]

    MY --> DY["Decisions for model_y"]
    MUV --> DUV["Decisions for model_uv"]
    DY -. "qual_map is computed on luma<br/>and reused for chroma" .-> MUV
```

This luma-support-info concatenation is the cross-component conditioning that the name
*CCS* (cross-component / conditional) refers to: chroma is coded conditioned on a downsampled
view of luma, and luma is always coded first.

Packing depends on the coded chroma format (`ccs_sgmm_tool.py:compress`):

| Coded format | Chroma packing into 8 channels |
| --- | --- |
| 4:4:4 (`c_ver=1, c_hor=1`) | `pixel_unshuffle(cat(U, V), 2)` |
| 4:2:0 (`c_ver=2, c_hor=2`) | `U.repeat(4)`, `V.repeat(4)` |
| 4:2:2 (`c_ver=1, c_hor=2`) | Even/odd rows of U and V, each repeated ×2 |

## 6. Inside one component branch

This is where the actual compression happens. `SepChannelsSGMMTool.compress()` →
`CommonEncDecModules.compress()`.

```mermaid
flowchart TB
    subgraph A["Step 1 — analysis, per colocated tile"]
        IMGT["image tile"] --> ANA["analysis transform<br/>bop_prim / hop_prim<br/>↓16 spatially"]
        ANA --> Y["y — latent<br/>(1, 160, H/16, W/16)"]
        Y --> HE["hyper_encoder<br/>↓4 more"]
        HE --> Z["z"]
        Z --> CLAMP["clamp to −z_offset … z_range−z_offset−1<br/>quantise → int8"]
        CLAMP --> ZH["z_hat<br/>(1, C, H/64, W/64)"]
    end

    subgraph B["Step 2 — scales, whole picture"]
        ZH --> HSD["hyper_scale_decoder"]
        HSD --> SL0["scale_log_origin"]
        SL0 --> Q1["quantizer.analyze + quantize_scale<br/>(everything except RVS)"]
        Q1 --> SKSL["skip_scale_log"]
        SKSL --> Q2["quantizer.analyze + quantize_scale<br/>(RVS stage)"]
        Q2 --> SL["scale_log — final"]
    end

    subgraph C["Step 3 — hyper decode, per tile"]
        ZH --> HD["hyper_decoder"]
        HD --> PSI["psi tiles"]
        PSI --> MERGE["merge overlaps → psi<br/>(1, 4C, 2·H/64, 2·W/64)"]
    end

    subgraph D["Step 4 — context + residual, per tile"]
        MERGE --> CTX["Context (MCM)<br/>4-phase autoregressive prediction"]
        Y --> CTX
        SL --> CTX
        CTX --> RES["residual = y − prediction"]
        RES --> QR["quantizer.quantize_resi<br/>gain unit, RVS, quality map"]
        QR --> RQ["residual_quant — int16<br/><b>the symbols that get coded</b>"]
        RQ --> DQ["dequantise → residual"]
        DQ --> YH["y_hat = prediction + residual"]
        YH --> SKIP["skip mode: cube flags<br/>16×16 latent cubes that code as zero"]
    end

    D --> OUT["Decisions"]
```

### Why the scale is computed twice

`encoder_get_scales()` runs the quantiser analysis in two passes with `excl_list=['rvs']` and
then `incl_list=['rvs']`. Residual variance scaling needs to know the scale *after* the other
quantiser stages have shaped it, so it is deliberately excluded from the first pass and applied
to its result. `skip_scale_log` is the intermediate; skip-mode decisions are made against it,
not against the RVS-scaled version.

### Tiling

Three tile managers with different granularity operate over the same picture:

| Manager | Used for | Configured by |
| --- | --- | --- |
| `tile_manager_enc` | Analysis + hyper-encoder | `model_{y,uv}.tile_manager_enc` |
| `tile_manager_hd` / `tile_manager_hyper` | Hyper-decoder and region layout | `hyper_decoder_overlap_in_latent_samples` |
| `tile_manager_mcm` | Context model and residual coding | `mcm_overlap_in_latent_samples` |
| `tile_manager_synthesis` | Synthesis transform | `model_{y,uv}.tile_manager_synthesis` |

Tiles overlap. Each manager can compute the "core" of an overlapping tile
(`get_core_of_overlapping_latent_tile`), and only the core is written into the full-picture
tensor — the overlap exists purely to give convolutions valid context at tile borders.
`tiling.ColocatedTiles.iter_colocated_grids()` walks image, `y` and `z` tile grids in lockstep so
the three resolutions stay aligned.

## 7. Rate control and RDO

Two special tools hook into the composite around the core model:

```mermaid
flowchart LR
    ORI["original image"] --> BRM["BitrateMatcher<br/><i>pre-processing RDO</i>"]
    BRM -->|"sets active_tool_idx<br/>and beta_displacement_log"| CORE["core model compress"]
    CORE --> RDLR["RDLR<br/><i>post-processing RDO</i>"]
    RDLR -->|"refined residual_quant"| OUT["Decisions"]
```

### BitrateMatcher (`coding_tools/bitrate_matcher/`)

Registered via `model.add_preproc_tool('bitrate_matcher', …)`. Its job is to choose the quality
operating point that lands closest to the requested bpp. Three modes, selected by `cfg/BRM/`:

| Mode | Behaviour |
| --- | --- |
| `default.json` | Look up `default_models` and `default_beta_disp_log` by rate-point index. No search — fast, deterministic |
| `use_list.json` | Read the per-image, per-rate beta from `cfg/betas/<op>/betas_tools_{on,off}.txt` |
| `regen_list.json` | Actually search: `match_luma()` runs the codec at trial betas and `beta_linear_interpolation()` interpolates towards the target bits. The chroma beta is then copied from luma when `independent_beta_UV` is `1`, or searched with `find_UV_beta_with_hyperopt()` when it is `0`. Distortion is a weighted MS-SSIM (`_calculate_mssim_distortion`) |

The search is what `--set_target_bpp` triggers, and it is the expensive path — it re-runs
compression several times per image. Regular test runs use the pre-computed lists instead.

### RDLR (`coding_tools/rdlr/`)

Registered via `model.add_postproc_tool('rdlr', …)`. Rate-Distortion Latent Refinement iterates
over the already-quantised latent and flips individual values when doing so lowers
`D + λ·R`. It works per tile (`numSamplesPerLumaTile`, `numSamplesPerChromaTile`) and per
iteration count (`numIteYLuma: 8` by default; z-refinement is available but disabled). Slope
estimation lives in `_get_slopes_parameters()`, with `_replace_not_sane_slopes_with_mean()`
guarding against degenerate tiles. RDLR is encoder-only — it changes what is coded, never how
it is parsed.

## 8. Phase 2 — writing the bitstream

```mermaid
sequenceDiagram
    participant CE as CodingEngine
    participant HC as HeaderCoder
    participant EC as ECModule
    participant BS as BitstreamStructure

    CE->>HC: encode_header_recursively()
    Note over HC: walks the WHOLE tool tree.<br/>Each CoderEngine writes its<br/>enable flag then its own fields.
    HC->>BS: bits land in the PIH / TON substreams

    CE->>EC: model.encode(ec, decisions)
    loop per component (Y then UV), per region
        EC->>BS: z_hat → SOZ substream
        EC->>BS: residual_quant → SORP (luma) / SORS (chroma)
    end
    opt quality map enabled
        EC->>BS: qp_map → SOQ substream
    end

    CE->>BS: fill_substreams()
    Note over BS: finalise every AE, collect region<br/>memories, prepend region sizes
    CE->>BS: write_substreams()
    Note over BS: SOC · [PIH · TON · SOQ · SOZ · SORP · SORS · UDI · RDI] · EOC
```

Header writing is recursive and order-defined by the tree, which is exactly why the decoder can
reconstruct the same values: it walks the identical tree in the identical order. Adding a header
field to a tool therefore automatically changes both sides — but it also breaks compatibility
with existing bitstreams, so header changes are the highest-risk edits in this codebase.

## 9. Output files

For `python -m src.reco.coders.encoder in.png out.bits -r rec.png --calc_metrics`:

| File | Content |
| --- | --- |
| `out.bits` | The bitstream |
| `rec.png` | The encoder-side reconstruction (bit depth from `--output_bit_depth` or the source) |
| stdout `MD5: (…, …, …)` | Per-component hashes of the reconstruction |
| stdout `=== Metrics ===` | bpp plus the enabled quality metrics |
| `<out>/log/enc/*` | Profiler results, if collectors are enabled |
