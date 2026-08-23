# 05 — Decoding Pipeline

## 1. Command

```bash
python -m src.reco.coders.decoder <INPUT.bits> <OUTPUT.png> [--device gpu|cpu]
```

The decoder takes **no `--cfg`**. `def_base_parser(..., has_cfg=False)` removes the option
entirely, and `CodecCoder.update_kwargs_params()` forces the configuration to
`get_pipeline_desc_paths()` — i.e. `cfg/pipeline.json`. Everything else comes from the
bitstream.

## 2. Bootstrapping problem and how it is solved

The decoder faces a chicken-and-egg problem: it must build the tool tree before it can parse the
headers, but the headers determine how the tree is configured (which beta model, which synthesis
transform, how many regions and threads).

`CodecCoder.open_bs()` resolves it in a specific order:

```mermaid
sequenceDiagram
    participant D as decoder
    participant BS as BitstreamStructure
    participant CE as CodingEngine

    D->>CE: build tree from cfg/pipeline.json<br/>(all four beta models, all transforms)
    D->>BS: read_substreams(file)
    Note over BS: split the file at markers,<br/>nothing is parsed yet
    D->>BS: parse_substreams(only_non_ae=True)
    Note over BS: parse ONLY the non-entropy-coded<br/>substreams: PIH, TON, UDI, RDI
    D->>D: init_ec_module()
    D->>CE: init_new_img_recursivly()
    D->>CE: decode_header_recursively(header_codec)
    Note over CE: now the tree knows the picture size,<br/>active model, regions, thread counts
    CE->>BS: set_ec_params() pushes region/thread<br/>counts back into BitstreamStructure
    D->>BS: parse_substreams(only_non_ae=False)
    Note over BS: NOW parse SOZ / SORP / SORS / SOQ,<br/>splitting them by region using the<br/>counts just decoded
```

The header substreams are deliberately *not* entropy coded (`use_ae = False` in
`SubstreamLayouts.FullmarkersDict`) precisely so that this first pass is possible without any
prior knowledge.

## 3. Top-level flow

```mermaid
sequenceDiagram
    autonumber
    participant CLI as decoder.py
    participant Coder as RecoDecoder
    participant CE as CodingEngine
    participant FS as filesystem

    CLI->>Coder: process_decoder()
    Coder->>Coder: init_common_codec(cfg = pipeline.json)
    opt --device gpu
        Coder->>Coder: init_cuda()
    end
    Coder->>Coder: setup_ptflops_custom_hooks()
    Coder->>Coder: load_models_recursively(downloader)

    Coder->>Coder: open_bs(bit_fpath)  ← headers decoded here
    Coder->>CE: decode(ec_module, with_headers=False)
    CE-->>Coder: decisions (z_hat, residual per component)
    Coder->>CE: check_complience()
    Coder->>CE: decompress(decisions)
    CE-->>Coder: rec_image
    Coder->>Coder: close_bs()

    Coder->>Coder: print MD5 per component
    Coder->>FS: write OUTPUT.png
    opt --calc_metrics with --ori_file
        Coder->>Coder: compute metrics
    end
    opt --calc_ptflops
        Coder->>Coder: report kMAC/px
    end
    Coder->>FS: save profiler results
```

`check_complience()` is the conformance gate. It verifies, against `cfg/profiles/`:

1. `decoder_profile_id` names a profile that exists;
2. the default synthesis transform is one the profile supports;
3. `level_idc` decomposes into a known `level_idc0` / `level_idc1` pair;
4. `img_height × img_width ≤ max_pic_size` for that level;
5. the active beta model index is permitted at that level.

Any failure is an assertion — a non-conforming stream stops the decoder rather than producing a
picture.

## 4. Entropy decoding — `CodingEngine.decode()`

```mermaid
flowchart TB
    START["model.decode(ec)"] --> TM["setup_dec_tile_managers_of_model()<br/>for model_y and model_uv"]
    TM --> QM["qual_map.decode(ec)<br/>from the SOQ substream, if present"]
    QM --> LOOP{"for each component<br/>model_y then model_uv"}

    LOOP --> DZ["decode_z()<br/>_ac_decode_z from the SOZ substream<br/>factorized/hyper entropy model"]
    DZ --> ZH["z_hat"]
    ZH --> HSD["hyper_scale_decoder(z_hat)"]
    HSD --> QS["quantizer.quantize_scale (excl. RVS)"]
    QS --> SCALE["scale_log"]
    SCALE --> MASK["skip_mode.mask()<br/>which latent positions carry symbols"]
    MASK --> DY["decode_y()<br/>_ac_decode_y per tile / region<br/>from SORP (luma) or SORS (chroma)"]
    DY --> RQ["residual_quant"]
    RQ --> DQ["quantizer.dequantize_resi<br/>gain unit, RVS, quality map"]
    DQ --> RES["residual"]
    RES --> OUT["Decisions, per component"]
```

Note what is **not** here: the hyper-decoder, the context model and the synthesis transform do
not run during `decode()`. Entropy decoding only needs `scale_log` (from the hyper-scale-decoder)
to drive the probability model. Prediction and reconstruction happen in the next phase. This
separation is what allows the residual substreams to be parsed independently of, and in parallel
with, the reconstruction work.

Order matters and is fixed: quality map, then luma, then chroma.

## 5. Reconstruction — `CodingEngine.decompress()`

```mermaid
flowchart TB
    D["Decisions from decode()"] --> HD["hyper_decoder(z_hat) per tile<br/>→ psi tiles"]
    HD --> MERGE["merge_psi_overlaps_of_tiles<br/>+ extract_psi_for_mcm"]
    MERGE --> AR["decompress_ar_scale_tile per MCM tile<br/><b>Context.decompress(residual, psi)</b>"]
    AR --> YHT["y_hat tiles"]
    YHT --> MY["merge_y_hat_overlaps_of_tiles"]
    MY --> LSP["ls_processing.post_processing()<br/><i>LSBS runs here</i>"]
    LSP --> EXT["extract_y_hat_for_synthesis_tiles"]

    EXT --> SY["synthesis transform, luma<br/>per tile → rec_Y"]
    EXT --> SUV["synthesis transform, chroma<br/>per tile, conditioned on luma y_hat → rec_UV"]

    SY --> CROP["crop to display size<br/>(minus diff_display_img_*)"]
    SUV --> CROP
    CROP --> IMGO["Image (YUV, coded subsampling)"]
    IMGO --> FMT["to_format_(source subsampling)"]
    FMT --> RCB["res_changer.backward_transform()"]
    RCB --> PF["post_filters.decompress()<br/>EFElinear → eICCI → EFEnonlinear → LEF"]
    PF --> CPP["colour_processing.post_processing()"]
    CPP --> REC["reconstructed Image"]
```

The chroma synthesis takes `supp_info_uv = decisions['model_y']['tiles_synthesis'][tile]['y_hat']` —
the luma latent for the colocated tile — mirroring the cross-component conditioning done on the
encoder side. **Luma must be fully reconstructed before chroma synthesis can start.**

## 6. The context model (MCM) in decode direction

`src/codec/components/contexts/context.py` implements the Multi-stage Context Model. It is the
autoregressive part of the entropy model and the reason latent decoding is sequential.

```mermaid
flowchart TB
    PSI["psi from hyper_decoder"] --> DSC["down_shuffle_conv_hyper(psi)<br/>→ 4 per-stage conditioning tensors"]
    LAT["residual (dequantised)"] --> DS["down_shuffle(residual)<br/>→ 4 spatial phases"]

    DSC --> S0
    DS --> S0

    subgraph STAGES["4 sequential MCM stages"]
        S0["MCM_phase0<br/>mu = f(psi_0)<br/>no spatial context"]
        S1["MCM_phase1<br/>mu = f(psi_1, y_hat_0)"]
        S2["MCM_phase2<br/>mu = f(psi_2, y_hat_0..1)"]
        S3["MCM_phase3<br/>mu = f(psi_3, y_hat_0..2)"]
        S0 --> S1 --> S2 --> S3
    end

    S3 --> UP["up_shuffle(y_hat_0..3)<br/>→ y_hat at full latent resolution"]
```

Each stage adds its dequantised residual to its predicted mean (`y_hat = resi_dq + mu`) and the
concatenation of all reconstructed stages so far becomes the spatial context for the next. The
four phases are the four positions of a 2×2 spatial pixel-unshuffle, so the model is a
four-pass checkerboard predictor rather than a per-pixel raster scan — this is what makes it
parallelisable within a stage while remaining causal across stages.

`MCM_phase0` is a special case: it has no previous reconstruction and predicts from the
hyperprior alone. `ChannelNet` and `FusionPredNet`
(`components/contexts/channel_net.py`, `fusion_pred_net.py`) are the sub-networks the phases
are built from.

When `use_context_module` is 0 (chroma, by default configuration) the context model is bypassed
and the mean comes straight from an upsampled hyper-decoder output.

## 7. Threading and regions in decode

If `region_partitioning_flag` is set, the picture is split into `numHorRegions × numVerRegions`
regions and each region's residual is decoded independently:

| `region_residual_in_its_own_substream_flag` | Layout | Property |
| --- | --- | --- |
| `1` (Independent Regions) | Each region gets its own `SORP`/`SORS` substream, prefixed with a 1-byte region index | Regions are fully independent — random access, and a region can be dropped |
| `0` (Dependent Regions) | All regions share one substream, prefixed with Exp-Golomb-coded sizes of all but the last | Regions overlap (`hyper_decoder_overlap_in_latent_samples`, `mcm_overlap_in_latent_samples`) which costs less rate but forbids independent extraction |

Independently of regions, an AE substream can be split into `num_threads` byte ranges. Thread
sizes are signalled as signed Exp-Golomb deltas from the mean thread size
(`AEMemObject.__encode_threads_deltas`), which is compact because threads are near-equal in size
by construction. `cfg/tools/ECThread8.json` sets 8 threads for `z`, residual and quality-map
substreams.

## 8. Progressive / partial decoding

The code contains the scaffolding for decoding only part of a stream:

- `num_decode_chs` (`CcsSharedModelsParams`) truncates the latent to the first N channels; the
  context model zeroes the rest (`diff[:, self.num_decode_chs:] = 0`).
- `scripts/bitstream_extractor.py` removes chosen residual substreams or the tools header from
  an existing bitstream, producing a valid but lower-quality stream.
- `scripts/progressive_decoding/` holds standalone encoder/decoder variants and channel-ordering
  utilities for this experiment.
- `CcsGvaeSGMM.forward()` carries commented-out `present_substreams_list_Y/UV` filtering — the
  hooks for skipping absent regions are present but disabled in this release.

## 9. Output

| File | Content |
| --- | --- |
| `OUTPUT.png` | The reconstruction. `--output_bit_depth` forces 8 or 10 bit; otherwise the bit depth signalled in the picture header is used |
| stdout `MD5: (…)` | Per-component hashes — must match the encoder's for a conforming run |
| `<out>/log/dec/*` | Profiler results |
