# PICO: PyTorch-реализация обучения и inference

Реализация кодека **PICO (Perceptual Image Codec)** из статьи
*"What Matters in Practical Learned Image Compression"*
(Tatwawadi et al., Apple, [arXiv:2605.05148](https://arxiv.org/abs/2605.05148)).

В репозитории [apple-aiml-research/ml-pico](https://github.com/apple-aiml-research/ml-pico)
нет кода модели — только страница проекта и реконструкции. Поэтому этот код написан
по тексту статьи и её рисункам (Fig. 3, 4, 11, Table 3, приложения B, C, F, H).
Всё, чего в статье нет, выбрано нами и перечислено ниже.

## Структура

```
pico/
  picocodec/
    layers.py         ConvScale, LearnedScale, Haar/iHaar-ресемплинг, ConvScale311, CS-Chain,
                      кондиционирование по уровню качества, репараметризация для inference
    model.py          Outer encoder/decoder, scale decoder, context decoder,
                      one-shot context model (2x2 grid), PICO (forward / compress / decompress)
    losses.py         rate + MSE + LPIPS + MS-SSIM + TilingArtifactLoss + TextFidelityLoss + GAN
    discriminator.py  PatchGAN-дискриминатор увеличенной ёмкости
    data.py           датасет случайных кропов (+ маски текста)
    codec.py          тайлинг 504/512 и формат битстрима
  train.py            обучение (фаза 1: MSE, фаза 2: perceptual + GAN)
  pico_codec.py       CLI: encode / decode
  evaluate.py         оценка на папке с изображениями по реальным битстримам
  scripts/make_text_masks.py   маски текста детектором CRAFT (через EasyOCR)
  configs/pico_mse.yaml, configs/pico_perceptual.yaml
  tests/test_pico.py
```

## Установка

```bash
cd pico
pip install -r requirements.txt
```

## Обучение

```bash
# Фаза 1: только MSE (lr 8e-4, ×0.3 на 70%, ×0.1 на 90% шагов)
python train.py -c configs/pico_mse.yaml --out runs/mse \
    --set "data.train=[/data/div2k,/data/flickr2k,/data/clic]" data.val=/data/clic_valid

# (опционально) маски текста для TextFidelityLoss
python scripts/make_text_masks.py --images /data/train --out /data/train_masks --gpu

# Фаза 2: perceptual fine-tuning с инициализацией из фазы 1
python train.py -c configs/pico_perceptual.yaml --out runs/perc --init runs/mse/last.pt \
    --set "data.train=[/data/train]" data.mask_root=/data/train_masks
```

Любой параметр конфига переопределяется через `--set key.sub=value` (значение
разбирается как YAML). Продолжить прерванное обучение: `--resume runs/mse/last.pt`.
Лог пишется в stdout и в `<out>/log.jsonl`.

Одна модель покрывает весь диапазон битрейтов. Каждому сэмплу в батче
назначается случайный уровень `l ∈ {0, 0.1, …, 7.0}` (71 уровень), и для него
берутся свои `λ_l` и `α_l`:
`L = E_l[α_l·D(x, x̂_l) + α_l·λ_l·R]`.

## Inference

```bash
python pico_codec.py encode -m runs/perc/last.pt -q 3.5 image.png image.pico
python pico_codec.py decode -m runs/perc/last.pt image.pico recon.png
python evaluate.py -m runs/perc/last.pt -d /data/clic2020_test -q 0 1 2 3 4 5 6 7 --lpips
```

`-q` задаёт уровень качества от 0 (минимальный битрейт) до 7 с шагом 0.1.
При загрузке модели слои ConvScale и Haar сворачиваются в обычные свёртки
(`--no-reparam` это отключает). Нейросетевые части работают на `--device`, а
scale decoder и энтропийное кодирование всегда выполняются на CPU.

## Что реализовано по статье

| Компонент | Где в статье | Реализация |
|---|---|---|
| Hyperprior с разделением hyper-decoder на **scale decoder** (σ) и **context decoder** (p); hyper-encoder встроен в encoder | Sec. 3.1, Fig. 3 | `model.py` |
| Factorized entropy model для `ŷ_hyper`, гауссиана со scale-таблицей для `ŷ` | Fig. 3 | compressai `EntropyBottleneck` / `GaussianConditional` (rANS) |
| **One-shot context model**: σ предсказывается за один проход, μ и learned quantization width q — авторегрессивно в 4 шага (сетка 2×2, лучший вариант в Table 1) | Sec. 4.1, Fig. 3 | `OneShotContext` |
| Квантование `ŷ = round((y−μ)/q)`, деквантование `qŷ + μ` | Sec. 4.1 | там же |
| **ConvScale**: `W' = s_in·s_out·W`, `b' = s_out·b`; при inference сворачивается в обычную свёртку | Sec. 4.1 | `layers.ConvScale` |
| **LearnedScale** в конце каждого блока пространственного масштаба | Sec. 4.1, Fig. 4 | `layers.LearnedScale` |
| **ConvScale311** (3x3 → GELU → 1x1 → GELU → 1x1 + residual), CS-Chain | Fig. 4 | `layers.py` |
| Группы по 32 канала в 3x3-свёртках | App. B | `group_size=32` |
| **Haar / iHaar ресемплинг** везде; Haar сворачивается с 1x1-свёрткой | Sec. 4.1, App. H | `HaarConv1x1`, `Conv1x1iHaar` + `reparameterize()` |
| Архитектура outer encoder/decoder с найденными NAS гиперпараметрами | Table 3, Fig. 4, 11 | `PICOConfig` |
| Scale decoder: 1x1(192→128), CS-Chain(128,2,4,1), 1x1iHaar ×2, 1x1(128→C) | Fig. 11 | `ScaleDecoder` |
| Управление качеством: 8 грубых уровней → 71 с интерполяцией one-hot; ConcatLevel в encoder/decoder; level-conditional channel-wise gain на латенте и обратный ему | App. F | `layers.level_weights`, `PICO.gain` |
| Loss: MSE + w1·LPIPS + w2·MS-SSIM + w3·TilingArtifactLoss + w4·TextFidelityLoss + w5·GAN | Sec. 4.2, Eq. 1 | `losses.py` |
| TextFidelityLoss: тяжёлый L1 в маске текста (CRAFT), GAN в этих областях ослаблен | Sec. 4.2 | `text_fidelity_loss`, `generator_loss(mask)` |
| TilingArtifactLoss: многомасштабный L1 по низким частотам | Sec. 4.2 | `tiling_artifact_loss` |
| PatchGAN-дискриминатор с увеличенными каналами и числом слоёв | Sec. 4.2 | `discriminator.py` |
| Двухфазное обучение, Adam, расписания LR, warm-up веса GAN | Sec. 3.3, 4.2, App. C | `train.py`, `configs/` |
| Тайлы 504×504, вход 512×512 с 4-пиксельным контекстом от соседей | Sec. 3.2 | `codec.py` |
| Scale decoder на CPU, чтобы энкодер и декодер получали одинаковые индексы CDF | Sec. 3.2 | `PICO.deterministic_sigma` |

## Что в статье не указано и выбрано нами

Эти значения стоит подбирать, они вынесены в конфиги или в `PICOConfig`:

- **Число каналов латента `C`**: 192. Число каналов `ŷ_hyper` (192) взято из Fig. 11.
- **Архитектура hyper-ветки энкодера и context decoder**: зеркало scale decoder из
  Fig. 11. Ширина prior `p` равна 128.
- **Predictor_n**: 1x1 → CS-Chain(128, 2, 2, 1) → 1x1 → (μ, q), где
  `q = 0.05 + softplus(·)`. Порядок шагов внутри блока 2×2: (0,0), (1,1), (0,1), (1,0).
- **Где стоит ConcatLevel в энкодере**: one-hot уровня конкатенируется ко входу
  перед первым HaarConv1x1. Это эквивалентно конкатенации после Haar, потому что
  Haar от константы даёт только LL.
- **Прокси квантования при обучении**: для rate — равномерный шум, для входа
  декодера и предикторов — STE-округление (как в большинстве современных работ).
- **λ_l**: восемь λ из CompressAI для MSE, переведённые в множитель rate
  (`λ_R = 1/(λ·255²)`), с геометрической интерполяцией между уровнями. **α_l = 1.**
- **Веса w1…w5**: по соотношениям из HiFiC (LPIPS ≈ 1/150 от MSE на [0,1], GAN 1e-3),
  TAL 0.02, текст 0.1. Длина warm-up GAN — 20k шагов.
- **Начальный LR фазы 2** (1e-4), число шагов (1M + 500k), batch 8, кроп 256,
  аугментации (случайный даунскейл, горизонтальный флип).
- **Дискриминатор**: безусловный, со spectral norm, non-saturating loss; Adam
  lr 1e-4, betas (0.5, 0.999).
- **Инициализация**: s_out последней свёртки residual-ветки ConvScale311 равен 0.1.
  Последний слой предикторов инициализируется так, что μ≈0 и q≈1.

## Отличия и ограничения

- **Кросс-платформенная детерминированность.** В статье scale decoder квантуется
  в UINT8, а оставшиеся FP-операции выполняются на CPU. Здесь scale decoder
  работает на CPU в FP32. Энкодер и декодер на одной платформе (одна версия
  PyTorch и одинаковое число потоков) дают побитово одинаковые σ, но между
  разными платформами это не гарантировано. Для такой гарантии scale decoder
  нужно квантовать в int8 (например, через `torch.ao.quantization`), это следующий шаг.
- **Конвейерная обработка тайлов** (CPU и ускоритель параллельно) не реализована:
  тайлы кодируются последовательно.
- **NAS** (Sec. 4.3) не воспроизводится: используются найденные в статье значения
  из Table 3.
- **Размер модели.** В статье у энкодера и декодера 15.2M и 9.6M параметров. В этой
  реализации outer encoder ≈1.1M (+1.6M hyper-ветка), outer decoder ≈1.0M,
  scale/context decoder ≈1.6M каждый, предикторы ≈1.3M. Вероятно, авторы
  считают параметры иначе или используют другие размеры, которые в статье не
  указаны (C, ширина вспомогательных сетей). Размеры настраиваются через `PICOConfig`.
- **Маски текста** нужно посчитать заранее (`scripts/make_text_masks.py`). Без
  них TextFidelityLoss отключён.
- **LPIPS** при первом запуске скачивает веса VGG.

## Тесты

```bash
python -m pytest -q tests
```

Тесты проверяют:
- обратимость Haar и эквивалентность репараметризации (ConvScale, HaarConv1x1,
  Conv1x1iHaar, ConvScale311);
- интерполяцию уровней качества;
- побитовое совпадение символов после энтропийного кодирования и декодирования;
- что one-shot context decode на стороне декодера воспроизводит латент энкодера;
- тайлинг и детерминированность декодирования;
- что `train.py` проходит обе фазы и CLI кодирует и декодирует полученным чекпойнтом.
