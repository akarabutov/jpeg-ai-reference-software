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
    discriminator.py  PatchGAN-дискриминатор увеличенной ёмкости, GAN-лоссы, R1
    lightning_module.py  LightningModule: шаги кодека и дискриминатора, валидация, логи
    data.py           датасет случайных кропов (+ маски текста), валидационный датасет
    codec.py          тайлинг 504/512 и формат битстрима
    quant.py          целочисленный (INT8/UINT8) scale decoder и его калибровка
    textmask.py       маски текста: детекторы, тайлы, компоненты, фильтр распознаванием
  train.py            обучение на PyTorch Lightning + TensorBoard
                      (фаза 1: MSE, фаза 2: perceptual + GAN)
  pico_codec.py       CLI: encode / decode
  evaluate.py         оценка на папке с изображениями по реальным битстримам
  compute_metrics.py  все метрики loss-функции по каждому PNG датасета -> CSV
  scripts/make_text_masks.py   маски текста: CRAFT (EasyOCR) или PaddleOCR, по карте вероятностей
  scripts/quantize_scale_decoder.py   послойное квантование scale decoder в INT8/UINT8
  configs/pico_mse.yaml, configs/pico_perceptual.yaml
  tests/test_pico.py
```

## Установка

```bash
cd pico
pip install -r requirements.txt
```

## Обучение

Обучение написано на **PyTorch Lightning**, логи пишутся в **TensorBoard**.

```bash
# Фаза 1: только MSE (lr 8e-4, ×0.3 на 70%, ×0.1 на 90% шагов)
python train.py -c configs/pico_mse.yaml --out runs/mse \
    --set "data.train=[/data/div2k,/data/flickr2k,/data/clic]" data.val=/data/clic_valid

# (опционально) маски текста для TextFidelityLoss
python scripts/make_text_masks.py --images /data/train --out /data/train_masks --gpu

# Фаза 2: perceptual + GAN с инициализацией из фазы 1
python train.py -c configs/pico_perceptual.yaml --out runs/perc \
    --init runs/mse/checkpoints/last.ckpt \
    --set "data.train=[/data/train]" data.mask_root=/data/train_masks

tensorboard --logdir runs
```

- Любой параметр конфига переопределяется через `--set key.sub=value`
  (значение разбирается как YAML).
- `--resume <ckpt>` продолжает обучение с полным состоянием: оптимизаторы,
  расписания LR, дискриминатор, счётчик итераций.
- `--init <ckpt>` загружает только веса кодека (переход от фазы 1 к фазе 2).
- **Несколько GPU:** `--set train.devices=4 train.strategy=ddp_find_unused_parameters_true`.
  `find_unused_parameters` нужен, потому что на шаге дискриминатора параметры
  кодека не участвуют.
- **Точность:** `train.precision=32-true` (по умолчанию) или `bf16-mixed`.
  fp16 со скейлером не поддерживается, потому что градиенты клипаются вручную.

**Чекпойнты** сохраняются в `<out>/checkpoints/` (`iter_<N>.ckpt` и `last.ckpt`).
В каждый из них дополнительно кладётся «чистый» кодек (`model`, `model_cfg`),
поэтому `pico_codec.py` и `evaluate.py` принимают `.ckpt` напрямую.

**Что пишется в TensorBoard** (`<out>/tb/version_N`, одна точка на итерацию):

| Тег | Что это |
|---|---|
| `train/loss`, `train/distortion`, `train/bpp`, `train/mse`, `train/psnr` | основной loss и его компоненты (средние по случайным уровням качества) |
| `train/lpips`, `train/msssim_loss`, `train/tal`, `train/text_l1` | перцептивные члены (фаза 2) |
| `train/aux_loss`, `train/grad_norm`, `train/lr`, `train/iteration` | служебные |
| `train/gan/g_loss`, `train/gan/d_loss`, `train/gan/d_real_logit`, `train/gan/d_fake_logit`, `train/gan/d_acc`, `train/gan/r1`, `train/gan/weight` | GAN (см. ниже) |
| `val/q{0,3.5,7}/bpp`, `val/q{…}/psnr` | валидация на уровнях `train.val_levels` |
| `val_images/<i>` | оригинал и реконструкции на этих уровнях (слева направо) |

Одна модель покрывает весь диапазон битрейтов. Каждому сэмплу в батче
назначается случайный уровень `l ∈ {0, 0.1, …, 7.0}` (71 уровень), и для него
берутся свои `λ_l` и `α_l`:
`L = E_l[α_l·D(x, x̂_l) + α_l·λ_l·R]`.

## Обучение с GAN (фаза 2)

### Что сказано в статье (Sec. 4.2, App. C)

- GAN заметно улучшает перцептивное качество. Но без pixel-matching и
  перцептивных лоссов (MSE, LPIPS, MS-SSIM) он даёт артефакты и галлюцинации,
  поэтому эти члены остаются и регуляризуют GAN.
- Дискриминатор — патчевый, как PatchGAN (Isola et al.), но с бо́льшим числом
  каналов и слоёв. Чем сильнее дискриминатор, тем лучше генерация.
- Сильный дискриминатор при лёгком декодере делает обучение нестабильным. Статья
  стабилизирует его двумя способами:
  1. **двухфазное обучение**: сначала только MSE, чтобы GAN стартовал с разумной
     инициализации;
  2. **warm-up**: вес надзора от дискриминатора постепенно увеличивается, пока
     дискриминатор ещё «молодой» и может сбить кодек с толку.
- В областях текста GAN ослаблен, а вместо него действует тяжёлый L1
  (TextFidelityLoss).

Конкретных значений (веса, LR, архитектура дискриминатора, тип GAN-лосса)
в статье нет. Всё ниже — наши настройки по умолчанию, они вынесены в
`configs/pico_perceptual.yaml`.

### Как устроена одна итерация

Используется ручная оптимизация Lightning (`automatic_optimization = False`):

```
x, маски текста m, случайные уровни l ∈ {0, 0.1, …, 7}

1) Шаг кодека (дискриминатор заморожен через toggle_optimizer):
     x̂, likelihoods = PICO(x, l)
     g_gan = generator_loss(D(x̂), m)                   # если w_gan(t) > 0
     D_total = MSE + w1·LPIPS + w2·(1−MS-SSIM) + w3·TAL + w4·TextL1(m) + w_gan(t)·g_gan
     L = mean_b[ α_l · (D_total + λ_l · bpp) ] + aux_loss(квантили entropy bottleneck)
     backward → clip_grad_norm(параметры кодека, 1.0) → Adam.step

2) Шаг дискриминатора (кодек заморожен):
     d_loss = discriminator_loss(D(x), D(x̂.detach()))
     каждые r1_every итераций: + 0.5 · γ · r1_every · E‖∇ₓD(x)‖²   (lazy R1)
     backward → Adam.step

3) Шаг расписаний LR кодека и дискриминатора (одни и те же точки снижения).
```

### Расписание веса GAN

```
w_gan(t) = 0                                            при t < gan_start_step
         = w_gan · min(1, (t − gan_start_step + 1) / gan_warmup_steps)   иначе
```

- **`gan_start_step`** (по умолчанию 5000): разогрев дискриминатора. Он учится
  с первой итерации, но кодек получает от него градиент только после этого
  шага. Так кодек не подстраивается под ещё случайный дискриминатор.
- **`gan_warmup_steps`** (20000): линейный рост веса — warm-up из статьи.
- **`w_gan`** (1e-3): итоговый вес. Остальные члены измеряются на шкале
  MSE по [0, 1], поэтому этот вес примерно соответствует β = 0.15 из HiFiC.

### Лоссы и маска текста

- `disc.loss: ns` — non-saturating: `D: softplus(−D(x)) + softplus(D(x̂))`,
  `G: softplus(−D(x̂))`.
- `disc.loss: hinge` — `D: relu(1 − D(x)) + relu(1 + D(x̂))`, `G: −D(x̂)`.
- **Маска текста** уменьшается до сетки патчей дискриминатора через max-pool:
  патч считается текстовым, если в нём есть хотя бы один пиксель текста. GAN-лосс
  генератора в таких патчах умножается на `1 − m`, то есть выключается, а
  фидельность текста обеспечивает TextFidelityLoss. Дискриминатор маску не видит.

### Дискриминатор

PatchGAN, вход переводится в [−1, 1]:

- свёртки 4×4: 128 → 256 → 512 (stride 2) → 1024 (stride 1);
- ещё одна свёртка 3×3 (1024 → 1024): это и есть «больше слоёв»;
- выходная свёртка 4×4 → 1 логит на патч;
- LeakyReLU(0.2) и spectral norm во всех свёртках;
- рецептивное поле 86 пикселей, ≈ 20.5M параметров; на кропе 256×256 выход 30×30 логитов.

Размеры задаются в `disc.arch`. Оптимизатор — Adam(lr 1e-4, betas 0.5/0.999),
LR снижается в те же моменты, что и у кодека.

### Что смотреть в TensorBoard и что делать при проблемах

| Симптом | Что видно | Что попробовать |
|---|---|---|
| Дискриминатор «выиграл» | `gan/d_acc` ≈ 1, `gan/d_real_logit` ≫ 0 ≫ `gan/d_fake_logit`, `gan/g_loss` растёт | увеличить `gan_warmup_steps`, уменьшить `disc.lr`, включить `disc.r1_gamma` (0.1–10), уменьшить `disc.arch` |
| Дискриминатор ничего не различает | `gan/d_acc` ≈ 0.5 долгое время, `gan/d_loss` ≈ 1.39 (= 2·ln 2) для `ns` | увеличить `disc.lr` или ёмкость, проверить `gan/weight` > 0 |
| Галлюцинации, «шум» на гладких областях | на `val_images` видны паттерны, LPIPS падает, PSNR резко падает | уменьшить `w_gan`, увеличить `w_lpips` / MSE-часть |
| Нестабильность, выбросы | скачки `train/grad_norm` и `gan/d_loss` | `disc.loss: hinge`, R1, меньший LR, `bf16-mixed` → `32-true` |
| Цвет «плывёт» между тайлами | на стыках тайлов видна разница цвета | увеличить `w_tiling` (TAL) |
| Текст нечитаем | плохие `train/text_l1` и примеры с текстом | посчитать маски (`make_text_masks.py`), увеличить `w_text` |

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

## Метрики по датасету в CSV

`compute_metrics.py` проходит по всем PNG в папке (рекурсивно). Для каждого
изображения он делает реальное кодирование в битстрим и декодирование (с
выходом в 8 бит) и считает все члены loss-функции теми же функциями, что при
обучении. Результат — CSV, одна строка на изображение (или на пару изображение
× уровень качества, если задано несколько `-q`):

```bash
python compute_metrics.py -m runs/perc/checkpoints/last.ckpt -d /data/kodak -q 3.5 \
    --out kodak_metrics.csv
python compute_metrics.py -m runs/perc/checkpoints/last.ckpt -d /data/clic -q 0 2 4 6 \
    --masks /data/clic_masks --out clic_metrics.csv
```

Пример строки (чекпойнт короткого пробного обучения, `-q 3.6 --no-lpips`, без масок):

```
file,bpp,mse,psnr,lpips,ms_ssim,msssim_loss,tal,text_l1,gan_g,d_logit,distortion,rd_loss
kodim15.png,0.867086,0.00322782,24.9109,,0.922942,0.0770578,0.0318503,,1.62707,-1.33124,0.00600048,0.00669333
```

| Столбец | Что это |
|---|---|
| `file` | путь относительно `-d` |
| `quality` | уровень качества; столбец есть, только если `-q` задано несколько раз |
| `bpp` | реальный размер битстрима: бит / пиксель |
| `mse`, `psnr` | по изображениям в [0, 1] |
| `lpips` | LPIPS-VGG, как в loss (`--no-lpips` — пропустить: при первом запуске скачиваются веса VGG) |
| `ms_ssim`, `msssim_loss` | MS-SSIM и член loss `1 − MS-SSIM` |
| `tal` | TilingArtifactLoss |
| `text_l1` | TextFidelityLoss; нужен `--masks` (те же относительные пути, `.png`); если файла маски нет, считается, что текста нет |
| `gan_g`, `d_logit` | GAN-лосс кодека и средний логит дискриминатора на реконструкции. Дискриминатор берётся из Lightning-чекпойнта фазы 2, с учётом маски текста |
| `distortion`, `rd_loss` | `D` и `α_l·(D + λ_l·bpp)` с весами, λ и α из конфига чекпойнта (`--loss-config` переопределяет); в суммы входят только доступные члены |

Пустая ячейка значит, что метрика недоступна. CSV пишется построчно, поэтому
при прерывании готовые строки сохраняются. В конце печатаются средние по
каждому уровню качества. `--no-8bit` считает метрики по float-реконструкции,
а не по 8-битной.

## Квантование scale decoder

Энтропийный декодер получает индексы CDF от scale decoder. Если они хоть немного
расходятся между энкодером и декодером (другой CPU, другая библиотека, другое
число потоков), файл не раскодируется. В статье (Sec. 3.2) scale decoder
поэтому квантуется в UINT8. Скрипт делает это **послойно** после обучения и
полностью убирает плавающую точку из рантайма:

```bash
python scripts/quantize_scale_decoder.py -m runs/perc/checkpoints/last.ckpt \
    --calib /data/calib_images --val /data/clic_valid --out runs/perc/pico_int.pt
python pico_codec.py encode -m runs/perc/pico_int.pt -q 3.5 image.png image.pico
python pico_codec.py decode -m runs/perc/pico_int.pt image.pico recon.png
```

**Схема квантования** (`picocodec/quant.py`):

| Элемент | Как квантуется |
|---|---|
| веса | INT8, симметрично, отдельный масштаб на каждый выходной канал |
| активации | UINT8, масштаб и нулевая точка на каждый тензор |
| вход | целые символы гиперлатента `k` (`ẑ = k + медианы`); медианы вшиты в bias первой свёртки |
| свёртка | целочисленный аккумулятор `Σ w_q·(x_q − zp) + b_q`, проверка `|acc| < 2³¹` |
| реквантование | `clamp(zp + round(acc·M0 / 2^S))` с целыми `M0`, `S` на каждый канал (как в gemmlowp/TFLite) |
| GELU | таблица UINT8 на 256 значений |
| residual-сложение | оба входа пересчитываются в выходной масштаб в фиксированной точке (2²⁴) |
| iHaar-апсемплинг | вшит в веса свёртки, остаётся только pixel shuffle (перестановка) |
| softplus + индекс CDF | целые пороги `T[c, i] = ⌊softplus⁻¹(t_i) / s_acc[c]⌋`; индекс = число порогов, которые превысил аккумулятор. Это ровно `build_indexes(softplus(·))` в точной арифметике |

Свёртки выполняются как float64-свёртки над целочисленными тензорами. Все
произведения и частичные суммы — целые числа много меньше 2⁵³, поэтому любая
реализация IEEE-754 float64 возвращает точный целый результат при любом порядке
суммирования, числе потоков, векторизации и FMA. На устройство ядра
переносятся 1:1 как int32 GEMM.

**Послойная калибровка.** Слои обрабатываются по порядку. Для каждого слоя:
- масштабы весов, коррекция bias и диапазон выходной активации подбираются на
  активациях, которые выдаёт **уже квантованный префикс** сети, так что ошибка
  предыдущих слоёв учитывается;
- коррекция bias (`--no-bias-correction` её отключает) вычитает из bias среднее
  по каналу смещение выхода квантованного слоя относительно float;
- диапазон активаций берётся по перцентилю `--percentile` (по умолчанию 0.9999).

**Что печатает скрипт:**
- SQNR каждой точки квантования относительно float-модели;
- долю индексов CDF, совпавших с float scale decoder;
- реальную стоимость кодирования латента `y` с обоими вариантами индексов;
- проверку детерминированности: 1 поток и по одному против многопоточного
  батча.

Результат сохраняется копией чекпойнта с ключом `scale_decoder_int`.
`pico_codec.py` и `evaluate.py` используют его автоматически (`--float-scale-decoder`
это отключает). В заголовке битстрима (версия 2) стоит флаг: файл, закодированный
целочисленным scale decoder, декодируется только им, и наоборот. При
несовпадении выдаётся понятная ошибка.

Пример на чекпойнте короткого пробного обучения: 32 кропа 512×512 для калибровки,
12 кропов Kodak для проверки, 8 уровней качества.

| Метрика | Значение |
|---|---|
| SQNR по слоям | 25–40 дБ, σ на выходе — 40.7 дБ |
| индексы CDF, совпавшие с float | 97.9% (отличие на 1 — 2.1%, больше 1 — 29 из 2.36M) |
| битрейт латента | 0.51213 → 0.51215 bpp (+0.004%) |
| размер весов | 1.56 MB (INT8) |
| детерминированность | индексы идентичны |

Реконструкции побитово совпадают с float-вариантом: σ влияет только на битрейт.

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
| Квантование scale decoder в UINT8 ради кросс-платформенной побитовой точности | Sec. 3.2 | `picocodec/quant.py`, `scripts/quantize_scale_decoder.py` |

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
- **Дискриминатор и детали GAN**: см. раздел «Обучение с GAN» выше.
- **Инициализация**: s_out последней свёртки residual-ветки ConvScale311 равен 0.1.
  Последний слой предикторов инициализируется так, что μ≈0 и q≈1.

## Отличия и ограничения

- **Кросс-платформенная детерминированность.** Без квантования scale decoder
  работает на CPU в FP32: на одной платформе σ побитово совпадают, между
  разными платформами — не гарантированно. Гарантию даёт целочисленный scale
  decoder (раздел «Квантование scale decoder»). В статье после квантования
  остаются FP-операции, которые выполняются на CPU, а здесь рантайм полностью
  целочисленный.
- **Конвейерная обработка тайлов** (CPU и ускоритель параллельно) не реализована:
  тайлы кодируются последовательно.
- **NAS** (Sec. 4.3) не воспроизводится: используются найденные в статье значения
  из Table 3.
- **Размер модели.** В статье у энкодера и декодера 15.2M и 9.6M параметров. В этой
  реализации outer encoder ≈1.1M (+1.6M hyper-ветка), outer decoder ≈1.0M,
  scale/context decoder ≈1.6M каждый, предикторы ≈1.3M. Вероятно, авторы
  считают параметры иначе или используют другие размеры, которые в статье не
  указаны (C, ширина вспомогательных сетей). Размеры настраиваются через `PICOConfig`.
- **Маски текста** нужно посчитать заранее (`scripts/make_text_masks.py`, см.
  раздел «Маски текста»). Без них TextFidelityLoss отключён.
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
- расписание веса GAN, GAN-лоссы, маскирование текста и R1;
- для квантования scale decoder:
  - float-граф совпадает со scale decoder;
  - реквантование в фиксированной точке и индексы совпадают с CompressAI;
  - целочисленный scale decoder близок к float и детерминирован по потокам и
    батчам, сериализация работает;
  - кодек с ним даёт те же реконструкции, а при несовпадении флага выдаётся ошибка;
  - скрипт квантования и CLI работают на обученном чекпойнте;
- что `train.py` (Lightning) проходит обе фазы: в TensorBoard есть все скаляры
  и картинки, по одной точке на итерацию; LR снижается как в статье; GAN
  включается по расписанию; `--init`, `--resume` и чекпойнты работают; CLI
  кодирует и декодирует прямо из `.ckpt`.
