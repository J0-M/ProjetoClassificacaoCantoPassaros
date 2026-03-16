# Transfer Learning com espectrogramas

Este modulo cria um pipeline novo para:
1. Gerar espectrogramas Mel a partir de segmentos de audio.
2. Treinar modelos de transfer learning (`resnet50` e `efficientnet_b0`) com validacao cruzada estratificada por grupo (`audioSource`).

## Estrutura

- `gerar_espectrogramas_segmentos.py`: cria imagens `.png` e um manifesto (`manifest.csv`).
- `treinar_transfer_learning.py`: treina modelos em cima do manifesto.
- `checkpoints/`: modelos salvos por fold.

## Exemplo de uso

### 1) Gerar espectrogramas

```bash
python transferLearning/gerar_espectrogramas_segmentos.py ^
  --segments-csv C:/dados/df_ROI.csv ^
  --audio-root C:/dados/wavs_20241104 ^
  --output-dir transferLearning/datasets/spec_v1 ^
  --jobs 4
```

Por padrao, os espectrogramas agora cobrem toda a faixa de frequencia do audio original:
- `--sr` omitido (`None`), mantendo o sample rate original de cada arquivo
- `--fmin 0`
- `--fmax` omitido (`None`), que usa automaticamente ate a frequencia de Nyquist (`sr/2`).

Se quiser forcar reamostragem para um valor fixo (ex.: `32000`), passe `--sr 32000`.

### 2) Treinar EfficientNet

```bash
python transferLearning/treinar_transfer_learning.py ^
  --manifest transferLearning/datasets/spec_v1/manifest.csv ^
  --model efficientnet_b0 ^
  --output-dir transferLearning/outputs/spec_v1/efficientnet_b0
```

### 3) Treinar ResNet50

```bash
python transferLearning/treinar_transfer_learning.py ^
  --manifest transferLearning/datasets/spec_v1/manifest.csv ^
  --model resnet50 ^
  --output-dir transferLearning/outputs/spec_v1/resnet50
```

## Colunas esperadas no CSV de segmentos

Padrao:
- `soundscape_file`
- `roi_label`
- `roi_start`
- `roi_end`

Voce pode trocar via argumentos (`--audio-col`, `--label-col`, etc.).

## Dependencias

- `torch`
- `torchvision`
- `pandas`
- `numpy`
- `Pillow`
- `librosa`
- `scikit-learn`
- `joblib`
- `tqdm`

## Observacoes de desenho

- O split usa `StratifiedGroupKFold`, com grupo em `audioSource`, para reduzir leakage entre treinos e testes.
- O treino ocorre em 2 fases:
  1. Cabeca de classificacao (backbone congelado).
  2. Fine-tuning com backbone liberado.
