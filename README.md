# Photo2Anime-CycleGAN

Photo2Anime-CycleGAN is a local AI image translation project that converts real portrait photos into anime-style images. It implements a compact CycleGAN pipeline in PyTorch, supports unpaired dataset preparation, GPU training, checkpoint-based inference, batch image conversion, and a simple web demo.

The project is designed for learning and experimentation: it can run a fast non-neural preview filter without a trained checkpoint, and it can train a real photo-to-anime CycleGAN model when paired with suitable `trainA` and `trainB` image folders.

## Features

- Photo-to-anime image translation using CycleGAN.
- Unpaired training data support: real photos and anime images do not need one-to-one matching.
- ResNet generator and PatchGAN discriminator implemented locally in PyTorch.
- Dataset preprocessing and starter-data download scripts.
- CLI tools for training, inference, and preprocessing.
- Gradio web demo with a built-in HTTP fallback if Gradio is unavailable.
- CUDA GPU support for NVIDIA cards.
- Classical anime-style preview filter when no checkpoint is available.

## How It Works

CycleGAN learns from two independent image domains:

```text
data/trainA/  real portrait photos
data/trainB/  anime portrait images
```

The model trains two translators:

```text
G_A: real photo -> anime image
G_B: anime image -> real photo
```

It also trains two discriminators that judge whether generated images look like the target domain. The cycle-consistency loss encourages the model to change style while preserving the original content structure.

After training, the important file is:

```text
checkpoints/anime_cyclegan/latest.pt
```

That checkpoint can be reused for future inference without retraining.

## Project Structure

```text
anime_style/
  classical.py        # Non-neural anime preview filter
  data.py             # Dataset loading and image transforms
  infer.py            # Inference wrapper and checkpoint loading
  models.py           # ResNet generator and PatchGAN discriminator
  preprocess.py       # Image preprocessing helpers
  train.py            # CycleGAN training loop
  utils.py            # Shared utilities

scripts/
  download_starter_data.py  # Download starter real/anime portrait data
  infer.py                  # CLI inference entrypoint
  preprocess.py             # CLI preprocessing entrypoint
  train.py                  # CLI training entrypoint

web_demo/
  app.py              # Gradio / built-in web demo

data/
  README.md           # Dataset layout notes
  trainA/             # Real photos, ignored by git
  trainB/             # Anime images, ignored by git
  testA/              # Test photos, ignored by git

checkpoints/          # Trained model weights, ignored by git
outputs/              # Generated examples and logs, ignored by git
raw/                  # Download cache and raw data, ignored by git
```

## What Is Not Committed

This repository intentionally excludes large or generated files:

- Python virtual environments: `.venv/`
- Downloaded datasets: `data/**/*.jpg`, `raw/`
- Training checkpoints: `checkpoints/`, `*.pt`, `*.pth`
- Generated images and logs: `outputs/`
- Dataset shards and archives: `*.parquet`, `*.zip`, `*.tar.gz`

Use the included scripts to recreate data and model artifacts locally.

## Requirements

- Python 3.9+
- NVIDIA GPU recommended for training
- CUDA-capable PyTorch recommended for fast training

Install dependencies:

```powershell
cd E:\big3\projects\cartoon
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For an RTX 40-series GPU, CUDA 12.8 PyTorch can be installed with:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Prepare Data

CycleGAN uses unpaired datasets:

```text
data/trainA/  real portrait photos
data/trainB/  anime portrait images
data/testA/   real photos for inference
```

You can preprocess your own image folders:

```powershell
.\.venv\Scripts\python.exe scripts\preprocess.py raw\photos data\trainA --size 256
.\.venv\Scripts\python.exe scripts\preprocess.py raw\anime data\trainB --size 256
```

You can also download a starter dataset:

```powershell
.\.venv\Scripts\python.exe scripts\download_starter_data.py --real-limit 2500 --anime-limit 2500 --image-size 256 --num-workers 24
```

The starter data script uses:

- Open Images validation images for real face crops.
- Hugging Face anime face datasets for anime portraits.

Review dataset licenses before using any data commercially.

## Train

Run a short smoke test first:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --dataroot data --name anime_cyclegan_test --gpu-ids 0 --batch-size 1 --epochs 1 --decay-epochs 0 --log-every 50 --save-every 1 --num-workers 2
```

Run a small first experiment:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --dataroot data --name anime_cyclegan_5ep --gpu-ids 0 --batch-size 1 --epochs 5 --decay-epochs 5 --log-every 100 --save-every 1 --num-workers 2
```

Run a longer first model:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --dataroot data --name anime_cyclegan --gpu-ids 0 --batch-size 1 --epochs 50 --decay-epochs 50 --log-every 100 --save-every 5 --num-workers 2
```

Resume interrupted training:

```powershell
.\.venv\Scripts\python.exe scripts\train.py --dataroot data --name anime_cyclegan --gpu-ids 0 --batch-size 1 --epochs 50 --decay-epochs 50 --log-every 100 --save-every 5 --num-workers 2 --resume checkpoints\anime_cyclegan\latest.pt
```

## Inference

Without a checkpoint, the project uses the fast preview filter:

```powershell
.\.venv\Scripts\python.exe scripts\infer.py data\testA --output outputs\preview
```

With a trained CycleGAN checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\infer.py data\testA --output outputs\anime --checkpoint checkpoints\anime_cyclegan\latest.pt
```

Single image inference:

```powershell
.\.venv\Scripts\python.exe scripts\infer.py input.jpg --output outputs\input_anime.png --checkpoint checkpoints\anime_cyclegan\latest.pt
```

## Web Demo

Start the demo without a checkpoint:

```powershell
.\.venv\Scripts\python.exe web_demo\app.py --host 127.0.0.1 --port 7860
```

Start with a trained checkpoint:

```powershell
.\.venv\Scripts\python.exe web_demo\app.py --checkpoint checkpoints\anime_cyclegan\latest.pt --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860/
```

## Training Tips

- Start with `--batch-size 1` on 8GB GPUs.
- Keep `trainA` and `trainB` visually similar. For portrait anime generation, both should be portrait-focused.
- Check `outputs/<experiment>/epoch_XXXX.jpg` regularly.
- Use short experiments first, such as `5 + 5` epochs, before running long training.
- If colors drift too much, tune `--lambda-identity`.
- If face structure changes too much, tune `--lambda-cycle`.

## Limitations

CycleGAN is a classic 2017 image-to-image translation method. It is useful for learning and local experiments, but it is not the current state of the art for commercial anime portrait generation. Modern diffusion pipelines such as Stable Diffusion, ControlNet, LoRA, IP-Adapter, and InstantID can usually produce higher-quality results.

This project is best viewed as a practical local training pipeline and a foundation for future upgrades.

## License and Data Notice

The code in this repository can be adapted for research and learning. Dataset images, downloaded starter data, model checkpoints, and generated outputs are not included in the repository. Always check source dataset licenses and portrait rights before commercial use.
