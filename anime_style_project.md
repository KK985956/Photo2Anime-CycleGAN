# Anime Style Image Generation Project

## 1. 项目目标

本项目已经从概念说明升级为可运行代码项目，用于将真实照片转换为动漫风格图像。实现上保留原方案中的 CycleGAN 训练路线，同时加入轻量预览滤镜作为 fallback：在没有训练权重、GPU 或数据集尚未准备好时，也可以先运行命令行推理和 Web Demo。

核心目标：

- 使用未配对数据训练照片域 `A` 到动漫域 `B` 的 CycleGAN。
- 提供图片预处理、训练、批量推理和 Web Demo。
- 让项目在“没有 checkpoint”时仍能运行，便于演示和调试。

## 2. 当前目录结构

```text
anime-style-project/
├─ anime_style/
│  ├─ __init__.py
│  ├─ classical.py      # 无模型 fallback 动漫化滤镜
│  ├─ data.py           # 未配对数据集和增强
│  ├─ infer.py          # 推理封装，优先使用 CycleGAN 权重
│  ├─ models.py         # ResNet Generator / PatchGAN Discriminator
│  ├─ preprocess.py     # 图片预处理逻辑
│  ├─ train.py          # CycleGAN 训练循环
│  └─ utils.py
├─ scripts/
│  ├─ preprocess.py     # 预处理 CLI
│  ├─ train.py          # 训练 CLI
│  └─ infer.py          # 推理 CLI
├─ web_demo/
│  └─ app.py            # Web Demo，优先 Gradio，缺失时回退到内置 HTTP 页面
├─ data/
│  ├─ trainA/           # 真实照片
│  ├─ trainB/           # 动漫图片
│  └─ testA/            # 待转换照片
├─ checkpoints/         # 训练权重
├─ outputs/             # 样例和推理输出
├─ requirements.txt
├─ pyproject.toml
└─ README.md
```

## 3. 技术方案调整

原文档建议直接使用官方 CycleGAN 仓库。为了让本项目自包含、便于修改和交付，当前实现改为本地 PyTorch 版本：

- 生成器：ResNet Generator，默认 9 个残差块。
- 判别器：70x70 PatchGAN。
- 损失：LSGAN adversarial loss、cycle consistency loss、identity loss。
- 数据：`trainA` 与 `trainB` 未配对随机采样。
- Demo：优先使用 Gradio；环境未安装 Gradio 时自动回退到标准库 HTTP 页面。默认可使用本地 fallback 滤镜，加载 checkpoint 后使用深度模型。

这个结构的优势是代码集中、入口清晰、无需依赖外部仓库；后续如果要接入更强模型，也可以只替换 `anime_style/infer.py` 和模型模块。

## 4. 环境安装

建议 Python 3.9+。GPU 训练建议 NVIDIA 8GB+ 显存。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果要充分利用 CUDA，请先按 PyTorch 官网命令安装与本机驱动匹配的 torch/torchvision，再安装其余依赖。

## 5. 数据准备

数据目录约定：

```text
data/
├─ trainA/   # 真实照片
├─ trainB/   # 动漫图片
└─ testA/    # 测试照片
```

预处理命令：

```powershell
python scripts/preprocess.py raw\photos data\trainA --size 256
python scripts/preprocess.py raw\anime data\trainB --size 256
```

也可以先拉取一个 starter 数据集：

```powershell
python scripts/download_starter_data.py --real-limit 2500 --anime-limit 2500 --image-size 256
```

该脚本会把 Open Images 验证集中的 `Human face` 框裁剪保存到 `data/trainA`，并从 Hugging Face 动漫脸数据集提取图片保存到 `data/trainB`。数量超过 120 时，动漫侧会自动使用 `minoruskore/anime-faces-256`。这批数据适合跑通训练和做第一版模型，但如果要更稳定的商业级效果，仍建议继续清洗和扩充数据。

建议：

- `trainA` 和 `trainB` 内容类型尽量一致，例如都以半身人像或头像为主。
- 清理水印、边框、低清图、强压缩图。
- 初始训练可以每个域准备 1k-5k 张，先跑通流程后再扩充。

## 6. 训练

基础训练：

```powershell
python scripts/train.py --dataroot data --name anime_cyclegan --gpu-ids 0 --batch-size 1
```

常用参数：

- `--epochs`: 固定学习率阶段，默认 100。
- `--decay-epochs`: 线性衰减阶段，默认 100。
- `--load-size`: 先缩放尺寸，默认 286。
- `--crop-size`: 随机裁剪尺寸，默认 256。
- `--lambda-cycle`: 循环一致性权重，默认 10。
- `--lambda-identity`: identity loss 权重，默认 0.5。
- `--resume`: 从 checkpoint 继续训练。

训练产物：

```text
checkpoints/anime_cyclegan/latest.pt
outputs/anime_cyclegan/epoch_0001.jpg
```

其中 `latest.pt` 内的 `G_A` 是真实照片到动漫域的生成器。

## 7. 推理

无权重快速预览：

```powershell
python scripts/infer.py data\testA --output outputs\preview
```

使用训练权重：

```powershell
python scripts/infer.py data\testA --output outputs\anime --checkpoint checkpoints\anime_cyclegan\latest.pt
```

单图输出：

```powershell
python scripts/infer.py input.jpg --output outputs\input_anime.png --checkpoint checkpoints\anime_cyclegan\latest.pt
```

## 8. Web Demo

无权重启动：

```powershell
python web_demo/app.py
```

使用训练权重启动：

```powershell
python web_demo/app.py --checkpoint checkpoints\anime_cyclegan\latest.pt
```

启动后打开终端输出的本地地址。无权重时页面显示的是快速预览效果；加载 checkpoint 后才是 CycleGAN 深度模型输出。

## 9. 质量优化路线

- 优先提升数据质量，而不是盲目加深模型。
- 人像数据建议做人脸/半身区域裁剪，减少背景域差异。
- 如果生成结果结构变形，提高 `--lambda-cycle`。
- 如果颜色过度偏移或风格污染，调整 `--lambda-identity`。
- 如果训练震荡明显，降低学习率或减小判别器强度。
- 保存中间样例并按 epoch 对比，避免只看最终 checkpoint。

## 10. 后续扩展

- 增加 FID/LPIPS 评估脚本。
- 加入视频逐帧处理和帧间稳定策略。
- 接入 Stable Diffusion img2img 或 ControlNet 作为高质量推理后端。
- 增加人脸检测和自动裁剪，提高人像场景稳定性。
