from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .data import UnpairedImageDataset
from .models import NLayerDiscriminator, ResnetGenerator, init_weights
from .utils import ImagePool, ensure_dir, seed_everything, set_requires_grad


def gan_loss(prediction: torch.Tensor, target_is_real: bool, criterion: nn.Module) -> torch.Tensor:
    target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
    return criterion(prediction, target)


def resolve_device(gpu_ids: str | None) -> torch.device:
    if gpu_ids in {None, "", "-1"} or not torch.cuda.is_available():
        return torch.device("cpu")
    first_id = str(gpu_ids).split(",")[0].strip()
    return torch.device(f"cuda:{first_id}")


def lr_lambda(epoch: int, epochs: int, decay_epochs: int) -> float:
    if epoch < epochs:
        return 1.0
    return max(0.0, 1.0 - (epoch - epochs + 1) / float(decay_epochs + 1))


def save_checkpoint(path: Path, epoch: int, args, models, optimizers) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "args": vars(args),
            "G_A": models["G_A"].state_dict(),
            "G_B": models["G_B"].state_dict(),
            "D_A": models["D_A"].state_dict(),
            "D_B": models["D_B"].state_dict(),
            "optimizer_G": optimizers["G"].state_dict(),
            "optimizer_D_A": optimizers["D_A"].state_dict(),
            "optimizer_D_B": optimizers["D_B"].state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, models, optimizers, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    for key in ("G_A", "G_B", "D_A", "D_B"):
        models[key].load_state_dict(checkpoint[key], strict=True)
    if "optimizer_G" in checkpoint:
        optimizers["G"].load_state_dict(checkpoint["optimizer_G"])
        optimizers["D_A"].load_state_dict(checkpoint["optimizer_D_A"])
        optimizers["D_B"].load_state_dict(checkpoint["optimizer_D_B"])
    return int(checkpoint.get("epoch", 0))


def save_samples(output_dir: Path, epoch: int, batch, fake_a, fake_b, max_items: int = 4) -> None:
    if max_items <= 0:
        return
    ensure_dir(output_dir)
    rows = []
    item_count = min(max_items, batch["A"].size(0))
    for index in range(item_count):
        rows.extend(
            [
                batch["A"][index].detach().cpu(),
                fake_b[index].detach().cpu(),
                batch["B"][index].detach().cpu(),
                fake_a[index].detach().cpu(),
            ]
        )
    if rows:
        grid = torch.stack(rows, dim=0)
        save_image(grid, output_dir / f"epoch_{epoch:04d}.jpg", nrow=4, normalize=True, value_range=(-1, 1))


def train(args) -> None:
    seed_everything(args.seed)
    device = resolve_device(args.gpu_ids)
    checkpoint_dir = ensure_dir(Path(args.checkpoints_dir) / args.name)
    sample_dir = ensure_dir(Path(args.outputs_dir) / args.name)

    dataset = UnpairedImageDataset(
        dataroot=args.dataroot,
        phase="train",
        image_size=args.load_size,
        crop_size=args.crop_size,
        augment=True,
        max_images=args.max_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    models = {
        "G_A": ResnetGenerator(3, 3, ngf=args.ngf, num_blocks=args.res_blocks).to(device),
        "G_B": ResnetGenerator(3, 3, ngf=args.ngf, num_blocks=args.res_blocks).to(device),
        "D_A": NLayerDiscriminator(3, ndf=args.ndf).to(device),
        "D_B": NLayerDiscriminator(3, ndf=args.ndf).to(device),
    }
    for model in models.values():
        model.apply(init_weights)

    optimizers = {
        "G": torch.optim.Adam(
            itertools.chain(models["G_A"].parameters(), models["G_B"].parameters()),
            lr=args.lr,
            betas=(args.beta1, 0.999),
        ),
        "D_A": torch.optim.Adam(models["D_A"].parameters(), lr=args.lr, betas=(args.beta1, 0.999)),
        "D_B": torch.optim.Adam(models["D_B"].parameters(), lr=args.lr, betas=(args.beta1, 0.999)),
    }

    start_epoch = 1
    if args.resume:
        loaded_epoch = load_checkpoint(Path(args.resume), models, optimizers, device)
        start_epoch = loaded_epoch + 1

    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: lr_lambda(start_epoch + epoch - 1, args.epochs, args.decay_epochs),
        )
        for optimizer in optimizers.values()
    ]

    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    fake_a_pool = ImagePool(args.pool_size)
    fake_b_pool = ImagePool(args.pool_size)
    total_epochs = args.epochs + args.decay_epochs

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_started = time.time()
        for step, batch in enumerate(loader, start=1):
            real_a = batch["A"].to(device, non_blocking=True)
            real_b = batch["B"].to(device, non_blocking=True)

            set_requires_grad([models["D_A"], models["D_B"]], False)
            optimizers["G"].zero_grad(set_to_none=True)

            fake_b = models["G_A"](real_a)
            rec_a = models["G_B"](fake_b)
            fake_a = models["G_B"](real_b)
            rec_b = models["G_A"](fake_a)

            loss_g_a = gan_loss(models["D_B"](fake_b), True, criterion_gan)
            loss_g_b = gan_loss(models["D_A"](fake_a), True, criterion_gan)
            loss_cycle_a = criterion_cycle(rec_a, real_a) * args.lambda_cycle
            loss_cycle_b = criterion_cycle(rec_b, real_b) * args.lambda_cycle

            loss_identity = torch.tensor(0.0, device=device)
            if args.lambda_identity > 0:
                idt_b = models["G_A"](real_b)
                idt_a = models["G_B"](real_a)
                loss_identity = (
                    criterion_identity(idt_b, real_b) + criterion_identity(idt_a, real_a)
                ) * args.lambda_cycle * args.lambda_identity

            loss_g = loss_g_a + loss_g_b + loss_cycle_a + loss_cycle_b + loss_identity
            loss_g.backward()
            optimizers["G"].step()

            set_requires_grad([models["D_A"], models["D_B"]], True)
            optimizers["D_A"].zero_grad(set_to_none=True)
            pred_real_a = models["D_A"](real_a)
            pred_fake_a = models["D_A"](fake_a_pool.query(fake_a.detach()))
            loss_d_a = (
                gan_loss(pred_real_a, True, criterion_gan)
                + gan_loss(pred_fake_a, False, criterion_gan)
            ) * 0.5
            loss_d_a.backward()
            optimizers["D_A"].step()

            optimizers["D_B"].zero_grad(set_to_none=True)
            pred_real_b = models["D_B"](real_b)
            pred_fake_b = models["D_B"](fake_b_pool.query(fake_b.detach()))
            loss_d_b = (
                gan_loss(pred_real_b, True, criterion_gan)
                + gan_loss(pred_fake_b, False, criterion_gan)
            ) * 0.5
            loss_d_b.backward()
            optimizers["D_B"].step()

            if step % args.log_every == 0:
                print(
                    " ".join(
                        [
                            f"epoch={epoch}/{total_epochs}",
                            f"step={step}/{len(loader)}",
                            f"G={loss_g.item():.4f}",
                            f"D_A={loss_d_a.item():.4f}",
                            f"D_B={loss_d_b.item():.4f}",
                            f"cycle={(loss_cycle_a + loss_cycle_b).item():.4f}",
                        ]
                    ),
                    flush=True,
                )

        for scheduler in schedulers:
            scheduler.step()

        save_samples(sample_dir, epoch, batch, fake_a, fake_b, max_items=args.sample_count)
        save_checkpoint(checkpoint_dir / "latest.pt", epoch, args, models, optimizers)
        if epoch % args.save_every == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pt", epoch, args, models, optimizers)

        elapsed = time.time() - epoch_started
        print(f"finished epoch {epoch} in {elapsed / 60:.1f} min", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CycleGAN for photo-to-anime translation.")
    parser.add_argument("--dataroot", default="data", help="Directory containing trainA and trainB.")
    parser.add_argument("--name", default="anime_cyclegan", help="Experiment name.")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--load-size", type=int, default=286)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--decay-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=0.5)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=9)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gpu-ids", default="0", help="Use -1 for CPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
