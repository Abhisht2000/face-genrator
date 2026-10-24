"""Train a DCGAN on a folder of face images.

Example:
python train.py --data data/faces --epochs 30 --batch-size 64
"""
import argparse
import csv
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import FaceFolderDataset
from models import build_models

ROOT = Path(__file__).resolve().parent


def choose_device(requested="auto"):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available. Try --device auto.")
        return torch.device("mps")
    if requested == "auto" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, epoch, generator, discriminator, opt_g, opt_d,
                    fixed_noise, history, latent_dim, base_channels):
    torch.save({
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict(),
        "fixed_noise": fixed_noise.detach().cpu(),
        "history": history,
        "latent_dim": latent_dim,
        "base_channels": base_channels,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train a mini DCGAN face generator.")
    parser.add_argument("--data", default="data/faces", help="Folder containing face images")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64, choices=[64])
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=0, help="0 is safest on macOS")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/latest.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)

    device = choose_device(args.device)
    print(f"Device: {device}")
    dataset = FaceFolderDataset(args.data, args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=False
    )
    if len(loader) == 0:
        raise RuntimeError(
            f"Dataset has {len(dataset)} images, fewer than batch size {args.batch_size}. "
            "Add more images or lower --batch-size."
        )
    print(f"Found {len(dataset):,} images in {args.data}")

    generator, discriminator = build_models(args.latent_dim, args.base_channels, device)
    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    checkpoints = ROOT / "checkpoints"
    samples_dir = ROOT / "samples"
    checkpoints.mkdir(exist_ok=True)
    samples_dir.mkdir(exist_ok=True)
    latest_path = checkpoints / "latest.pt"
    generator_path = checkpoints / "generator.pt"
    history_path = ROOT / "training_history.csv"

    history = {"g_loss": [], "d_loss": []}
    start_epoch = 0
    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError("No checkpoints/latest.pt found to resume from.")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if (checkpoint["latent_dim"] != args.latent_dim or
                checkpoint["base_channels"] != args.base_channels):
            raise ValueError("Model dimensions differ from checkpoint; use matching latent/base channel values.")
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        opt_g.load_state_dict(checkpoint["optimizer_g"])
        opt_d.load_state_dict(checkpoint["optimizer_d"])
        history = checkpoint.get("history", history)
        start_epoch = int(checkpoint["epoch"])
        print(f"Resuming after epoch {start_epoch}")

    fixed_noise = torch.randn(25, args.latent_dim, 1, 1, device=device)
    if args.resume and latest_path.exists():
        saved_noise = checkpoint.get("fixed_noise")
        if saved_noise is not None:
            fixed_noise = saved_noise.to(device)

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "generator_loss", "discriminator_loss"])
        for old_idx, (g, d) in enumerate(zip(history["g_loss"], history["d_loss"]), start=1):
            writer.writerow([old_idx, g, d])

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        generator.train()
        discriminator.train()
        running_g, running_d, steps = 0.0, 0.0, 0

        for real in loader:
            real = real.to(device)
            batch = real.size(0)
            real_targets = torch.ones(batch, device=device)
            fake_targets = torch.zeros(batch, device=device)

            # Train discriminator on real and detached fake images.
            opt_d.zero_grad(set_to_none=True)
            real_logits = discriminator(real)
            loss_real = criterion(real_logits, real_targets)
            noise = torch.randn(batch, args.latent_dim, 1, 1, device=device)
            fake = generator(noise)
            fake_logits = discriminator(fake.detach())
            loss_fake = criterion(fake_logits, fake_targets)
            d_loss = (loss_real + loss_fake) * 0.5
            d_loss.backward()
            opt_d.step()

            # Train generator to make discriminator classify fakes as real.
            opt_g.zero_grad(set_to_none=True)
            noise = torch.randn(batch, args.latent_dim, 1, 1, device=device)
            generated = generator(noise)
            generated_logits = discriminator(generated)
            g_loss = criterion(generated_logits, real_targets)
            g_loss.backward()
            opt_g.step()

            running_g += float(g_loss.detach().cpu())
            running_d += float(d_loss.detach().cpu())
            steps += 1

        avg_g, avg_d = running_g / steps, running_d / steps
        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)
        generator.eval()
        with torch.no_grad():
            preview = generator(fixed_noise)
            save_image(
                preview, samples_dir / f"epoch_{epoch + 1:03d}.png",
                nrow=5, normalize=True, value_range=(-1, 1)
            )

        save_checkpoint(
            latest_path, epoch + 1, generator, discriminator, opt_g, opt_d,
            fixed_noise, history, args.latent_dim, args.base_channels
        )
        torch.save({
            "generator": generator.state_dict(),
            "latent_dim": args.latent_dim,
            "base_channels": args.base_channels,
            "epoch": epoch + 1,
        }, generator_path)

        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, avg_g, avg_d])

        elapsed = time.time() - started
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"G loss: {avg_g:.4f} | D loss: {avg_d:.4f} | {elapsed:.1f}s"
        )

    print(f"Training complete. Generator: {generator_path}")
    print("Run: streamlit run app.py")


if __name__ == "__main__":
    main()
