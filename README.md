# Mini GAN-Based Human Face Image Generator

A small DCGAN project for training on a folder of face images and generating 64×64 RGB synthetic faces. Includes MPS support for Apple Silicon, checkpoints, epoch samples, CSV loss history, and a Streamlit UI.

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4) recommended; CPU fallback is supported.
- Python 3.11 or 3.12 is a sensible starting point.
- A user-provided face-image dataset that you have permission to use.
- Allow several GB of free disk space for Python packages, images, checkpoints, and samples.

## 1. Create environment

```bash
mkdir -p data/faces
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you are missing Apple's command line tools:

```bash
xcode-select --install
```

Check PyTorch and MPS:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

## 2. Add dataset images

Put JPG, JPEG, PNG, WEBP, or BMP face images inside `data/faces/`. Subfolders are scanned recursively. The project does not automatically download a face dataset: obtain one from its official source and follow its licence and usage conditions. For example, CelebA is restricted to non-commercial research use; review the current agreement before using it.

A dataset of a few thousand aligned/cropped faces is a reasonable initial experiment. Ensure you have rights/permission to use the images. Avoid collecting private or non-consensual face photos.

## 3. Train

Start with a smaller batch size on the MacBook Air:

```bash
python train.py --data data/faces --epochs 30 --batch-size 32 --device auto
```

For a quick smoke test:

```bash
python train.py --data data/faces --epochs 1 --batch-size 8 --device auto
```

If memory pressure or out-of-memory errors occur, stop with Ctrl+C and retry with `--batch-size 16` or `--batch-size 8`. `--device auto` chooses MPS when available, otherwise CPU.

Resume from the latest checkpoint:

```bash
python train.py --data data/faces --epochs 60 --batch-size 32 --device auto --resume
```

`--epochs 60` means train until epoch 60, not 60 additional epochs.

## 4. Run the UI

In another terminal, from this project folder:

```bash
source .venv/bin/activate
streamlit run app.py
```

The app loads `checkpoints/generator.pt`. Train at least one epoch first.

## Project files

- `models.py`: generator/discriminator architectures and initialization.
- `data.py`: recursive folder dataset and preprocessing.
- `train.py`: adversarial training, MPS/CPU device selection, checkpoints, image grids and CSV loss history.
- `app.py`: Streamlit generation UI, image downloads, loss chart and saved epoch preview.
- `checkpoints/latest.pt`: full training state for resuming.
- `checkpoints/generator.pt`: generator weights used by the app.
- `samples/`: fixed-noise preview grid for each epoch.
- `training_history.csv`: generator/discriminator losses per epoch.

## Notes and troubleshooting

- If the dataset has fewer images than the batch size, reduce `--batch-size`.
- `num_workers=0` is the default for simpler macOS behavior.
- Loss curves alone do not prove quality. Inspect sample grids and look for diversity, coherent facial structure, and mode collapse (many nearly identical outputs).
- GAN training can be unstable. Try learning rate `0.0001` or a smaller batch size if training diverges.
- 64×64 DCGAN outputs are low resolution by design. This is a learning/research prototype, not a production portrait generator.
- CelebA's published terms restrict use to non-commercial research and restrict redistribution; do not publish dataset images or derived data without confirming permission.
