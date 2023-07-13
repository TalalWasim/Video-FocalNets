
## Installation Instructions

- Clone this repo:
```bash
git clone https://github.com/TalalWasim/Video-FocalNets
cd Video-FocalNets
```

- Create a conda virtual environment and activate it:

```bash
conda create -n focal python=3.8 -y
conda activate focal
```

- Install `PyTorch==1.11.0` and `torchvision==0.12.0` with `CUDA==11.3`:

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

- Install `timm==0.5.4`:

```bash
pip install timm==0.5.4
```

- Install `Apex`:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

- Install other requirements:

```bash
pip install decord==0.6.0 einops==0.4.1 imgaug==0.4.0 numpy==1.22.3 pandas==1.4.2 Pillow==9.0.1 PyYAML==6.0 termcolor==2.3.0 thop yacs
```
