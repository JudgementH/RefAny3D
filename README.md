# RefAny3D: 3D Asset-Referenced Diffusion Models for Image Generation

<a href='https://judgementh.github.io/RefAny3D'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://huggingface.co/JudgementH/RefAny3D"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>

![teaser](assets/teaser.png)

## TODO List
- [x] Inference code and pretrained models.
- [ ] Training code.
- [ ] Training dataset.

## Quickstart

### Create environment
1. Clone the repository and create a conda environment: 
```
git clone https://github.com/JudgementH/RefAny3D.git
conda create -n r3d python=3.10
conda activate r3d
```
2. Install requirements
```
pip install -r requirements.txt
```


### Inference
```
python demo.py \
    --prompt "<text-prompt>" \
    --glb_path "<glb-file-path>" \
    --output_dir "<output-dir>"

```

Alternatively, we provide some example scripts:

```
bash scripts/demo_chair.sh

bash scripts/demo_traffic_cone.sh

```
