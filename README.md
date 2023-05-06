## :hammer_and_wrench: Install

**Note:**

If you have a CUDA environment, please make sure the environment variable `CUDA_HOME` is set. It will be compiled under CPU-only mode if no CUDA available.

**Installation**

Create a virtual environment and install the necessary dependencies.

```bash
pip install -r requirements.txt
```

Install Segment Anything

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Navigate to the GroundingDINO directory.

```bash
cd GroundingDINO
```

Install GroundingDINO using the following command:

```bash
pip install -e .
```

Create a new directory called "checkpoints" to store the latest model weights, which can be downloaded here:

Grounding DINO:

```bash
https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Segment Anything:

```bash
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```