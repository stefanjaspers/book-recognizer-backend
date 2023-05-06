## :hammer_and_wrench: Install

**Note:**

If you have a CUDA environment, please make sure the environment variable `CUDA_HOME` is set. It will be compiled under CPU-only mode if no CUDA available.

**Installation**

Create a virtual environment and install the necessary dependencies.

```pip install -r requirements.txt```

Navigate to the GroundingDINO directory.

```cd GroundingDINO```

Install GroundingDINO using the following command:

```pip install -e .```

Create a new directory called "checkpoints" to store the latest model weight, which can be downloaded here:

```https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth```