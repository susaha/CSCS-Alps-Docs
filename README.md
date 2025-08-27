# Building a Docker Container on Alps

This guide explains how to build and run a Docker container on the Alps system using **Podman**, **Enroot**, and **SLURM**. Follow the steps carefully to set up your environment and run workloads inside a container.

## STEP-1: Create a Project Folder
Navigate to your project location and create a folder to store your `Dockerfile` and the `.sqsh` file (SquashFS image) generated from the Docker image.

```bash
cd /capstor/store/cscs/sdsc/<project-name>
mkdir MyDocker && cd MyDocker
```

## STEP-2: Create a Dockerfile

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3
ENV DEBIAN_FRONTEND=noninteractive
# Install python venv and OpenCV dependencies, then clean up
RUN apt-get update && apt-get install -y \
    python3.10-venv \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6
```
## STEP-3: Configure Podman Storage
Now that we’ve set up the Dockerfile, we can pass it to Podman to build a container. Podman requires some storage configuration. Create the following file:
```bash
vim $HOME/.config/containers/storage.conf
```

Insert this content:

```ini
[storage]
  driver = "overlay"
  runroot = "/dev/shm/$USER/runroot"
  graphroot = "/dev/shm/$USER/root"
[storage.options.overlay]
  mount_program = "/usr/bin/fuse-overlayfs-1.13"
```

## STEP-4: Build and Convert the Container
To build a container with Podman, request a compute node shell from SLURM, pass the Dockerfile to Podman, and finally import the built container using Enroot.
### STEP-4.1: Request an Interactive Session

```bash
srun --partition debug --pty bash
```

### STEP-4.2: Build the Dockerfile

```bash
podman build -t ExampleDocker .
```

### STEP-4.3: Convert to SquashFS
Use Enroot to convert the Podman image into an Enroot-compatible SquashFS image:
```bash
enroot import -x mount -o ExampleDocker.sqsh podman://ExampleDocker
```

Exit the SLURM allocation:
```bash
exit
```

At this point, you should see both the Dockerfile and the new .sqsh file, an example below:
```bash
ls -l /capstor/store/cscs/sdsc/<project-name>/MyDocker
...
-rw-r-----+ 1 ssaha sd24 16440033280 Aug  5 18:28 ExampleDocker.sqsh
-rw-rw----+ 1 ssaha sd24        1446 Aug  5 18:25 Dockerfile
...
```

## STEP-5: Create an Environment Definition File (EDF)
We need to set up an EDF (Environment Definition File) to tell SLURM which container to use. Create the file:

```bash
vim ~/.edf/ExampleDocker.toml
```
Insert the following:

```toml
image = "/capstor/store/cscs/sdsc/<project-name>/MyDocker/ExampleDocker.sqsh"

mounts = ["/capstor", "/users"]

writable = true

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"

[env]
FI_CXI_DISABLE_HOST_REGISTER = "1"
FI_MR_CACHE_MONITOR = "userfaultfd"
NCCL_DEBUG = "INFO"

```
## STEP-6: Set Up Python Virtual Environment
### STEP-6.1: Request Interactive Session with EDF

```bash
srun --environment=ExampleDocker --container-workdir=$PWD --pty bash
```

### STEP-6.2: Create Python Virtual Environment
```bash
python -m venv --system-site-packages ./<ExampleEnv>
```

### STEP-6.3: Activate Environment and Install Libraries
```bash
source /capstor/store/cscs/sdsc/<project-name>/ExampleDocker/ExampleEnv/bin/activate
```

Install required packages:
```bash
python -m pip install \
    omegaconf==2.3.0 \
    opencv-python==4.9.0.80 \
    tensorboardX==2.6 \
    tensorboard==2.15.1 \
    h5py==3.14.0 \
    scikit-image==0.25.2

```

## Below is an example SLURM job script that uses the Docker environment (ExampleDocker.toml).

```bash
#!/bin/bash
#SBATCH --job-name=cad
#SBATCH --partition debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:15:00
##SBATCH --environment=taming-transformer-pytorch-250325
#SBATCH --account=sd24
#SBATCH --output=/capstor/scratch/cscs/ssaha/Experiments/CAD-August-8-2025/daint_exp_07-08-2025-001_debug/bs_0_ngpu_4_250808_1141_70540/slurm_out_%j.out  # Output log file

CONFIG=no-defined
CODE_PATH=/capstor/scratch/cscs/ssaha/Jobs/cad
CODE_TARGZ_FILE_NAME=250808_114151_cd6941f6.tar.gz

echo "*** CONFIG ***"
echo $CONFIG
echo "*** CODE_PATH ***"
echo $CODE_PATH
echo "*** CODE_FOLDER_NAME ***"
echo 250808_114151_cd6941f6
echo "*** CODE_TARGZ_FILE_NAME ***"
echo $CODE_TARGZ_FILE_NAME

# Path to your Docker environment config
TOML_PATH="/users/ssaha/.edf/ExampleDocker.toml"

# Distributed training setup
# export MASTER_ADDR=$(hostname)  # Activate if you use DistributedDataParallel instead of DataParallel in PyTorch
# export MASTER_PORT=29501        # Activate if you use DistributedDataParallel instead of DataParallel in PyTorch
export OMP_NUM_THREADS=4

# Setting the codebase path to PYTHONPATH
export PYTHONPATH=/capstor/scratch/cscs/ssaha/Code/250808_114151_cd6941f6/cad/:$PYTHONPATH
echo "*** PYTHONPATH ***"
echo $PYTHONPATH

srun --export=ALL,CONFIG="$CONFIG",CODE_PATH="$CODE_PATH",CODE_TARGZ_FILE_NAME="$CODE_TARGZ_FILE_NAME" \
     --environment=$TOML_PATH \
     -u -l \
     bash -c '

  # Extract codebase
  echo "mkdir -p /capstor/scratch/cscs/ssaha/Code/250808_114151_cd6941f6/cad"
  mkdir -p /capstor/scratch/cscs/ssaha/Code/250808_114151_cd6941f6/cad

  echo "tar -xzf $CODE_PATH/$CODE_TARGZ_FILE_NAME -C /capstor/scratch/cscs/ssaha/Code/250808_114151_cd6941f6/cad/ --strip-components=1"
  tar -xzf $CODE_PATH/$CODE_TARGZ_FILE_NAME -C /capstor/scratch/cscs/ssaha/Code/250808_114151_cd6941f6/cad/ --strip-components=1

  # Change directory to code folder
  echo "cd /capstor/scratch/cscs/ssaha/Code/250808_114151_cd6941f6/cad/"
  cd /capstor/scratch/cscs/ssaha/Code/250808_114151_cd6941f6/cad/

  echo "[Node $SLURM_PROCID] In container: $(hostname)"
  echo "[Node $SLURM_PROCID] PWD: $(pwd)"

  # ✅ Activate your virtual environment inside the container
  echo "source /capstor/store/cscs/sdsc/sd24/docker4cad/python_venv/cad/bin/activate"
  source /capstor/store/cscs/sdsc/sd24/docker4cad/python_venv/cad/bin/activate

  set -x

  echo ""
  echo "*** which python"
  which python

  # echo "*** which torchrun"
  # which torchrun

  # ✅ Optional sanity checks
  python -c "import torch; print(\"Torch version:\", torch.__version__)"
  python -c "import omegaconf; print(\"OmegaConf OK\")"
  python -c "import cv2; print(\"cv2 OK\")"

  # Numeric safety settings
  export NVIDIA_TF32_OVERRIDE=0                  # Disable TF32 in all CUDA libraries
  export TORCH_FLOAT32_MATMUL_PRECISION=highest  # Use highest precision for PyTorch GEMMs
  echo "*** Numeric Safety Settings ***"
  echo "NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"
  echo "TORCH_FLOAT32_MATMUL_PRECISION=$TORCH_FLOAT32_MATMUL_PRECISION"

  # Use this for Distributed Data Parallel
  # python -m torch.distributed.run --standalone --nproc-per-node=4 main.py $CONFIG

  # Use this for Data Parallel
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  python -m scripts_2_5d_3d/main_CAD.py
'

echo "✅ Finished at: $(date)"


```
### 📚 BibTeX Citation  

```bibtex
@misc{cscs_llm_inference_tutorial,
  title        = {LLM Inference Tutorial: Build a Modified NGC PyTorch Container},
  author       = {{Swiss National Supercomputing Centre (CSCS)}},
  year         = {2025},
  howpublished = {\url{https://docs.cscs.ch/tutorials/ml/llm-inference/#build-a-modified-ngc-pytorch-container}},
  note         = {Accessed: 2025-08-27}
}
