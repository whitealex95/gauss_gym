# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm -f /etc/apt/sources.list.d/cuda.list 2>/dev/null || true

echo "Installing NS dependencies"
echo $SCRIPT_DIR

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/ns
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_dev

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi


  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda env create -f $SCRIPT_DIR/environment.yml
  fi

  $CONDA_ROOT/bin/conda run -n ns bash $SCRIPT_DIR/conda_env/install_conda_hooks.sh

  # Fix PyTorch Intel JIT symbol issue
  $CONDA_ROOT/bin/conda run -n ns pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
  $CONDA_ROOT/bin/conda run -n ns pip install ml_collections open3d tqdm opencv-python Pillow

  # tiny-cuda-nn uses a legacy setup.py that imports pkg_resources, which
  # setuptools>=81 removed. Build it against an older setuptools without isolation.
  if [[ -z "${TCNN_CUDA_ARCHITECTURES:-}" ]]; then
    export TCNN_CUDA_ARCHITECTURES=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  fi
  $CONDA_ROOT/bin/conda run -n ns pip install "setuptools<80" ninja
  $CONDA_ROOT/bin/conda run -n ns pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

  # Install Nerfstudio.
  NS_PATH=$WORKSPACE_DIR/nerfstudio
  rm -rf $NS_PATH
  git clone https://github.com/escontra/nerfstudio-gauss-gym.git $NS_PATH
  # git clone https://github.com/nerfstudio-project/nerfstudio.git $NS_PATH
  # $CONDA_ROOT/bin/conda run -n ns pip install --upgrade pip setuptools
  $CONDA_ROOT/bin/conda run -n ns pip install -e $NS_PATH/.
  # $CONDA_ROOT/bin/conda run -n ns pip install nerfstudio

  # Extras for the phone-scan -> MuJoCo/viser tools (scene_generation/iphone/).
  $CONDA_ROOT/bin/conda run -n ns pip install mujoco coacd "plyfile<1.1"

  # Pins, applied last so nothing above can undo them:
  # - torch 2.1.2 was built against numpy 1.x.
  # - nerfstudio's fast image loader uses a private Pillow encoder call that Pillow 11+ changed.
  $CONDA_ROOT/bin/conda run -n ns pip install "numpy==1.26.4" "pillow<11"

  # COLMAP for photos without poses. Kept in its own env so its solver cannot
  # touch the torch/CUDA packages in `ns`. 3.12+ renamed the CLI options that
  # nerfstudio's wrapper passes, so stay on 3.11.
  if [[ ! -d $CONDA_ROOT/envs/colmap ]]; then
    $CONDA_ROOT/bin/conda create -y -n colmap -c conda-forge "colmap=3.11.*=cuda*"
  fi

  # Call this conda directly: inside `conda run` a bare `conda` may resolve to
  # another conda install on the user's PATH.
  $CONDA_ROOT/bin/conda install -y -n ns -c conda-forge s5cmd

  source $CONDA_ROOT/bin/activate ns
  
  touch $SENTINEL_FILE
fi
