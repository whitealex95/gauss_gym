# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# rm -rf /etc/apt/sources.list.d/cuda.list

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
echo "Installing ${ENV_NAME} dependencies"
echo $SCRIPT_DIR
ENV_ROOT=$CONDA_ROOT/envs/${ENV_NAME}
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
    $CONDA_ROOT/bin/conda env create -f $SCRIPT_DIR/environment.yml -n ${ENV_NAME}
  fi

  $CONDA_ROOT/bin/conda run -n ${ENV_NAME} bash build/conda_env/install_conda_hooks.sh

  # Download Isaac Gym
  if [[ ! -d isaacgym ]]; then
    ISAACGYM_TAR=IsaacGym_Preview_4_Package.tar.gz
    if [[ ! -f $ISAACGYM_TAR ]]; then
      wget https://developer.nvidia.com/isaac-gym-preview-4 -O $ISAACGYM_TAR
    fi
    # NVIDIA sometimes returns an HTML page instead of the archive.
    if ! gzip -t $ISAACGYM_TAR 2>/dev/null; then
      echo "ERROR: $ISAACGYM_TAR is not a valid gzip archive. Download Isaac Gym Preview 4 manually from https://developer.nvidia.com/isaac-gym and place the tarball in $SCRIPT_DIR" >&2
      rm -f $ISAACGYM_TAR
      exit 1
    fi
    tar -xzf $ISAACGYM_TAR
  fi

  $CONDA_ROOT/bin/conda run -n ${ENV_NAME} uv pip install -e .

  source $CONDA_ROOT/bin/activate ${ENV_NAME}

  touch $SENTINEL_FILE
fi
