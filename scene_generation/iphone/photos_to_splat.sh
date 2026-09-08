# Train a Gaussian splat from plain photos (no LiDAR, no poses), e.g. the
# "Images" zip that Polycam lets you download for a photo-mode capture.
#
#   source ~/.ns_deps/miniconda3/bin/activate ns
#   bash scene_generation/iphone/photos_to_splat.sh <IMAGES_DIR> <OUT_DIR>
#
# COLMAP is expected in its own conda env at ~/.ns_deps/miniconda3/envs/colmap
# (conda create -n colmap -c conda-forge "colmap=3.11.*=cuda*"). COLMAP 3.12+
# renamed its CLI options and nerfstudio's wrapper does not know the new names.
# Outputs: <OUT_DIR>/transforms.json, <OUT_DIR>/splatfacto/ (checkpoint, config.yml, splat.ply)

set -e

IMAGES_DIR="$1"
OUT_DIR="$2"
if [[ -z "$IMAGES_DIR" || -z "$OUT_DIR" ]]; then
  echo "usage: $0 <IMAGES_DIR> <OUT_DIR>" >&2
  exit 1
fi
export PATH="$HOME/.ns_deps/miniconda3/envs/colmap/bin:$PATH"
command -v colmap >/dev/null || { echo "colmap not found" >&2; exit 1; }

mkdir -p "$OUT_DIR"
if [[ ! -f "$OUT_DIR/transforms.json" ]]; then
  ns-process-data images --data "$IMAGES_DIR" --output-dir "$OUT_DIR" \
    --matching-method exhaustive --sfm-tool colmap --gpu
fi

ns-train splatfacto \
    --pipeline.model.use-scale-regularization=True \
    --pipeline.model.rasterize-mode=antialiased \
    --pipeline.model.camera-optimizer.mode=SO3xR3 \
    --pipeline.model.use-bilateral-grid=True \
    --pipeline.model.strategy=mcmc \
    --experiment-name '' \
    --timestamp '' \
    --output-dir "$OUT_DIR" \
    --viewer.quit-on-train-completion True \
    nerfstudio-data \
    --data "$OUT_DIR" \
    --train-split-fraction=1.0

ns-export gaussian-splat --load-config "$OUT_DIR/splatfacto/config.yml" --output-dir "$OUT_DIR/splatfacto"
echo "SPLAT PIPELINE DONE"
