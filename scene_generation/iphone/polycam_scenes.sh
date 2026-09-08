# Steps to collect PolyCam data:
# 1. Use PolyCam in LIDAR / "Space" mode to collect data.
# 2. Process the data in the PolyCam app.
# 3. Export "Raw data" and "GLTF" (glb) from the app. Place the unzipped contents
#    in the <POLYCAM_PATH>. Rename the .glb file to "raw.glb".
# 4. Run this script to convert the data and train a splat model:
#    bash scene_generation/iphone/polycam_scenes.sh <POLYCAM_PATH>
# 5. Generate meshes with:
#    python scene_generation/generate_mesh_slices.py --config=scene_generation/configs/polycam.py --config.load_dir=<POLYCAM_PATH>

set -e

POLYCAM_PATH="$1"
if [[ -z "$POLYCAM_PATH" ]]; then
  echo "usage: $0 <POLYCAM_PATH or raw_data.zip>" >&2
  exit 1
fi

if [[ -f "$POLYCAM_PATH" && "$POLYCAM_PATH" == *.zip ]]; then
  zip_path="$POLYCAM_PATH"
  POLYCAM_PATH="${zip_path%.zip}"
  mkdir -p "$POLYCAM_PATH"
  unzip -q -o "$zip_path" -d "$POLYCAM_PATH"
  # Some exports wrap everything in a single top-level folder.
  if [[ ! -e "$POLYCAM_PATH/keyframes" ]]; then
    inner=$(find "$POLYCAM_PATH" -mindepth 2 -maxdepth 2 -type d -name keyframes | head -1)
    if [[ -n "$inner" ]]; then
      mv "$(dirname "$inner")"/* "$POLYCAM_PATH"/ && rmdir "$(dirname "$inner")"
    fi
  fi
  if [[ ! -e "$POLYCAM_PATH/raw.glb" ]]; then
    glb=$(ls "$POLYCAM_PATH"/*.glb 2>/dev/null | head -1)
    [[ -n "$glb" ]] && mv "$glb" "$POLYCAM_PATH/raw.glb"
  fi
  echo "Unzipped $zip_path into $POLYCAM_PATH"
fi

missing=0
for f in raw.glb mesh_info.json keyframes/depth; do
  if [[ ! -e "$POLYCAM_PATH/$f" ]]; then
    echo "ERROR: missing $POLYCAM_PATH/$f" >&2
    missing=1
  fi
done
if [[ ! -d "$POLYCAM_PATH/keyframes/corrected_images" && ! -d "$POLYCAM_PATH/keyframes/images" ]]; then
  echo "ERROR: missing $POLYCAM_PATH/keyframes/corrected_images (or keyframes/images)" >&2
  missing=1
fi
if [[ $missing -ne 0 ]]; then
  echo "The GLTF export alone is not enough. Export \"Raw Data\" from Polycam (may need Polycam Pro), unzip it into $POLYCAM_PATH, and put the .glb next to it as raw.glb." >&2
  exit 1
fi

ns-process-data polycam --use-depth --data $POLYCAM_PATH --output-dir $POLYCAM_PATH

ns-train splatfacto \
    --pipeline.model.use-scale-regularization=True \
    --pipeline.model.output-depth-during-training=False \
    --pipeline.model.rasterize-mode=antialiased \
    --pipeline.model.camera-optimizer.mode=SO3xR3 \
    --pipeline.model.use-bilateral-grid=True \
    --pipeline.model.strategy=mcmc \
    --experiment-name '' \
    --timestamp '' \
    --output-dir $POLYCAM_PATH \
    --viewer.quit-on-train-completion True \
    nerfstudio-data \
    --data $POLYCAM_PATH \
    --train-split-fraction=1.0 \
    --depth-unit-scale-factor=0.001

ns-export gaussian-splat --load-config $POLYCAM_PATH/splatfacto/config.yml --output-dir $POLYCAM_PATH/splatfacto
