#!/usr/bin/env bash
# Download or verify monocular depth model weights for vision_arm_control.
# Does not embed private Scholar paths or credentials.
set -euo pipefail

DEST="${FRANKA_MODEL_DIR:-${1:-./models}}"
mkdir -p "$DEST"

echo "Model directory: $DEST"
echo
echo "MonoDepth2 pretrained weights are distributed by Niantic under research terms."
echo "This script does NOT automatically rehost proprietary weights."
echo
echo "Options:"
echo "  1) Copy research weights you already have (HUMANS MOVE / local training):"
echo "       cp encoder.pth depth.pth \"\$DEST/\""
echo
echo "  2) Follow upstream MonoDepth2 instructions:"
echo "       https://github.com/nianticlabs/monodepth2#pretrained-models"
echo
echo "  3) On Purdue Scholar, place weights under:"
echo "       \$RCAC_SCRATCH/franka-teleop-data/models/"
echo "     then: export FRANKA_MODEL_DIR=\$RCAC_SCRATCH/franka-teleop-data/models"
echo

# Optional: verify SHA256 if checksums file is provided next to this script
CHECKSUMS="$(cd "$(dirname "$0")" && pwd)/model_checksums.sha256"
if [[ -f "$CHECKSUMS" ]]; then
  echo "Verifying checksums from $CHECKSUMS ..."
  (
    cd "$DEST"
    sha256sum -c "$CHECKSUMS"
  )
  echo "Checksum verification OK."
else
  echo "No scripts/model_checksums.sha256 found — skipping hash verification."
  echo "After placing weights, generate one with:"
  echo "  (cd \"\$DEST\" && sha256sum encoder.pth depth.pth > scripts/model_checksums.sha256)"
fi

missing=0
for f in encoder.pth depth.pth; do
  if [[ ! -f "$DEST/$f" ]]; then
    echo "MISSING: $DEST/$f"
    missing=1
  else
    ls -lh "$DEST/$f"
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo
  echo "ERROR: Required model files are missing in $DEST" >&2
  exit 1
fi

echo
echo "Models present. Export:"
echo "  export FRANKA_MODEL_DIR=\"$DEST\""
