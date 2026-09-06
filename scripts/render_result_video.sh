#!/bin/bash
# Renders the submitted result.mp4: 9 aimed penalty shots, seeds 0-8, zones cycling left/centre/right.
# These are the first 9 seeds of the reported evaluation, not selected takes.
set -e
cd "$(dirname "$0")/.."
. .venv/bin/activate
python -m duck_kick.evaluate --shots 1 --video runs/final/shootout9.mp4 --video-shots 9 \
  --video-label "Commanded zone shown top-left. Learned aim + learned celebration. MuJoCo simulation." \
  --json runs/final/shootout9.json
ffmpeg -v error -y -i runs/final/shootout9.mp4 -c:v libx264 -preset slow -crf 26 -pix_fmt yuv420p -movflags +faststart result.mp4
echo "result.mp4 written"
