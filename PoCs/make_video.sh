!/bin/bash

echo -e "enter flock number e.g. <flock_01>"

read flock_num

ffmpeg -framerate 30 -i %04d.jpg -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -y -an "${flock_num}.mp4"

# to gif

ffmpeg  -i "${flock_num}.mp4" -vf "fps=30,scale=-1:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 "${flock_num}.gif"

cp "${flock_num}.gif" "${HOME}/Documents/ML-Control-Rob/Reachability/LargeBRAT/BRATVisualization"