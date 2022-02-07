!/bin/bash
ffmpeg -framerate 30 -i %04d.jpg -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -y -an flock_01.mp4

# to gif

ffmpeg -t 10 -i flock_01.mp4 -vf "fps=10,scale=-1:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 flock_01.gif