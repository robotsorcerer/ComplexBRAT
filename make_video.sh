!/bin/bash
ffmpeg -framerate 20 -i murmurations-%04d.jpg -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -y -an out.mp4

