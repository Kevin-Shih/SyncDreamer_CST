#!/bin/bash
GPU=$1
export CUDA_VISIBLE_DEVICES="$GPU"

NAME=${3:-GSO}
scene_dir=("rendering/$NAME"/*)
out_dir="output/$2/$NAME"
mkdir -p $out_dir
log_fp="output/$2/${NAME}_neus_log.txt"
echo "GPU=$1, NAME=${3:-GSO}, out_dir="output/$2/$NAME"" > "$log_fp"
# TOTAL_LEN=${#scene_dir[@]}
# LEN_HALF=$(( TOTAL_LEN / 2 ))
for entry in "${scene_dir[@]:0}"
do
    transform="$entry/train/transform.json"
    output="$out_dir/${entry##*/}/"
    echo $transform
    echo $output
    python train_renderer.py -g '0,' -l $output -i 0 -t $transform
    python train_renderer.py -g '0,' -l $output -i 1 -t $transform
    cp $output/render_0.png $output/render.png
done
export CUDA_VISIBLE_DEVICES=4,5,6,7