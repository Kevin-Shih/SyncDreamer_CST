#!/bin/bash
GPU=$1
NAME=${3:-GSO}
scene_dir=("rendering/$NAME"/*)
out_dir="output/$2/$NAME"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p $out_dir
log_fp="output/$2/${NAME}_log.txt"
echo "GPU=$1, NAME=${3:-GSO}, out_dir="output/$2/$NAME"" > "$log_fp"
for entry in "${scene_dir[@]:0:10}"
do
    input="$entry/train/rgb/0.png"
    output="$out_dir/${entry##*/}/"
    transform="$entry/train/transform.json"

    echo $output
    echo $input
    python generate.py \
         --ckpt ckpt/syncdreamer-pretrain.ckpt \
         --input $input \
         --output $output \
         --transform_fp $transform \
         --sample_num 1 \
         --cfg_scale 2.0 \
         --crop_size -1 \
         --input_idx 0 >> "$log_fp"
    mv "$output/0_8views.png" "$output/demo.png"
    mv "$output/0.png" "$output/demo_colmap.png"
    cp "$output/demo.png" "$output/demo_0.png"
    cp "$output/demo_colmap.png" "$output/demo_0_16view.png"

    input_idx=${4:-1}
    input="$entry/train/rgb/$input_idx.png"
    echo $input
    python generate.py \
         --ckpt ckpt/syncdreamer-pretrain.ckpt \
         --input $input \
         --output $output \
         --transform_fp $transform \
         --sample_num 1 \
         --cfg_scale 2.0 \
         --crop_size -1 \
         --input_idx $input_idx >> "$log_fp"
    mv "$output/0_8views.png" "$output/demo_1.png"
    mv "$output/0.png" "$output/demo_1_16view.png"
done
export CUDA_VISIBLE_DEVICES=4,5,6,7