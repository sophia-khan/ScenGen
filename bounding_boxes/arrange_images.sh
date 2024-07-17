#!/bin/bash

# Path to the folder containing the images
IMAGE_FOLDER="add_image_path"

# Create the destination folders
for i in {1..10}; do
    mkdir -p "$IMAGE_FOLDER/folder_$i"
done

# Get the list of all image files
image_files=($(find "$IMAGE_FOLDER" -maxdepth 1 -type f -name "*.png" ! -name "._*" ! -path "*__MACOSX*"))

# Shuffle the image files array
shuf_image_files=($(shuf -e "${image_files[@]}"))

# Distribute images to folders
for ((i = 0; i < ${#shuf_image_files[@]}; i++)); do
    folder_index=$((i % 10 + 1))
    mv "${shuf_image_files[$i]}" "$IMAGE_FOLDER/folder_$folder_index/"
done

echo "Images have been distributed into 10 folders."
