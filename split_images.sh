#heck if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 source_folder n"
    exit 1
fi

source_folder=$1
n=$2

# Create destination folders
for ((i=1; i<=n; i++))
do
    mkdir -p "${source_folder}/new_folder_${i}"
done

# Get the list of image files
images=($(find "$source_folder" -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.gif" \)))

# Calculate the number of images and the number of images per folder
total_images=${#images[@]}
images_per_folder=$(( (total_images + n - 1) / n ))  # ceil(total_images / n)

# Distribute images into folders
for ((i=0; i<total_images; i++))
do
    folder_index=$(( (i / images_per_folder) + 1 ))
    mv "${images[$i]}" "${source_folder}/new_folder_${folder_index}/"
done

echo "Images have been split into $n folders."
