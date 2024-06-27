import os
import sys
import json
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# Function to clear GPU memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# Function to extract landmark entities from a caption using keyword matching
def extract_landmarks(caption):
    landmarks_list = ['mountain', 'river', 'lake', 'ocean', 'desert', 'forest', 'tree', 'trail', 'city', 'road']
    landmarks = [landmark for landmark in landmarks_list if landmark in caption.lower()]
    return landmarks

def generate_captions(image_dir, output_dir):
    # Clear GPU memory before loading the model
    clear_gpu_memory()

    # Setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Load pretrained/finetuned BLIP2 captioning model
    try:
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
        )
    except RuntimeError as e:
        print(f"Failed to load model on GPU: {e}. Switching to CPU.")
        device = torch.device("cpu")
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Initialize a list to store data for each tile
    tiles_data = []

    # Iterate through each image file
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            raw_image = Image.open(img_path).convert('RGB')
            raw_image = raw_image.resize((raw_image.width // 2, raw_image.height // 2))
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            # Generate caption using beam search
            caption = model.generate({"image": image})
            captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3)

            # Save the image and captions
            base_name = os.path.basename(img_path)
            raw_image.save(os.path.join(output_dir, base_name))

            with open(os.path.join(output_dir, base_name + '.txt'), 'w') as f:
                f.write("Caption using beam search:\n")
                f.write(caption[0] + "\n\n")
                f.write("Captions using nucleus sampling:\n")
                for c in captions:
                    f.write(c + "\n")

            # Extract landmarks from caption
            landmarks = extract_landmarks(caption[0])

            # Prepare data for the current tile
            tile_data = {
                'tile_name': base_name,
                'caption': caption[0],
                'landmarks': landmarks
            }

            # Append the current tile's data to the list
            tiles_data.append(tile_data)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Save tiles_data to a JSON file
    output_json = os.path.join(output_dir, "tiles_data.json")
    with open(output_json, 'w') as f:
        json.dump(tiles_data, f, indent=4)

    print(f"Processing completed. Results saved to: {output_json}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_captions.py <image_folder> <output_folder>")
        sys.exit(1)

    image_folder = sys.argv[1]
    output_folder = sys.argv[2]
    if not os.path.isdir(image_folder):
        print(f"The provided path '{image_folder}' is not a directory.")
        sys.exit(1)

    generate_captions(image_folder, output_folder)
