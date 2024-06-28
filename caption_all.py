import argparse
import logging
import json
import os
import time

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess




# Function to clear GPU memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()



def main():

    parser = argparse.ArgumentParser(description="Process a folder of images and caption.")
    parser.add_argument('--img_folder_path', type=str, required=True, help='Path to the folder containing images.')
    parser.add_argument('--caption_folder_path', type=str, required=True, help='Path to the folder to write captions to.')
    args = parser.parse_args()

    image_dir = args.img_folder_path
    output_dir  = args.caption_folder_path
    # Clear GPU memory before loading the model
    clear_gpu_memory()

    # Setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    print(f"device is: {device}")

    # Load pretrained/finetuned BLIP2 captioning model
    # Use a smaller model variant if available
    try:
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
        )
        print("model loaded")
    except RuntimeError as e:
        print(f"Failed to load model on GPU: {e}. Switching to CPU.")
        device = torch.device("cpu")
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
        )

    print(vis_processors.keys())

    #List all image files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    results_dict = {}

    #Iterate through each image file
    start = time.time()
    for idx,img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        try:
            raw_image = Image.open(img_path).convert('RGB')

            # Downscale the image to reduce memory usage
            raw_image = raw_image.resize((raw_image.width // 2, raw_image.height // 2))
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            #Generate caption using beam search
            caption = model.generate({"image": image})
            beam_caption = f"Generated caption for {img_path} using beam search:"+ caption
            print(f"Generated caption for {img_path} using beam search:", caption)


            # Generate multiple captions using nucleus sampling
            captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3)
            print(f"Generated captions for {img_path} using nucleus sampling:", captions)
            nucleus_captions = f"Generated captions for {img_path} using nucleus sampling: " + captions

            results_dict[img_file] = (beam_caption,nucleus_captions)
            # Save the image and captions
            base_name = os.path.basename(img_path)
            raw_image.save(os.path.join(output_dir, base_name))

            with open(os.path.join(output_dir, base_name + '.txt'), 'w') as f:
                f.write("Caption using beam search:\n")
                f.write(caption[0] + "\n\n")
                f.write("Captions using nucleus sampling:\n")
                for c in captions:
                    f.write(c + "\n")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
        
        
        if idx %500:
            print(f"saving partial results up to image # {idx}")
            json_outpath = "json_results_" + str(idx) + ".json"
            with open(os.path.join(output_dir,json_outpath), 'w') as file:
                json.dump(results_dict,file)
            
            end = time.time()
            print(f"Elapsed time since start: { end - start} seconds ")


    print("Processing completed. Results saved to:", output_dir)
if __name__ == "__main__":
	main()
