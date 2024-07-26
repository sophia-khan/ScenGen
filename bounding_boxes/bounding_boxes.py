import os
import json
import sys

import torch
import time
import argparse
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, GroundingDinoForObjectDetection

def ensure_packages_installed():
    os.system('pip install transformers torch torchvision Pillow matplotlib tqdm')

def load_model_and_processor():
    print("Loading model and processor...")
    start_time = time.time()  # Start timestamp
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
    print(f"Model and processor loaded in {time.time() - start_time:.2f} seconds")
    return processor, model

def process_images(image_paths, processor, model, features, confidence_threshold):
    images = []
    valid_image_paths = []
    
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            valid_image_paths.append(image_path)
        except UnidentifiedImageError:
            print(f"Unidentified image error for file: {image_path}")
            continue
    
    if not images:
        return None
    
    batch_results = {}
    
    for feature in features:
        text = [f"a {feature}." for _ in range(len(images))]
        inputs = processor(images=images, text=text, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        
        # Post-process the outputs
        target_sizes = torch.tensor([image.size[::-1] for image in images])
        results = processor.image_processor.post_process_object_detection(
            outputs, threshold=0.35, target_sizes=target_sizes
        )
        
        for idx, result in enumerate(results):
            filtered_results = {
                "boxes": [],
                "scores": [],
                "labels": []
            }
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                if score.item() >= confidence_threshold:
                    filtered_results["boxes"].append([round(i, 2) for i in box.tolist()])
                    filtered_results["scores"].append(round(score.item(), 3))
                    filtered_results["labels"].append(feature)
            
            if filtered_results["boxes"]:
                if valid_image_paths[idx] not in batch_results:
                    batch_results[valid_image_paths[idx]] = []
                batch_results[valid_image_paths[idx]].append(filtered_results)
    
    return batch_results 


def main():
    parser = argparse.ArgumentParser(description="Process images for object detection")
    parser.add_argument('--image_folder_path', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--confidence_threshold', type=float, default=0.8, help='Confidence threshold for object detection')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of images to process in each batch')
    parser.add_argument('--output_json_path', type=str, default='detection_results.json', help='Path to save the detection results JSON file')
    parser.add_argument('--resume_json_path', type=str, default=None, help='Path to json containing end_idx')
    parser.add_argument('--slurm_task_id', type=int, required=True, help='SLURM array task ID')

    args = parser.parse_args()
    
    ensure_packages_installed()

    #extract slurm task array id
    slurm_array_task_id = args.slurm_task_id
    print(f"slurm_array_task_id: {slurm_array_task_id}")
    
    end_idx = args.resume_json_path
    if end_idx is not None:
        with open(args.resume_json_path, 'r') as file:
            data = json.load(file)
        for item in data:
            if item["SLURM_ARRAY_TASK_ID"] == slurm_array_task_id:
                end_idx = item.get('end_idx')
            
    else:

        end_idx = 0

    
    
    # Load the model and processor
    processor, model = load_model_and_processor()
    
    # Define object detection parameters
    features = ["mountain", "river", "desert", "forest", "lake", "ocean", "grass", "sand", 
                "freeway", "reservoir", "railroad", "canal", "waterbody", "plant", "elevation", 
                "contour", "crossroad", "waves", "snow", "city", "town", "trail", "highway", "route",
                "region", "terrain", "tree", "road", "water", "topographic"]
    
    # Process all images in batches and save results in increments
    print("Processing images...")
    start_time = time.time()  # Start timestamp
    output_data = {}
    
    # Load previously saved data if it exists
    #if os.path.exists(args.output_json_path):
        #with open(args.output_json_path, 'r') as json_file:
            #output_data = json.load(json_file)
    
    all_image_paths = []
    for root, _, files in os.walk(args.image_folder_path):
        for file in files:
            if file.endswith('.png') and not file.startswith('._') and '__MACOSX' not in root:
                all_image_paths.append(os.path.join(root, file))
    
    for i in tqdm(range(end_idx, len(all_image_paths), args.batch_size), desc="Processing Batches", unit="batch"):
        batch_image_paths = all_image_paths[i:i + args.batch_size]
        batch_results = process_images(batch_image_paths, processor, model, features, args.confidence_threshold)
        if batch_results:
            output_data.update(batch_results)
            # Save results every 20 batches
            if (i // args.batch_size + 1) % 5 == 0:
                start_idx = i
                end_idx = min(i + 20 * args.batch_size, len(all_image_paths)) - 1
                json_file_name = f"results_json_{start_idx}_{end_idx}.json"
                json_path = os.path.join(args.output_json_path, json_file_name)
                with open(json_path, 'w') as json_file:
                    json.dump(output_data, json_file, indent=4)
                print(f"Saved results to {json_path}")
    
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    
    # Load JSON data and print
    print("Loading JSON data...")
    start_time = time.time()  # Start timestamp
    json_path_final = os.path.join(args.output_json_path, "final.json")
    with open(json_path_final, 'r') as json_file:
        data = json.load(output_data,json_file)
    print(f"JSON data loaded in {time.time() - start_time:.2f} seconds")
    
    # Print loaded data
    print(json.dumps(data, indent=4))

if __name__ == "__main__":
    main()
