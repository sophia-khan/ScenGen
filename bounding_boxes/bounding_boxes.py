import zipfile
import os
import json
import torch
import time  
from tqdm import tqdm  
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, GroundingDinoForObjectDetection

# Ensure required packages are installed
os.system('pip install transformers torch torchvision Pillow matplotlib tqdm')

# Set up paths
zip_file_path = '/lab_src/notebooks/tile_dataset.zip'  # change path here
extract_folder_path = 'extracted_images'

# Create the extract folder if it doesn't exist
if not os.path.exists(extract_folder_path):
    os.makedirs(extract_folder_path)

# Unzip the images
print("Extracting images...")
start_time = time.time()  # Start timestamp
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder_path)
print(f"Extraction completed in {time.time() - start_time:.2f} seconds")

# Load the model and processor
print("Loading model and processor...")
start_time = time.time()  # Start timestamp
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
print(f"Model and processor loaded in {time.time() - start_time:.2f} seconds")

# Define object detection parameters
features = ["mountain", "river", "desert", "forest", "lake", "ocean", "grass", "sand", 
            "freeway", "reservoir", "railroad", "canal", "waterbody", "plant", "elevation", 
            "contour", "crossroad", "waves", "snow", "city", "town", "trail", "highway", "route",
            "region", "terrain", "tree", "road", "water", "topographic"]
confidence_threshold = 0.8
batch_size = 8  # Number of images to process in each batch
output_json_path = 'detection_results.json'

# Function to process a batch of images
def process_images(image_paths):
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

# Count total number of images to process
total_images = sum(len(files) for _, _, files in os.walk(extract_folder_path) if files)

# Estimate total runtime based on the number of images
estimated_runtime_per_image = 10  
estimated_total_runtime = total_images * estimated_runtime_per_image

print(f"Estimated total runtime: {estimated_total_runtime} seconds ({estimated_total_runtime / 60:.2f} minutes)")

# Process all images in batches and save results in increments
print("Processing images...")
start_time = time.time()  # Start timestamp
output_data = {}

# Load previously saved data if it exists
if os.path.exists(output_json_path):
    with open(output_json_path, 'r') as json_file:
        output_data = json.load(json_file)

all_image_paths = []
for root, _, files in os.walk(extract_folder_path):
    for file in files:
        if file.endswith('.png') and not file.startswith('._') and '__MACOSX' not in root:
            all_image_paths.append(os.path.join(root, file))

for i in tqdm(range(0, len(all_image_paths), batch_size), desc="Processing Batches", unit="batch"):
    batch_image_paths = all_image_paths[i:i + batch_size]
    batch_results = process_images(batch_image_paths)
    if batch_results:
        output_data.update(batch_results)
        # Save results in increments
        with open(output_json_path, 'w') as json_file:
            json.dump(output_data, json_file, indent=4)
    print(f"Processed batch {i // batch_size + 1} of {len(all_image_paths) // batch_size + 1}")

print(f"Processing completed in {time.time() - start_time:.2f} seconds")

# Load JSON data and print
print("Loading JSON data...")
start_time = time.time()  # Start timestamp
with open(output_json_path, 'r') as json_file:
    data = json.load(json_file)
print(f"JSON data loaded in {time.time() - start_time:.2f} seconds")

# Print loaded data
print(json.dumps(data, indent=4))  
