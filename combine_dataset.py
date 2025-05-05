import os
import shutil
from mimetypes import guess_type

# Configuration
dataset_root = os.path.join(os.path.expanduser("~"), "Desktop", "computer_vision_dataset")
output_dir = os.path.join(dataset_root, "all_data")
datasets = [
    "abdulaziz",
    "dania",
    "Feras",
    "Feras2",
    "My First Project.v1i.yolov8"
]

# Supported extensions (case-insensitive)
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

def clean_filename(filename):
    """Clean complex filenames by removing random string portions"""
    if '.rf.' in filename:
        return filename.split('.rf.')[0] + os.path.splitext(filename)[1]
    return filename

def is_valid_image(filepath):
    """Check if file is actually an image using magic numbers"""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(32)
            if header.startswith(b'\xff\xd8\xff'):  # JPEG
                return True
            elif header.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                return True
            elif header.startswith(b'BM'):  # BMP
                return True
            elif header.startswith(b'RIFF') and header[8:12] == b'WEBP':  # WEBP
                return True
        return False
    except:
        return False

def combine_datasets():
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    total_images = 0
    total_labels = 0
    invalid_files = 0
    
    print(f"Base dataset root: {dataset_root}")

    for dataset in datasets:
        dataset_path = os.path.join(dataset_root, dataset)
        print(f"\nProcessing dataset: {dataset} at {dataset_path}")
        
        if not os.path.exists(dataset_path):
            print(f"  Dataset directory not found!")
            continue
            
        for split in ['train', 'test', 'valid']:
            split_path = os.path.join(dataset_path, split)
            img_dir = os.path.join(split_path, 'images')
            lbl_dir = os.path.join(split_path, 'labels')
            
            print(f"  Checking {split} at {img_dir}")
            
            if not os.path.exists(img_dir):
                print(f"  {split}: images directory not found - skipping")
                continue
            
            # Get all files that appear to be images
            potential_files = [f for f in os.listdir(img_dir) 
                             if os.path.isfile(os.path.join(img_dir, f))]
            
            valid_images = []
            for f in potential_files:
                filepath = os.path.join(img_dir, f)
                
                # Check both extension and actual file content
                if (f.lower().endswith(IMAGE_EXTS) and is_valid_image(filepath)):
                    valid_images.append(f)
                else:
                    print(f"      Invalid image: {f}")
                    invalid_files += 1
            
            print(f"  {split}: found {len(valid_images)} valid images")
            
            for img_file in valid_images:
                # Clean both image and label names
                clean_img = clean_filename(img_file)
                output_prefix = f"{dataset}_{split}_"
                
                # Copy image
                src_img = os.path.join(img_dir, img_file)
                dst_img = os.path.join(output_dir, "images", output_prefix + clean_img)
                shutil.copy2(src_img, dst_img)
                total_images += 1
                
                # Find and copy corresponding label
                original_label_name = os.path.splitext(img_file)[0] + '.txt'
                src_label = os.path.join(lbl_dir, original_label_name)
                
                if os.path.exists(src_label):
                    clean_label = clean_filename(original_label_name)
                    dst_label = os.path.join(output_dir, "labels", output_prefix + clean_label)
                    shutil.copy2(src_label, dst_label)
                    total_labels += 1
                else:
                    print(f"      Warning: Missing label for {img_file}")

    print("\nCombination complete!")
    print(f"Total valid images copied: {total_images}")
    print(f"Total labels copied: {total_labels}")
    print(f"Invalid files skipped: {invalid_files}")
    print(f"Output directory: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    combine_datasets()