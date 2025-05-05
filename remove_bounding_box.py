import os
import shutil

# Define paths
dataset_root = os.path.join(os.path.expanduser("~"), "Desktop", "computer_vision_dataset")
output_dir = os.path.join(dataset_root, "all_data")
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")

# Create modified directory structure
modified_dir = os.path.join(output_dir, "modified_labels")
os.makedirs(modified_dir, exist_ok=True)

# Initialize counters
processed_count = 0
modified_count = 0
removed_count = 0
skipped_count = 0

print(f"\nDataset root: {dataset_root}")
print(f"Images directory: {images_dir}")
print(f"Labels directory: {labels_dir}")

# Verify directories exist
if not os.path.exists(labels_dir):
    print(f"\nERROR: Labels directory not found at {labels_dir}")
    exit(1)

print(f"\nFound {len(os.listdir(labels_dir))} items in labels directory")

# Process files
print("\nStarting processing...")
for label_file in os.listdir(labels_dir):
    # File filtering
    if not (label_file.lower().endswith('.txt') or 
            os.path.splitext(label_file)[1].lower() in ['.txt', '.text', '.dat'] or
            '.' not in label_file):
        skipped_count += 1
        continue
        
    label_path = os.path.join(labels_dir, label_file)
    processed_count += 1
    
    try:
        # Process label file
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        classes = {line.split()[0] for line in lines if len(line.split()) > 0}
        
        if len(classes) > 1:
            print(f"Removing {label_file} (multiple classes: {', '.join(classes)})")
            removed_count += 1
        else:
            # For single-class files, keep only class ID
            new_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) > 0:
                    new_lines.append(parts[0])  # Keep only class ID
            
            # Write modified file
            modified_path = os.path.join(modified_dir, label_file)
            with open(modified_path, 'w') as f:
                f.write('\n'.join(new_lines))
            
            print(f"Modified {label_file} (kept class: {new_lines[0] if new_lines else 'empty'})")
            modified_count += 1
            
    except Exception as e:
        print(f"Error processing {label_file}: {str(e)}")
        skipped_count += 1

print("\nProcessing complete!")
print(f"Total files found: {len(os.listdir(labels_dir))}")
print(f"Files processed: {processed_count}")
print(f"Files modified (single class): {modified_count}")
print(f"Files removed (multiple classes): {removed_count}")
print(f"Files skipped: {skipped_count}")
print(f"Modified files are in: {modified_dir}")

# Verification
print("\nVerification:")
print(f"Original label files: {len(os.listdir(labels_dir))}")
print(f"Modified label files: {len(os.listdir(modified_dir))}")