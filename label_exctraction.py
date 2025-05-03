import os
import re

# Directory containing the images
image_dir = r"all_data\all_data\images"  # Replace with your directory path if needed

# Loop through all files in the directory
# Dictionary to track how many times each speed appears
speed_counter = {}

# Loop through all files in the directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith((".jpg", "_jpg", ".jpeg")):  # Check for image files
        # Extract the speed value using regex
        match = re.search(r"speed_(\d+)", filename)
        if match:
            speed = match.group(1)
            
            # Update counter for this speed
            if speed in speed_counter:
                speed_counter[speed] += 1
            else:
                speed_counter[speed] = 1
            
            # New filename: speed_counter adds uniqueness (e.g., 60_1.jpg, 60_2.jpg)
            new_filename = f"{speed}_{speed_counter[speed]}.jpg"
            
            # Rename the file
            old_path = os.path.join(image_dir, filename)
            new_path = os.path.join(image_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")