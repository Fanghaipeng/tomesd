import os
from pathlib import Path

def pad_filenames(directory):
    # Walk through all directories and files in the provided directory
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            # Check if the file is a PNG file
            if filename.endswith('.png'):
                old_path = os.path.join(subdir, filename)
                # Extract the file number from the filename
                file_number = filename.split('.')[0]
                # Pad the file number with zeros to make it 12 digits long
                new_filename = f"{file_number.zfill(12)}.jpg"
                new_path = os.path.join(subdir, new_filename)
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed {old_path} to {new_path}")

# Replace 'your_directory_path' with the path to your directory
pad_filenames('/data1/fanghaipeng/project/sora/tomesd/SD3/samples')