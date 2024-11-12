import os

# Path to the directory
directory = "/data1/fanghaipeng/project/sora/tomesd/SD3/samples"

# Dictionary to hold directory and their file counts
directory_file_counts = {}

# Walking through the directory and subdirectories
for subdir, dirs, files in os.walk(directory):
    directory_file_counts[subdir] = len(files)

# Print the counts for each directory
for dir_path, count in directory_file_counts.items():
    print(f"Directory: {dir_path} has {count} files")