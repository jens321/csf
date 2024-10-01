import os
import fnmatch

def remove_files(folder):
    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(folder):
        for filename in files:
            # Check if the file matches the pattern
            if fnmatch.fnmatch(filename, "itr*.pkl"):
                # Construct full file path
                file_path = os.path.join(root, filename)
                try:
                    # Remove the file
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

if __name__ == "__main__":
    # Define the folder path, change to your target folder
    folder_path = "anonymous-il-scale/metra-with-avalon/exp"
    
    # Call the function to remove files
    remove_files(folder_path)