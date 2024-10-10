import os
import subprocess

# Set the size limit for large files (in bytes)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
FILE_LIMIT = 1000  # Maximum number of files allowed in a folder to include

def is_large_file(file_path):
    """
    Check if the file is larger than the defined MAX_FILE_SIZE.
    
    Parameters:
    file_path (str): The path to the file.
    
    Returns:
    bool: True if the file is large, False otherwise.
    """
    return os.path.getsize(file_path) > MAX_FILE_SIZE

def count_files_in_directory(directory_path):
    """
    Count the number of files in a directory.
    
    Parameters:
    directory_path (str): The path to the directory.
    
    Returns:
    int: The number of files in the directory.
    """
    file_count = sum([len(files) for _, _, files in os.walk(directory_path)])
    return file_count

cnt_files = 0 
def auto_add_to_git_root_only(root_folder):
    global cnt_files 
    """
    Add files from the root folder to the Git working tree, skipping large files
    and folders with more than FILE_LIMIT files.
    
    Parameters:
    root_folder (str): The path to the root folder where Git is initialized.
    """
    try:
        # List all files and folders in the root folder
        for item in os.listdir(root_folder):
            item_path = os.path.join(root_folder, item)

            # Check if it's a file (not a directory)
            if os.path.isfile(item_path):
                # Skip large files
                if is_large_file(item_path):
                    print(f"---> Skipped large file: {item_path}")
                else:
                    # Add the file to the Git staging area
                    subprocess.run(["git", "add", item_path], check=True)
                    # print(f"Added '{item_path}' to the Git staging area.")
                    cnt_files += 1 

            # If it's a directory, check the file count and possibly skip it
            elif os.path.isdir(item_path):
                # Count the number of files in the directory
                file_count = count_files_in_directory(item_path)
                if file_count > FILE_LIMIT:
                    print(f"---> Skipped folder '{item_path}' (contains {file_count} files).")
                else:
                    # Add the folder if it has fewer than FILE_LIMIT files
                    auto_add_to_git_root_only(item_path) 
                    print(f"Number of added files: {cnt_files}")

        # Optionally show status after adding the files
        # subprocess.run(["git", "status"], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Failed to add files to Git: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Specify the root folder (where the git repository is initialized)
    root_folder = input("Enter the root folder (add) >")

    # Call the function to add files
    auto_add_to_git_root_only(root_folder)
