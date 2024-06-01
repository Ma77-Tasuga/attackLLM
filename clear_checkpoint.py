import os
import shutil


def clear_directory(path):
    if not os.path.exists(path):
        print(f"Path '{path}' does not exist.")
        return

    # Ensure the path is a directory
    if not os.path.isdir(path):
        print(f"Path '{path}' is not a directory.")
        return

    # Iterate over all the entries in the directory
    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)

        try:
            # If the entry is a directory, remove it and its contents
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
                print(f"Directory '{entry_path}' has been removed.")
            else:
                # If the entry is a file, remove it
                os.remove(entry_path)
                print(f"File '{entry_path}' has been removed.")
        except Exception as e:
            print(f"Failed to remove '{entry_path}'. Reason: {e}")


if __name__ == "__main__":
    # Replace 'your_directory_path' with the path to the directory you want to clear
    directory_to_clear = './check_point'
    clear_directory(directory_to_clear)
