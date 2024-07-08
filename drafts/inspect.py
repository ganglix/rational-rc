import os

def find_non_utf8_files(directory):
    non_utf8_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        f.read()
                except UnicodeDecodeError:
                    non_utf8_files.append(file_path)
    return non_utf8_files

if __name__ == "__main__":
    project_directory = "./src/rational_rc"  # Adjust the path to your project directory
    non_utf8_files = find_non_utf8_files(project_directory)
    if non_utf8_files:
        print("Non-UTF-8 encoded files found:")
        for file in non_utf8_files:
            print(file)
    else:
        print("All files are UTF-8 encoded.")
