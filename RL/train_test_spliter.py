import os
import shutil
import random

def move_random_files(training_folder, test_folder, nb_test_files):
    # Get all file paths from the training folder
    all_files = []
    for root, _, files in os.walk(training_folder):
        for file in files:
            all_files.append(os.path.join(root, file))

    # Randomly select files to move
    selected_files = random.sample(all_files, min(nb_test_files, len(all_files)))

    for file_path in selected_files:
        # Compute new file path in the test folder, preserving the folder structure
        relative_path = os.path.relpath(file_path, training_folder)
        new_file_path = os.path.join(test_folder, relative_path)
        
        # Create directories in the test folder if they don't exist
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        
        # Move the file
        shutil.move(file_path, new_file_path)
        print(f"Moved: {file_path} to {new_file_path}")

# Example usage
training_folder = '/root/ssh-rlkex/Generated_Graphs'
test_folder = '/root/ssh-rlkex/Test_Graphs'
nb_test_files = 200  # Number of test files to move

move_random_files(training_folder, test_folder, nb_test_files)
