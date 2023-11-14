import os
from sklearn.model_selection import train_test_split


# Function to get file paths and corresponding labels
def get_paths_and_labels(dataset_root):
    file_paths = []
    labels = []

    for folder_name in os.listdir(dataset_root):
        folder_path = os.path.join(dataset_root, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                file_paths.append(file_path)
                labels.append(folder_name)

    return file_paths, labels


# Set the root folder of your dataset
dataset_root = "data"  # Update this to your dataset folder

# Get file paths and labels
file_paths, labels = get_paths_and_labels(dataset_root)

print("Number of file paths:", len(file_paths))
print("Number of labels:", len(labels))

# Split the dataset into training and testing sets
train_paths, test_paths, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.2,
                                                                      random_state=42)

# Now you have the training and testing sets ready for further processing
