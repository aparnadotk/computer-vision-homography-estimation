import torch
import torchvision.transforms as transforms
from load_input import load_transform_input
import model
from train import train
import cv2
import numpy as np
import os
import shutil
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from load_input import HomographyDataset
from model import HomographyLoss

# Function to load and preprocess the image pair
def preprocess_images(img1_path, img2_path, transform):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    img1 = transform(img1)
    img2 = transform(img2)
    
    img_pair = torch.cat((img1, img2), dim=0)
    img_pair = img_pair.unsqueeze(0)  # Add batch dimension
    return img_pair

def load_homography_matrices(directory, num_matrices):
    """
    Load homography matrices from text files in a specified directory.

    Args:
        directory (str): Path to the directory containing homography matrix files.
        num_matrices (int): The number of matrices to load.

    Returns:
        np.ndarray: Array of homography matrices of shape (num_matrices, 3, 3).
    """
    matrices = []
    for i in range(num_matrices):
        file_path = os.path.join(directory, f'homography_{i}.txt')
        if os.path.exists(file_path):
            matrix = np.loadtxt(file_path).reshape((3, 3))
            matrices.append(matrix)
        else:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    return np.array(matrices)

def calculate_homography_accuracy(predictions_dir, ground_truths_dir, num_matrices, threshold=1e-4):
    """
    Calculate the accuracy of homography matrix predictions by comparing them with ground truth.

    Args:
        predictions_dir (str): Path to the directory containing predicted homography matrices.
        ground_truths_dir (str): Path to the directory containing ground truth homography matrices.
        num_matrices (int): The number of matrices to compare.
        threshold (float): The threshold for considering a prediction as correct.

    Returns:
        float: The accuracy of the predictions.
    """
    predicted_matrices = load_homography_matrices(predictions_dir, num_matrices)
    ground_truth_matrices = load_homography_matrices(ground_truths_dir, num_matrices)
    
    # Calculate the difference between predicted and ground truth matrices
    differences = np.abs(predicted_matrices - ground_truth_matrices)
    
    # Calculate the element-wise mean error across all matrices
    mean_errors = np.mean(differences, axis=(1, 2))
    
    # Consider a prediction as correct if its mean error is below the threshold
    correct_predictions = mean_errors < threshold
    
    # Calculate accuracy
    accuracy = np.mean(correct_predictions)
    
    return accuracy

# Function to write the predicted homography matrix to a file
def write_homography_matrix(matrix, output_file_path):
    np.savetxt(output_file_path, matrix, fmt='%.6f')

# Inference function to process all image pairs in a directory
def infer_homographies(model, input_dir, output_dir, transform):
    # Check if the output directory exists
    if os.path.exists(output_dir):
        # If it exists, empty the contents
        shutil.rmtree(output_dir)
        print(f"Emptied the directory '{output_dir}'.")
    
    # Create the directory (this will be empty if it existed and was cleared, or newly created)
    os.makedirs(output_dir)
    print(f"Created the directory '{output_dir}'.")

    # Loop through image pairs in the input directory
    filenum = -1
    for filename in os.listdir(input_dir):
        if filename.endswith('_original.png'):
            base_filename = filename[:-13]  # Remove '_1.png' to get the base name
            img1_path = os.path.join(input_dir, base_filename + '_original.png')
            img2_path = os.path.join(input_dir, base_filename + '_transformed.png')
            filenum += 1
            output_file_path = os.path.join(output_dir, 'homography_' + str(filenum) + '.txt')
            
            # Preprocess the image pair
            img_pair = preprocess_images(img1_path, img2_path, transform)
            
            # Inference
            with torch.no_grad():
                prediction = model(img_pair)
                predicted_homography = prediction.numpy().flatten()

            # Convert the flattened homography matrix back to a 3x3 matrix
            predicted_homography_matrix = np.append(predicted_homography, 1).reshape((3, 3))
            
            # Write the homography matrix to the output directory
            write_homography_matrix(predicted_homography_matrix, output_file_path)
            print(f"Predicted homography matrix for {base_filename} written to {output_file_path}")

def evaluate(model):
    # Load test dataset
    #test_dataset, test_loader = load_transform_input('test_dataset1')
    test_dataset = HomographyDataset('data/test_dataset1', num_pairs=100, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
    # Evaluation on the test dataset
    model.eval()
    test_loss = 0.0
    #criterion = nn.MSELoss()
    # Use this custom loss function in your training loop
    criterion = HomographyLoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    print(f"\nTest Loss (CustomLossFunction): {test_loss/len(test_loader)}")


parser = argparse.ArgumentParser(description="Homography CNN Training and Inference")
    
subparsers = parser.add_subparsers(dest="mode", help="Mode to run the script in")
    
# Subparser for training mode
train_parser = subparsers.add_parser("train", help="Run the training")
    
# Subparser for inference mode
infer_parser = subparsers.add_parser("infer", help="Run the inference")

args = parser.parse_args()

if args.mode in ["train", "both"]:
    train_dataset, train_dataloader = load_transform_input('data/homography_dataset1')

    # Create the model
    cnn = model.HomographyCNN()
    print(cnn)

    # Train the model
    train(cnn, train_dataloader, 'data/homography_dataset1')

if not args.mode or args.mode in ['', 'both']:
    # Inference
    # Load the model
    cnn = model.HomographyCNN()
    cnn.load_state_dict(torch.load('homography_cnn_model.pth', map_location=torch.device('cpu')))
    cnn.eval()

    # Transform to normalize the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Define input and output directories
    input_dir = 'data/test_dataset1'  # Replace with the actual path to the input directory
    output_dir = 'data/test_predictions1'  # Replace with the actual path to the output directory

    # Perform inference on all image pairs in the input directory
    #infer_homographies(cnn, input_dir, output_dir, transform)
    evaluate(cnn)

    accuracy = calculate_homography_accuracy('data/test_predictions1', 'data/test_dataset1', 100)
    print(f"Accuracy: {accuracy * 100:.2f}%")




