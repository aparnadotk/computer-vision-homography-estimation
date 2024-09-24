import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from model import HomographyLoss
import time

def visualize_predictions(original_image, true_H, pred_H, image_index, save_dir=None):
    # Apply the true and predicted homographies to the original image
    h, w = original_image.shape
    true_transformed = cv2.warpPerspective(original_image, true_H, (w, h))
    pred_transformed = cv2.warpPerspective(original_image, pred_H, (w, h))

    if save_dir:
        # Create the directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the images
        orig_image_path = os.path.join(save_dir, f'true_original_{image_index}.png')
        true_image_path = os.path.join(save_dir, f'true_transformed_{image_index}.png')
        pred_image_path = os.path.join(save_dir, f'pred_transformed_{image_index}.png')
        
        cv2.imwrite(orig_image_path, original_image)
        cv2.imwrite(true_image_path, true_transformed)
        cv2.imwrite(pred_image_path, pred_transformed)
        
        print(f"Saved true and predicted transformed images for Image {image_index}.")

    else:
        # Display the original and transformed images
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(original_image, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title("True Transformed Image")
        plt.imshow(true_transformed, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Predicted Transformed Image")
        plt.imshow(pred_transformed, cmap='gray')

        plt.show()


def train(model, dataloader, output_dir):
    begin_time = time.time()

    # Set device to CPU
    device = torch.device('cpu')

    # Move model to device
    model.to(device)

    # Define loss function and optimizer
    #criterion = nn.MSELoss()
    # Use this custom loss function in your training loop
    criterion = HomographyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define a learning rate scheduler that reduces LR when the loss has stopped improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # Training loop
    num_epochs = 300

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (img_pairs, H) in enumerate(dataloader):
            img_pairs, H = img_pairs.to(device), H.to(device)
        
            optimizer.zero_grad()
        
            outputs = model(img_pairs)
            loss = criterion(outputs, H)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()

            # Store the last batch of actual and predicted matrices
            if i == len(dataloader) - 1:
                last_batch_actual = H.detach().cpu().numpy()
                last_batch_predicted = outputs.detach().cpu().numpy()
            
            #break # TBD - remove this !!!!!!!!!!!!!!!!!!!!!!!!!!
    
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Step the learning rate scheduler based on the epoch loss
        scheduler.step(epoch_loss)

        # Print the current learning rate after stepping the scheduler
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Learning rate after epoch {epoch+1}: {current_lr:.6f}")

    # Print the actual and predicted matrices for the images in the last batch
    num_images = len(last_batch_actual)
    print(f"\nActual vs. Predicted Homography Matrices for the Last {min(10, num_images)} Images:")
    for i in range(-min(10, num_images), 0):
        # Ensure matrices are reshaped correctly
        #actual_matrix = last_batch_actual[i].reshape(3, 3)
        #predicted_matrix = last_batch_predicted[i].reshape(3, 3)
        actual_matrix = np.append(last_batch_actual[i], 1).reshape(3, 3)
        predicted_matrix = np.append(last_batch_predicted[i], 1).reshape(3, 3)
    
        print(f"\nImage {i + num_images}:")
        print("Actual Matrix:")
        print(actual_matrix)
        print("Predicted Matrix:")
        print(predicted_matrix)

    # Save the model
    torch.save(model.state_dict(), 'homography_cnn_model.pth')

    # Visualize the predictions for the last 10 images
    save_dir = 'predicted_images'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in range(min(10, num_images)):
        original_image = cv2.imread(f'{output_dir}/image_{i}_original.png', cv2.IMREAD_GRAYSCALE)
        true_H = np.loadtxt(f'{output_dir}/homography_{i}.txt')
        #pred_H = last_batch_predicted[i].reshape(3, 3)  # Ensure reshaping
        pred_H = np.append(last_batch_predicted[i], 1).reshape(3, 3)
        visualize_predictions(original_image, true_H, pred_H, i, save_dir)

    end_time = time.time()

    print('TOTAL TRAINING TIME: ', end_time - begin_time)



