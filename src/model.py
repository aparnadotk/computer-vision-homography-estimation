import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np

# Define the CNN model
class HomographyCNN(nn.Module):
    def __init__(self):
        super(HomographyCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 8)
        


    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(-1, 256 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HomographyLoss(nn.Module):
    def __init__(self):
        super(HomographyLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, predicted, target):
        # Normalize both predicted and target homographies
        #predicted = predicted / predicted[:, 8].unsqueeze(1)
        #target = target / target[:, 8].unsqueeze(1)
        # Assume the model outputs 8 elements, and we append 1 as the 9th element
        batch_size = predicted.size(0)
        
        # Append 1 to the end of each predicted homography matrix
        ones = torch.ones(batch_size, 1, device=predicted.device)
        predicted_homography = torch.cat((predicted, ones), dim=1)

        # Normalize by the last element (which is set to 1 after appending)
        predicted_homography = predicted_homography / predicted_homography[:, 8].unsqueeze(1)
        #print('=========predicted matrix: ', predicted_homography)

        # Append 1 to the target homography to make it (batch_size, 9)
        target_homography = torch.cat((target, ones), dim=1)
        #print('=====target matrix: ', target_homography)
        
        # Calculate the MSE loss
        mse_loss = self.mse_loss(predicted_homography, target_homography)

            
        # Calculate additional penalty for large deviations (optional)
        deviation_penalty = torch.mean(torch.abs(predicted_homography - target_homography))
        
        # Combine losses (you can adjust the weight of deviation_penalty if needed)
        total_loss = mse_loss + 1.0 * deviation_penalty
        
        
        return total_loss


