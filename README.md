# computer-vision-homography-estimation
A deep learning based computer vision application to automate homography estimation for pairs of related images..

The goal of this project is to develop a Convolutional Neural Network (CNN) to estimate the homography matrix between pairs of images, providing an alternative to traditional homography computation methods. The application explore whether a neural network can directly estimate the homography matrix between related images. This approach bypasses the conventional computer vision process of feature detection, matching, RANSAC, and algebraic computation, potentially enhancing robustness and accuracy.

# Tools Used
•	PyTorch, OpenCV, NumPy.

# Dataset:
Due to the scarcity of available datasets for homography estimation and the high computational requirements of existing ones, I opted to generate a synthetic dataset. This approach allowed me to create simple geometric shapes as base images and apply random homographic transformations to them. These image pairs, along with their corresponding homography matrices, were saved for training and testing. The OpenCV library was used for generating the shapes and applying the transformations.

