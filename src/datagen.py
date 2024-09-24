import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm

# Function to create a synthetic image with a geometric shape
def create_geometric_image(image_size=(256, 256)):
    #img = np.zeros(image_size, dtype=np.uint8)
    img = np.random.randint(0, 50, image_size, dtype=np.uint8)
    shape_type = np.random.choice(['rectangle', 'circle', 'triangle', 'ellipse'])
    
    if shape_type == 'rectangle':
        top_left = (np.random.randint(20, 100), np.random.randint(20, 100))
        bottom_right = (np.random.randint(150, 230), np.random.randint(150, 230))
        img = cv2.rectangle(img, top_left, bottom_right, 255, -1)
    
    elif shape_type == 'circle':
        center = (np.random.randint(50, 200), np.random.randint(50, 200))
        radius = np.random.randint(20, 100)
        img = cv2.circle(img, center, radius, 255, -1)
    
    elif shape_type == 'triangle':
        pt1 = (np.random.randint(50, 200), np.random.randint(50, 200))
        pt2 = (np.random.randint(50, 200), np.random.randint(50, 200))
        pt3 = (np.random.randint(50, 200), np.random.randint(50, 200))
        pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
        img = cv2.fillPoly(img, [pts], 255)
    
    elif shape_type == 'ellipse':
        center = (np.random.randint(50, 200), np.random.randint(50, 200))
        axes = (np.random.randint(20, 100), np.random.randint(20, 100))
        angle = np.random.randint(0, 360)
        start_angle = 0
        end_angle = 360
        img = cv2.ellipse(img, center, axes, angle, start_angle, end_angle, 255, -1)
    
    return img

# Function to apply a random homography to an image
def apply_homography(img):
    h, w = img.shape
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    pts2 = np.float32([[np.random.randint(0, w//4), np.random.randint(0, h//4)],
                       [w - np.random.randint(0, w//4), np.random.randint(0, h//4)],
                       [np.random.randint(0, w//4), h - np.random.randint(0, h//4)],
                       [w - np.random.randint(0, w//4), h - np.random.randint(0, h//4)]])
    H = cv2.getPerspectiveTransform(pts1, pts2)
    img_transformed = cv2.warpPerspective(img, H, (w, h))
    return img_transformed, H

# Function to generate the dataset
def generate_dataset(output_dir, num_pairs, image_size=(256, 256)):
    # Check if the directory exists and empty it if it does
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Emptied the directory '{output_dir}'.")
    
    # Create the directory
    os.makedirs(output_dir)
    print(f"Created the directory '{output_dir}'.")

    # Generate the image pairs
    for i in tqdm(range(num_pairs), desc=f"Generating dataset in '{output_dir}'"):
        img = create_geometric_image(image_size)
        img_transformed, H = apply_homography(img)
        cv2.imwrite(f'{output_dir}/image_{i}_original.png', img)
        cv2.imwrite(f'{output_dir}/image_{i}_transformed.png', img_transformed)
        np.savetxt(f'{output_dir}/homography_{i}.txt', H)

# Generate the training dataset
generate_dataset('data/homography_dataset1', num_pairs=5000)

# Generate the test dataset
generate_dataset('data/test_dataset1', num_pairs=500)
