import os
from PIL import Image
import numpy as np
from pathlib import Path

# Define supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

def is_image_file(path):
    return path.suffix.lower() in IMAGE_EXTENSIONS

def load_image_as_array(image_path):
    with Image.open(image_path) as img:
        # Convert to RGB mode to ensure consistent channels
        img = img.convert('RGB')
        img = img.resize((512, 512))  # resize for faster comparison
        return np.array(img)

def calculate_similarity(img1_array, img2_array):
    """
    Calculate similarity between two images using pixel-wise comparison.
    Returns mean squared error - lower values mean more similar images.
    """
    # Normalize pixel values to 0-1 range
    img1_norm = img1_array.astype('float32') / 255.0
    img2_norm = img2_array.astype('float32') / 255.0
    
    # Calculate mean squared error
    mse = np.mean((img1_norm - img2_norm) ** 2)
    return mse

def find_matching_pairs():
    clean_dir = Path('tmp/clean')
    noisy_dir = Path('tmp/noisy')
    
    # Filter image files only
    clean_images = [f for f in clean_dir.glob('*.*') if is_image_file(f)]
    noisy_images = [f for f in noisy_dir.glob('*.*') if is_image_file(f)]
    
    if not clean_images or not noisy_images:
        print("No image files found in one or both directories.")
        return
    
    print(f"Found {len(clean_images)} clean images and {len(noisy_images)} noisy images")
    
    # Load all clean images
    clean_arrays = {
        img_path: load_image_as_array(img_path)
        for img_path in clean_images
    }
    
    # For each noisy image, find the best matching clean image
    for noisy_path in noisy_images:
        try:
            noisy_array = load_image_as_array(noisy_path)
            best_match = None
            best_score = float('inf')
            
            for clean_path, clean_array in clean_arrays.items():
                score = calculate_similarity(clean_array, noisy_array)
                # print(f"MSE between {noisy_path.name} and {clean_path.name}: {score:.6f}")
                if score < best_score:
                    best_score = score
                    best_match = clean_path
            
            if best_match:
                new_name = noisy_dir / best_match.name
                if noisy_path != new_name:
                    noisy_path.rename(new_name)
                    print(f"Renamed {noisy_path.name} to {new_name.name} (MSE: {best_score:.6f})")
        except Exception as e:
            print(f"Error processing {noisy_path}: {e}")

if __name__ == '__main__':
    find_matching_pairs()
