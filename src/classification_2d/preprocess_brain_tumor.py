import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

def preprocess_brain_tumor_dataset(dataset_path, output_path):
    """
    Processes the brain tumor dataset and creates a CSV file with image paths and labels,
    ready for deep learning training.
    
    Args:
        dataset_path (str): Path to original dataset with 'yes' and 'no' folders
        output_path (str): Path to output directory
    
    Returns:
        pd.DataFrame: DataFrame containing image paths and labels
    """
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize lists for DataFrame
    image_paths = []
    labels = []
    
    # Process each class
    for class_name in ['no', 'yes']:
        class_path = os.path.join(dataset_path, class_name)
        label = 0 if class_name == 'no' else 1
        
        # Check if the directory exists
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} not found!")
            continue
        
        # Process all images in the class
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Copy image to output directory
                src_path = os.path.join(class_path, img_name)
                dst_path = os.path.join(output_path, f"{class_name}_{img_name}")
                shutil.copy2(src_path, dst_path)
                
                # Store path and label
                image_paths.append(dst_path)
                labels.append(label)
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })
    
    # Save DataFrame to CSV
    csv_path = os.path.join(output_path, 'dataset.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Processing completed. Dataset info:")
    print(f"Total images: {len(df)}")
    print(f"No tumor images: {len(df[df['label'] == 0])}")
    print(f"Tumor images: {len(df[df['label'] == 1])}")
    print(f"CSV file saved to: {csv_path}")
    
    return df

if __name__ == "__main__":
    dataset_path = "datasets/brain_mri"
    output_path = "datasets/brain_tumor"
    
    df = preprocess_brain_tumor_dataset(dataset_path, output_path)
