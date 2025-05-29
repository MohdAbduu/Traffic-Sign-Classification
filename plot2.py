import os
import pathlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def visualize_traffic_sign_examples():
    # Parameters
    NUM_CATEGORIES = 43  # Number of classes
    IMG_WIDTH, IMG_HEIGHT = 32, 32

    # Define paths
    data_dir = "Data"  # Adjust if your data directory is different
    train_path = os.path.join(data_dir, "Train")
    img_dir = pathlib.Path(train_path)

    # Define the classes dictionary for reference
    classes = {
        0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
        3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
        6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
        9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Right-of-way at the next intersection', 12: 'Priority road',
        13: 'Yield', 14: 'Stop', 15: 'No vehicles',
        16: 'Vehicles over 3.5 metric tons prohibited', 17: 'No entry', 18: 'General caution',
        19: 'Dangerous curve to the left', 20: 'Dangerous curve to the right',
        21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road',
        24: 'Road narrows on the right', 25: 'Road work', 26: 'Traffic signals',
        27: 'Pedestrians', 28: 'Children crossing', 29: 'Bicycles crossing',
        30: 'Beware of ice/snow', 31: 'Wild animals crossing',
        32: 'End of all speed and passing limits', 33: 'Turn right ahead',
        34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
        37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
        40: 'Roundabout mandatory', 41: 'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'
    }

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Create figure for plotting
    plt.figure(figsize=(15, 15))

    # Load and display one image from each class
    for i in range(NUM_CATEGORIES):
        try:
            # Plot in a 7x7 grid
            plt.subplot(7, 7, i + 1)

            # Find the first image in the class folder
            class_folder = os.path.join(train_path, str(i))
            if not os.path.exists(class_folder):
                # If class folder doesn't exist, show empty plot
                plt.text(0.5, 0.5, f"No images\nfor class {i}",
                         horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
                continue

            # Get list of image files
            image_files = [f for f in os.listdir(class_folder)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm'))]

            if not image_files:
                # If no valid images, show empty plot
                plt.text(0.5, 0.5, f"No valid images\nfor class {i}",
                         horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
                continue

            # Load the first image
            img_path = os.path.join(class_folder, image_files[0])
            img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))

            # Display the image
            plt.imshow(img)
            plt.title(f"{i}", fontsize=8)
            plt.axis('off')

        except Exception as e:
            print(f"Error displaying class {i}: {e}")
            plt.text(0.5, 0.5, f"Error loading\nclass {i}",
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')

    plt.suptitle("Traffic Sign Examples (One per Class)", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig('plots/traffic_sign_examples.png')
    plt.show()

    print("Traffic sign examples visualization saved to 'plots/traffic_sign_examples.png'")


# Run the visualization
visualize_traffic_sign_examples()