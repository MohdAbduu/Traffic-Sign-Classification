import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define classes dictionary - these are standard German traffic sign classes
# You may need to adjust this based on your specific dataset
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}


# Read CSV file to get class distribution
def analyze_class_distribution():
    try:
        # Try to read from CSV first
        train_df = pd.read_csv('Data/Train.csv')

        # Count occurrences of each class
        class_counts = train_df['ClassId'].value_counts().sort_index().to_dict()

        # Prepare DataFrame with class names and counts
        data = {
            'Class ID': [],
            'Class Name': [],
            'Image Count': []
        }

        for class_id, count in sorted(class_counts.items()):
            data['Class ID'].append(int(class_id))
            data['Class Name'].append(classes.get(int(class_id), f"Unknown ({int(class_id)})"))
            data['Image Count'].append(count)

        return pd.DataFrame(data)

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print("Falling back to directory analysis...")

        # If CSV fails, try to analyze directory structure
        train_path = 'Data/Train'
        class_counts = {}

        # Count images per class folder
        for class_id in os.listdir(train_path):
            class_dir = os.path.join(train_path, class_id)
            if os.path.isdir(class_dir):
                count = len([f for f in os.listdir(class_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm'))])
                class_counts[int(class_id)] = count

        # Prepare DataFrame with class names and counts
        data = {
            'Class ID': [],
            'Class Name': [],
            'Image Count': []
        }

        for class_id, count in sorted(class_counts.items()):
            data['Class ID'].append(class_id)
            data['Class Name'].append(classes.get(class_id, f"Unknown ({class_id})"))
            data['Image Count'].append(count)

        return pd.DataFrame(data)


# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Get class distribution data
df = analyze_class_distribution()

# Add combined labels with class ID for better display (optional)
df['Display Label'] = [f"{row['Class ID']}: {row['Class Name']}" for _, row in df.iterrows()]

# Plotting
plt.figure(figsize=(22, 10))
ax = sns.barplot(x='Class Name', y='Image Count', data=df, palette="viridis")
plt.xticks(rotation=90, fontsize=8)
plt.title('Number of Images per Traffic Sign Class in Training Dataset', fontsize=16)
plt.xlabel('Class Name', fontsize=12)
plt.ylabel('Image Count', fontsize=12)

# Remove the for-loop that adds class ID annotations

plt.tight_layout()
plt.savefig('plots/class_distribution.png')
plt.show()

print("Class distribution plot saved to 'plots/class_distribution.png'")