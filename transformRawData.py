import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

TRAINING_FILE = '/Train.csv'
IMAGE_PATH_EXT = '/images/'


def addPath(path, save: bool = False):
    df = pd.read_csv(path + TRAINING_FILE)
    image_dir = path + IMAGE_PATH_EXT

    df['Path'] = df['Image_ID'].apply(lambda x: image_dir + x)

    if save:
        df.to_csv("transformData.csv", encoding='utf-8', index=False)

    #  testing
    #  print(df.head(3))
    #  print(df['class'].value_counts())


def loadImage(img_path, image_size=(255, 255)):
    image = Image.open(img_path).convert("RGB")

    # image transformation constraints. Normalizes to [0, 1]
    # transforms.Normalize(mean=[0.6452, 0.6014, 0.6137],
    #                              std=[0.2514, 0.2423, 0.2315])
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    img_tensor = transform(image)

    return img_tensor


def showSample(num_of_samples=3):
    df = pd.read_csv('transformData.csv')

    unique_classes = df["class"].unique()
    fig, axes = plt.subplots(len(unique_classes), num_of_samples, figsize=(12, len(unique_classes) * 5))

    for i, unique_class in enumerate(unique_classes):
        class_df = df[df["class"] == unique_class]
        sample_df = class_df.sample(num_of_samples, replace=True)
        sample_df_images = [loadImage(path) for path in sample_df["Path"]]

        for j, image in enumerate(sample_df_images):
            print(image)
            ax = axes[i, j]
            reformated_tensor = image.permute(1, 2, 0).numpy()
            ax.imshow(reformated_tensor)
            ax.axis('off')

            if j == 0:
                ax.set_title(unique_class)

    plt.tight_layout()
    plt.show()


#showSample(num_of_samples=3)


def calculate_mean_std(path):
    df = pd.read_csv(path)

    transform = transforms.Compose([
        transforms.ToTensor()  # Convert images to [0, 1] range
    ])

    mean = torch.zeros(3)  # For 3 channels: R, G, B
    std = torch.zeros(3)
    total_images = 0

    # Loop over the image paths from the DataFrame
    for img_path in df["Path"]:
        img = Image.open(img_path)  # Open the image
        img_tensor = transform(img)  # Convert to tensor
        total_images += 1

        mean += img_tensor.mean(dim=[1, 2])  # Sum of means for each channel
        std += img_tensor.std(dim=[1, 2])  # Sum of standard deviations for each channel

        print(total_images)

    mean /= total_images
    std /= total_images

    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")

# calculate_mean_std('transformData.csv')


def classDis(path):
    data = pd.read_csv(path)
    class_counts = data['class'].value_counts()

    class_percentages = (class_counts / len(data)) * 100

    print("Class Counts:\n", class_counts)
    print("\nClass Percentages:\n", class_percentages)
