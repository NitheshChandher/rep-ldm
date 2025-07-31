import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import argparse

def make_image_grid(image_folder, grid_size=(6, 12), output_path="image_grid.png", image_size=(128, 128)):
    # Get list of image files
    valid_exts = (".png", ".jpg", ".jpeg")
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if f.lower().endswith(valid_exts)]

    num_images = grid_size[0] * grid_size[1]
    if len(image_files) < num_images:
        raise ValueError(f"Not enough images in the folder. Required: {num_images}, Found: {len(image_files)}")

    # Randomly sample
    selected_files = random.sample(image_files, num_images)

    # Create figure with no spacing
    fig, axes = plt.subplots(*grid_size, figsize=(grid_size[1], grid_size[0]))
    plt.subplots_adjust(wspace=0, hspace=0)

    for ax, img_path in zip(axes.flatten(), selected_files):
        img = Image.open(img_path).convert("RGB").resize(image_size)
        ax.imshow(img)
        ax.axis("off")

    # Remove padding around the figure
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved image grid to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to folder with images")
    parser.add_argument("--output", default="image_grid.png", help="Output file name")
    args = parser.parse_args()

    make_image_grid(args.folder, output_path=args.output)
