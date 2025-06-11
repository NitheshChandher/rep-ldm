import os
import shutil
import argparse

def split_ffhq_dataset(base_dir, test_count):
    # Full path to images
    all_files = sorted([f for f in os.listdir(base_dir) if f.endswith(".png") or f.endswith(".jpg")])

    # Create train and test directories
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move files to test (first 10000) and train (remaining)
    for i, file in enumerate(all_files):
        src_path = os.path.join(base_dir, file)
        if i < test_count:
            dst_path = os.path.join(test_dir, file)
        else:
            dst_path = os.path.join(train_dir, file)
        shutil.move(src_path, dst_path)

    print(f"✅ Moved {test_count} files to {test_dir} and {len(all_files) - test_count} files to {train_dir}")

def main():
    parser = argparse.ArgumentParser(description="Split FFHQ dataset into train and test sets.")
    parser.add_argument("--base_dir", type=str, default="data/ffhq512", help="Base directory containing the FFHQ images.")
    parser.add_argument("--test_count", type=int, default=10000, help="Number of images to move to the test set.")
    args = parser.parse_args()
    split_ffhq_dataset(base_dir=args.base_dir, test_count=args.test_count)

if __name__ == "__main__":
    main()