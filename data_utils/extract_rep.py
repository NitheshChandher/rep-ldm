import os
import argparse
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from timm.models.vision_transformer import VisionTransformer
from transformers import CLIPModel, CLIPProcessor

def vit_small(pretrained, progress, **kwargs):
    patch_size = kwargs.get("patch_size", 8)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = f"https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/dino_vit_small_patch8_ep200.torch"
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model

def load_dino_model():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval()
    return model

def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

def collate_pil(batch):
    images, names = zip(*batch)  # keeps them as a tuple of lists
    return list(images), list(names)


class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name

# Default transform for DINO and PE
default_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224,224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_representations(args):
    image_folder= args.data 
    output_folder= args.output
    model_name= args.model
    batch_size= args.batch_size

    os.makedirs(output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if model_name == "DINOv2":
        model = load_dino_model()
        model.to(device)
        processor = None

    elif model_name == "PE":
        model = vit_small(pretrained=True, progress=True)
        model.to(device)
        processor = None

    elif model_name == "CLIP":
        model, processor = load_clip_model()
        model.to(device)

    elif model_name == "DIFFAE":
        model = torch.load(args.model_path, map_location=device)
        model.eval()
        processor = None

    else:
        raise ValueError("Invalid model name. Choose 'DINOv2', 'PE', or 'CLIP'.")

    # Load dataset
    if model_name == "CLIP":
        transform = None
    elif model_name == "DIFFAE":
        # Transform for DIFFAE
        transform= transforms.Compose([
                        transforms.Resize(args.img_size),
                        transforms.ToTensor(),        
                        transforms.Normalize(mean=[0.5], std=[0.5])
                    ])
        print(f"Using DIFFAE image size: {args.img_size}, default transform applied!") 

    else:
        transform = default_transform

    dataset = ImageDataset(image_folder, transform=transform)
    collate_fn = collate_pil if is_clip else None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for images, img_names in dataloader:
            if model_name == "CLIP":
                # Convert tensors to PIL for CLIPProcessor
                inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
                vision_outputs = model.vision_model(**inputs)
                features = vision_outputs.last_hidden_state[:, 0, :]  # CLS token
            else:
                images = images.to(device)
                features = model(images)  # [batch_size, embedding_dim]

            for feature, img_name in zip(features, img_names):
                file_name = os.path.splitext(img_name)[0] + ".npy"
                file_path = os.path.join(output_folder, file_name)
                np.save(file_path, feature.cpu().numpy())
                print(f"Saved: {file_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the image folder")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--model", type=str, default="DINOv2", help="DINOv2, Pathology Encoder (PE), DIFFAE, or CLIP")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model weights if not using default pretrained models")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for DIFFAE")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise ValueError(f'{args.data} does not exist!')

    if os.path.exists(args.output):
        print(f'{args.output} already exists!')
    else:
        print(f"Creating a directory at {args.output}!")
        os.makedirs(args.output, exist_ok=True)

    extract_representations(args)

if __name__ == "__main__":
    main()