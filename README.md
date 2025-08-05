# Is Representation Conditioning the Key to Controllable Image Manipulation in Diffusion Models?

##### [Nithesh Chandher Karthikeyan](https://liu.se/en/employee/nitch36), [Jonas Unger](https://liu.se/medarbetare/jonun48), and [Gabriel Eilertsen](https://liu.se/en/employee/gabei62)

---

This is the official code repository for the paper:

> **Is Representation Conditioning the Key to Controllable Image Manipulation in Diffusion Models?**

![Project Pipeline](/assets/pipeline.jpg)

---

## 📋 Table of Contents

- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
  - [FFHQ](#ffhq)
  - [ImageNet-100](#imagenet-100)
  - [LSUN-Church](#lsun-church)
  - [CelebA-HQ](#celebahq)
- [Model Training](#model-training)
  - [Unconditional Latent Diffusion Model](#unconditional-latent-diffusion-model)
  - [Representation-Conditioned Model](#representation-conditioned-model)
  - [Diffusion Autoencoder](#diffusion-auto-encoder)
  - [Training on GPU Cluster](#training-on-gpu-cluster)
- [Sampling](#sampling)
- [Support](#support)
- [Acknowledgements](#acknowledgements)

---

## ✅ Requirements

To set up the environment:

```bash
conda env create -f environment.yml
conda activate dino-ldm
```
---

## 📁 Data Preparation

Ensure that all datasets are organized under the `./data/` directory.

### 😎 FFHQ

- **Download** from [Kaggle](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq).
- **Extract to**: `./data/ffhq/`
- **Split into**:
  - `./data/ffhq/train/` → all images except the first 10,000  
  - `./data/ffhq/test/` → first 10,000 images

---

### 🖼️ ImageNet-100

- **Download** from [Kaggle](https://www.kaggle.com/datasets/ambityga/imagenet100).
- **Extract to**:
  - `./data/imagenet100/train/`  
  - `./data/imagenet100/val/`

---

### 🏛️ LSUN-Church

- **Follow instructions** on the [LSUN GitHub repo](https://github.com/fyu/lsun).
- **Export images** and organize as:
  - `./data/lsun-church/train/` → 120,000 images  
  - `./data/lsun-church/val/` → remaining images

---

### 😎 CelebA-HQ

- **Download** from [Kaggle](https://www.kaggle.com/datasets/vincenttamml/celebamaskhq512).
- **Extract to**: `./data/celeba/img_align_celeba/`

---

**Precompute** DINOv2 or CLIP representations for the above datasets using the following example command:

```bash
python3 src/extract_rep.py --data="./data/celeba/img_align_celeba" --output="./data/celeba/rep"
```
---

## 🧠 Model Training

### 🔹 Unconditional Latent Diffusion Model

Based on [`CompVis/stable-diffusion-v1-4`](https://github.com/CompVis/stable-diffusion), using the Hugging Face [Diffusers](https://huggingface.co/docs/diffusers/en/index) library:

```bash
accelerate launch baseline.py --config="/path/to/config_file/.yaml"
```

### 🔹 Representation-Conditioned Model

Train with DINOv2 or CLIP representations:

```bash
accelerate launch src/rep-ldm.py --config="/path/to/config_file/.yaml"
```

### 🔹 Diffusion Auto-Encoder

To train the Diffusion Autoencoder model, use the following command:

```bash
accelerate launch train-diffae.py --config="/path/to/config_file/.yaml"
```

> 📁 **Note**: All model configs are stored in `./configs/` and structured by dataset names.

---

## 🖥️ Training on GPU Cluster

Use Slurm-compatible job scripts:

Edit the bash scripts in the `./bash/` folder to match your configuration.

Submit jobs:

```bash
sbatch {job_script}.sh
```

To cancel a job:

```bash
scancel [job-id]
```

---

## 🎨 Sampling

### 1. Generate New Samples

```bash
python3 src/experiment.py
--dataset="{dataset_name}"
--data_path="data/{dataset_name}"
--output_path="output/{model_name}-{dataset_name}/images"
--model_path="output/{model_name}-{dataset_name}/model/{checkpoint_name}.pth"
```


### 2. DDPM Inversion (for real image projection)

```bash
python3 src/ddpm_inv.py
--dataset="{dataset_name}"
--data_path="data/{dataset_name}"
--output_path="output/{model_name}-{dataset_name}/images"
--model_path="output/{model_name}-{dataset_name}/model/{checkpoint_name}.pt"
```

### 3. Attribute Manipulation

```bash
python3 src/attribute_manipulation.py
--dataset="{dataset_name}"
--data_path="data/{dataset_name}"
--output_path="output/{model_name}-{dataset_name}/images"
--model_path="output/{model_name}-{dataset_name}/model/{checkpoint_name}.pt"
```

### 4. Interpolation in Representation Space

```bash
python3 src/interpolate.py
--dataset="{dataset_name}"
--data_path="data/{dataset_name}"
--output_path="output/{model_name}-{dataset_name}/images"
--model_path="output/{model_name}-{dataset_name}/model/{checkpoint_name}.pt"
```

---

## 🤝 Support

For issues, bug reports, or questions, please open an issue on this GitHub repository.

---

## 🙏 Acknowledgements

- **Stable Diffusion:** For the foundational latent diffusion model and pre-trained components.  
- **Hugging Face Diffusers:** For accessible diffusion pipelines.  
- **Accelerate:** For scalable multi-GPU training.  
- **DINOv2 and CLIP:** For robust image representations.
- **DiffAE:** For providing the trainable encoder diffusion model.