# Is Representation Conditioning the Key to Controllable Image Manipulation in Diffusion Models?

##### [Nithesh Chandher Karthikeyan](https://liu.se/en/employee/nitch36), [Jonas Unger](https://liu.se/medarbetare/jonun48), and [Gabriel Eilertsen](https://liu.se/en/employee/gabei62)

---

This is the official code repository for the paper "**Is Representation Conditioning the Key to Controllable Image Manipulation in Diffusion Models?**".

![Project Pipeline](/assets/pipeline.jpg)

---

## Table of Contents

- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
  - [CelebA](#celeba)
  - [FFHQ](#ffhq)
- [Model Training](#model-training)
  - [Unconditional Latent Diffusion Model](#unconditional-latent-diffusion-model)
  - [Representation-Conditioned Model](#representation-conditioned-model)
  - [Training on Berzelius](#training-on-berzelius)
- [Sampling](#sampling)
- [Support](#support)
- [Acknowledgements](#acknowledgements)

---

## Requirements

To set up the environment, create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate dino-ldm
```

## Data Preparation

### CelebA

Follow the instructions in the [taming-transformers](https://github.com/CompVis/taming-transformers#celeba-hq) repository to download CelebA. Then, extract the images to `./data/celeba/img_align_celeba`.

### FFHQ

Download the FFHQ dataset from [Kaggle](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) and extract the images to `./data/ffhq`.

Once the images are extracted, precompute DINOv2 representations using the following command:

```bash
python3 src/extract_rep.py --data="./data/celeba/img_align_celeba" --output="./data/celeba/rep"
```

## Model Training

### Unconditional Latent Diffusion Model

The configuration for training an unconditional latent diffusion model is located in `configs/u-ldm.yaml`. This model uses a pre-trained KL-regularized autoencoder and DDPM scheduler from CompVis/stable-diffusion-v1-4, integrated via the [Diffusers](https://huggingface.co/docs/diffusers/en/index) library. Start training by running:

```bash
accelerate launch src/u-ldm.py configs/u-ldm.yaml
```

### Representation-Conditioned Model

To train the representation-conditioned model, use the configuration in `configs/dino-ldm.yaml`. Begin training by executing:

```bash
accelerate launch src/dino-ldm.py configs/dino-ldm.yaml
```

### Training on GPU Cluster

If using a GPU cluster for training, modify the `launch_job_dino-ldm.sh` script according to your model training configuration and run:

```bash
sbatch launch_job_dino-ldm.sh
```

To cancel a job:

```bash
scancel [job-id]
```

## Sampling

To sample images, run:

```bash
python3 src/sample.py --dataset="{dataset_name}" --data_path="data/{dataset_name}" --output_path="output/dinov2-ldm-{dataset_name}/images" --model_path="output/dinov2-ldm-{dataset_name}/model/{checkpoint_name}.pth"
```

For image-editing based sampling in the DINOv2-LDM condition space, run:

```bash
python3 src/ddpm_inv.py --dataset="{dataset_name}" --data_path="data/{dataset_name}" --output_path="output/dinov2-ldm-{dataset_name}/images" --model_path="output/dinov2-ldm-{dataset_name}/model/{checkpoint_name}.pt"
```

## Support 

If you have questions or need assistance, please open an [issue](https://github.com/NitheshChandher/dino-ldm/issues/new) on our GitHub repository.

## Acknowledgements

We extend our thanks to the following:
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion.git): For providing the foundational latent diffusion model, along with pre-trained checkpoints for the Autoencoder and DDPM Scheduler.
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/en/index): For their easy-to-use library components for diffusion models.
- [Accelerator](https://huggingface.co/docs/accelerate/en/package_reference/accelerator): For their distributed training framework, enabling multi-GPU training for DINOv2-LDM.
- [DINOv2](https://github.com/facebookresearch/dinov2): For providing the DINOv2 model and its pre-trained checkpoint, which are used to generate image representations in this project.
```