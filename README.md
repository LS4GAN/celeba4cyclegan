# Overview

This repository provides a script and a set of instructions to construct
datasets (e.g. Male-to-Female or Glasses Removal) suitable for training of the
unpaired image-to-image translation models (CycleGAN, CouncilGAN,
[UVCGAN](https://github.com/LS4GAN/uvcgan), etc).

# Instructions

1. Download the
   [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
   The following files are required:

   1. Train/Val/Test Partitions. File name `list_eval_partition.txt`
   2. Attributes Annotations. File name `list_attr_celeba.txt`
   3. Aligned and Cropped Images.  Archive name `img_align_celeba_png.7z`

2. Unpack the CelebA image archive `img_align_celeba_png.7z`. For example,

```bash
7z x img_align_celeba_png.7z        # requires 7zip installed
```

3. Use the provided `convert_celeba.py` script to convert the raw CelebA
   dataset into the CycleGAN form.

To create a Male-to-Female dataset you can use the following command:

```bash
python3 convert_celeba.py \
    --list-attr PATH_TO_list_attr_celeba.txt \
    --list-part PATH_TO_list_eval_partition.txt \
    --attr Male \
    PATH_TO_EXTRACTED_CELEBA_IMAGES \
    OUTPUT_DIRECTORY
```

Or, to create a Glasses removal dataset, you can run the following command:

```bash
python3 convert_celeba.py \
    --list-attr PATH_TO_list_attr_celeba.txt \
    --list-part PATH_TO_list_eval_partition.txt \
    --attr Eyeglasses \
    PATH_TO_EXTRACTED_CELEBA_IMAGES \
    OUTPUT_DIRECTORY
```

# Requirements

To run the `convert_celeba.py` script one needs to have a working python3
interpreter and the following additional packages installed:

```
pandas
tqdm
```

To unpack the original CelebA dataset one needs to have a `7zip` installed.

# Checksums

The `checksums` directory contains the reference checksums of the
Male-to-Female and Glasses removal datasets. These checksums were calculated
over the datasets provided by CouncilGAN (CouncilGAN file names changed to
match the CelebA).

The `checksums` directory also contans the reference checksums of the original
CelebA dataset archive.

