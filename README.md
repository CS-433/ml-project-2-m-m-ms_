# ML4Science: Image Denoising Project

This project focuses on improving image quality by reducing noise using state-of-the-art denoising techniques and models. The repository includes scripts for training and evaluating models, dataset (of denoised images), and metrics results of denoised images using various algorithms

---

## Table of Contents

- [Background](#background)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage:-instructions-for-running-the-denoisers)
- [Usage](#usage-instructions-for-running-the-denoisers)
- [Denoising Techniques](#denoising-techniques)
- [Results and Metrics](#results-and-metrics)
- [Running TensorBoard](#running-tensorboard)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Background

Noise in images often obstructs meaningful analysis, particularly in scientific imaging. This project aims to tackle the challenge of denoising images using a variety of techniques, including:  

- **Traditional methods**: Gaussian Filtering, Median Filtering, Non-Local Means (NL-Means), TV-Chambolle, Wavelet filtering.  
- **Advanced methods**: TV-ISO, BM3D, and deep learning-based models (Noise2Noise and Noise2Void).  

Our primary objective is to evaluate and improve denoising techniques, contributing to the advancement of image processing for scientific applications.  

This project is also inspired by the [AI4Life Challenge](https://ai4life-mdc24.grand-challenge.org/ai4life-mdc24/), which focuses on leveraging machine learning techniques to enhance image quality in biological research. While the challenge is no longer active, it provided a compelling framework for exploring state-of-the-art denoising methods. If the challenge was still open, participation in it would have been a valuable opportunity to benchmark the project's outcomes and contribute to the broader scientific community.  

---

## Features

- **Diverse Techniques**: Includes classical and machine learning-based denoising methods.
- **End-to-End Pipeline**: Scripts for data preprocessing, denoising, training models, and evaluation.
- **Extensive Metrics**: Provides quantitative evaluation of denoised images.

---

## Project Structure

```
ML4Science/
├── data/
│   ├── raw/                # Raw data (original images)
│   │   ├── readme.txt                        # description of the dataset
│   │   ├── Image%03d/                        # folders for different FOVs, ID range from 001 to 120
│   │   │   ├── wf_channel{0,1,2}.npy         # 400 shots of the FOV captured using widefield, each file contains a 400x512x512 uint16 tensor
│   │   │   ├── sim_input_channel{0,1,2}.npy  # SIM inputs for getting the SIM reconstructed results, each file contains a 1536x2560 tensor
│   │   │   └── sim_channel{0,1,2}.npy        # SIM reconstructed results of the 3 channels, each file contains a 1024x1024 tensor
│   ├── processed/          # Processed data (denoised images)
│   │   ├── Gaussian/               # Contains .npy files for all parameters of Image001, channel0
│   │   ├── Median/ 
│   │   ├── NL-Means/ 
│   │   ├── TV-Chambolle/ 
│   │   ├── TV-ISO/ 
│   │   ├── Wavelet/ 
│   │   ├── BM3D/ 
│   │   └── Noise2Noise/ 
│   ├── output/             # CSV files (results of denoised images)
│   │   ├── Gaussian/               # Contains channel by channel and average .csv files of results (metrics)
│   │   ├── Median/ 
│   │   ├── NL-Means/ 
│   │   ├── TV-Chambolle/ 
│   │   ├── TV-ISO/ 
│   │   ├── Wavelet/ 
│   │   ├── BM3D/ 
│   │   └── Noise2Noise/ 
├── models/                 # Models used for TV-ISO and Noise2Noise
│   ├── __init__.py                 # Makes models a package
│   ├── prox_tv_iso.py              
│   ├── tv_iso.py                   # Proximal TV-ISO and TV-ISO model definition
│   ├── drunet.py                   
│   ├── resblock.py                 # DRUNet and ResBlocks model definition
├── Noise2Noise/            # Files related to Noise2Noise
│   ├── config.json                 # Configuration file
│   ├── convert_dataset_to_h5.py    # Converts dataset to h5 format
│   ├── export_plots_tensorboard.py # Exports from TensorBoard to png images 
│   ├── torch_metrics.py                  # Computes metrics
|   ├── exps                        # Directory created during training phase 
│   │   ├── Noise2Noise/   
│   │   │     ├── checkpoints/            
│   │   │           ├── checkpoint.pth      # File that contains model's weight
│   │   │     ├── tensorboard_logs/       # Flie that stores the different parameters measured during training 
│   │   │     ├── config.json             # File that stores the training parameters used to train the model 
│   ├── tensorboard_plots/          # TensorBoard logs and plots directories
│   ├── train.py                    # Training script
│   ├── trainer.py                  
│   ├── w2s_bw.h5                   # Dataset compressed to load data more rapidly 
│   └── w2s_dataset.py              # Trainer and Dataset class
├── Noise2Void/             # Files related to Noise2Void
│   ├── train_n2v.ipynb             # Jupyter notebook for training Noise2Void
│   └── train_n2v.py                # Training script
├── scripts/                # Various utility scripts
│   ├── __init__.py                 # Makes scripts a package
│   ├── helpers.py                  # Helper functions
│   ├── metrics.py                  # Metrics computation script
│   ├── denoiser_selection.py       # Denoiser selection script
│   ├── denoising_pipeline.py       # Denoising pipeline script
│   └── explorations.ipynb          # Plot generation script
├── run.ipynb               # Jupyter notebook to run the project
├── run.py                  # Script to run the project
├── requirements.txt        # Project dependencies
├── LICENSE.txt             # Project license
├── .gitignore
├── .gitattributes
└── README.md               # Project description and instructions
```
---
## Getting Started

### Prerequisites

- Python 3.8 or higher
- [PyTorch](https://pytorch.org/) (for Noise2Noise and Noise2Void)
- Additional dependencies specified in `requirements.txt`

### Installation

1. **Clone** the repository:
   ```bash
   git clone https://github.com/CS-433/ml-project-2-m-m-ms_.git 
   cd ml-project-2-m-m-ms_
2. **Create** a dedicated python environnement: 
    ```bash 
        conda create -n denoising_project
        conda activate denoising_project
3. **Install** the dependencies: 
    ```bash 
    pip install -r requirements.txt

4. **Download** the [dataset](https://datasets.epfl.ch/w2s/W2S_raw.zip) (40GB compressed, 70GB uncompressed) and extract it into `data/raw/` directory 

---
## Usage: Instructions for Running the Denoisers

### For Non-Deep Learning Denoisers:
1. Run the Python script `run.py` or the Jupyter notebook `run.ipynb`.
2. Specify the `denoiser_name` to select a denoising method and its tunable values in `PARAMETER_RANGES`.
3. After denoising is complete:
   - A `.csv` file containing metrics is saved in `data/output/<denoiser_name>/`.
   - The denoised images are stored in `data/processed/<denoiser_name>/`.

### For Deep Learning Denoisers:

#### **Noise2Noise**:
**Note:** The weights of the trained model are provided in the directory `/Noise2Noise/exps/Noise2Noise/checkpoints/checkpoint.pth`.

1. **To train the model**:
   - Convert the dataset to `.h5` format using `convert_dataset_to_h5.py`.
   - Edit `config.json` to modify the training parameters.
   - Train the model with:
     ```bash
     python train.py -d [device]
     ```
   - Launch TensorBoard to monitor the training process (see Running TensorBoard).
   - After training, a `checkpoint.pth` file with the model weights is generated.

2. **To evaluate the model**:
   - Run run.py or run.ipynb with `denoiser_name = 'Noise2Noise'`.

#### **Noise2Void**:
1. **To train and evaluate the model**:
   - Use the Python script `train_n2v.py` or the Jupyter notebook `train_n2v.ipynb`.
   - The results of the denoised images will appear after the training is completed.
2. **Note**: Training `Noise2Void` hasn't been tested properly due to difficulties encountered with [CAREAmics](https://careamics.github.io/0.1/) library.

--- 

## Denoising Techniques

The following denoising methods are implemented and evaluated:

### **Traditional Methods:**
- Gaussian Filtering  
- Median Filtering  
- Non-Local Means (NL-Means)   

### **Advanced Methods** : 

#### **Non Deep Learning**: 
- Total Variation (TV) - Chambolle and ISO variants  
- Wavelet Transform  
- BM3D 

#### **Deep Learning Models:**
- **Noise2Noise**:  
  A neural network trained to map noisy images to clean images.  
- **Noise2Void**:  
  A self-supervised method requiring only noisy data.  

---

## Results and Metrics

The results are stored in the `data/output/` directory and include:

- Channel-wise and average metrics (e.g., PSNR, SSIM) in CSV format.  
- Visualizations of denoised images for qualitative comparison.  

### **Metrics Used:**
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the quality of denoised images.
- **SI-PSNR (Scale Invariant Signal-to-Noise Ratio)**: Evaluates the quality of denoised images while being invariant to scale.
- **SSIM (Structural Similarity Index)**: Evaluates the structural fidelity.
- **Runtime**: Amount of time to denoise an image.

The denoised images for each denoiser are stored in the `data/processed/denoiser_name` directory. 

Visualize the denoised images using the `explorations.ipynb` notebook in the `scripts/` folder.

---

## Running TensorBoard

To monitor the training process for **Noise2Noise**:

1. Navigate to the `Noise2Noise` directory:
   ```bash
    cd Noise2Noise
    ```
2. Lauch TensorBoard : 
    ```bash 
    tensorboard --logdir=exps/Noise2Noise/tensorboard_logs
    ```
---

## Acknowledgements 

We extend our heartfelt thanks to Daniel Sage, Stanislas Ducotterd, and Mehrsa Pourya for their guidance and support throughout this project.

---

## License 

This project is licensed under the MIT License. See the `LICENSE.txt` file for details.
