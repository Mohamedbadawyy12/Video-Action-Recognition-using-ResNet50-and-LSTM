# Video Action Recognition using ResNet50 and LSTM

This project demonstrates a deep learning approach for action recognition in videos. It leverages a two-stage model: first, a pre-trained **ResNet50** is used to extract spatial features from individual video frames, and second, an **LSTM** network is used to model the temporal sequence of these features to classify the action.

The implementation is contained within a single Jupyter Notebook (`Action_Detection_In_Video.ipynb`) and uses the **UCF-101 dataset**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Workflow](#workflow)
- [Results](#results)
- [Inference on a New Video](#inference-on-a-new-video)
- [Contributing](#contributing)

---

## Project Overview
The goal of this project is to classify human actions in video clips. The notebook walks through the entire pipeline, from data preparation and feature extraction to model training and evaluation.

**Key Features:**
- **Data Preprocessing:** Scripts to handle and create a smaller, manageable subset of the large UCF-101 dataset.
- **Efficient Feature Extraction:** Uses a pre-trained ResNet50 model to generate feature vectors for video frames, avoiding the need to train a CNN from scratch.
- **Temporal Modeling:** Employs an LSTM network to learn the sequence of actions from the frame features.
- **Complete Pipeline:** Includes code for training, validation with early stopping, and final testing.
- **Inference Ready:** A dedicated section to run predictions on a new, unseen video.

---

## Model Architecture
The model consists of two main components:

1. **Spatial Feature Extractor (ResNet50):**  
   - For each video, 16 frames are sampled.  
   - Each frame is passed through a pre-trained ResNet50 (final classification layer removed).  
   - Converts each frame into a 2048-dimensional feature vector capturing spatial information.

2. **Temporal Sequence Classifier (LSTM):**  
   - The sequence of 16 feature vectors is fed into an LSTM network.  
   - The LSTM learns temporal dynamics of the action.  
   - The final output is passed through a linear classifier to predict the action class.

---

## Dataset
This project uses the **UCF-101 Action Recognition Dataset**.  

- **Classes:** 25 selected action classes  
- **Videos per Class:** Up to 200 videos  
- **Data Splits:** Training (80%), Validation (10%), Testing (10%)  
- **Metadata:** Subset metadata and data splits saved as `.csv` files  

> A smaller subset is used to allow faster experimentation.

---

## Getting Started
This project is designed to be run in **Google Colab** to utilize free GPU resources.

### Prerequisites
- Google Account (for Google Colab & Google Drive)  
- UCF-101 dataset uploaded to Google Drive  

---

## Setup
1. **Clone the repository (Optional):**
```bash
git clone <your-repository-url>
