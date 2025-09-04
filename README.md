

-----

# 🎥 Video Action Recognition using ResNet50 and LSTM

This project demonstrates a deep learning approach for action recognition in videos. It leverages a two-stage model: first, a pre-trained `ResNet50` is used to extract spatial features from individual video frames, and second, an `LSTM` network is used to model the temporal sequence of these features to classify the action.

The implementation is contained within a single Jupyter Notebook (`Action_Detection_In_Video.ipynb`) and uses the **UCF-101 dataset**.

-----

## Table of Contents
Project Overview

Model Architecture

Dataset

Getting Started

Results

Inference on a New Video

Contributing

-----

## Project Overview

The goal of this project is to classify human actions in video clips. The notebook walks through the entire pipeline, from data preparation and feature extraction to model training and evaluation.

### Key Features:

  * **Data Preprocessing:** Scripts to handle and create a smaller, manageable subset of the large UCF-101 dataset.
  * **Efficient Feature Extraction:** Uses a pre-trained ResNet50 model to generate feature vectors for video frames, avoiding the need to train a CNN from scratch.
  * **Temporal Modeling:** Employs an LSTM (Long Short-Term Memory) network to learn the sequence of actions from the frame features.
  * **Complete Pipeline:** Includes code for training, validation with early stopping, and final testing.
  * **Inference Ready:** A dedicated section to run predictions on a new, unseen video.

-----

## Model Architecture

The model is composed of two main components:

1.  **Spatial Feature Extractor (ResNet50):** For each video, we sample **16 frames**. Each frame is passed through a pre-trained ResNet50 model (with its final classification layer removed). This process converts each frame into a **2048-dimensional feature vector**, capturing the spatial information within that frame.

2.  **Temporal Sequence Classifier (LSTM):** The sequence of 16 feature vectors (one for each frame) is then fed into an LSTM network. The LSTM processes this sequence to understand the temporal dynamics of the action. The final output of the LSTM is passed through a linear classifier to predict the action class.

-----

## Dataset 📊

This project uses the **UCF-101 Action Recognition Dataset**. Due to its size, the initial steps in the notebook create a smaller subset for faster experimentation.

  * **Classes:** 25 action classes are selected.
  * **Videos per Class:** Up to 200 videos are sampled from each selected class.
  * **Splits:** The subset is divided into training (80%), validation (10%), and testing (10%) sets.

The metadata for the subset and the data splits are saved as `.csv` files.

-----

## Getting Started 🚀

This project is designed to be run in a **Google Colab** environment to take advantage of free GPU resources.

### Prerequisites

  * A Google Account to use Google Colab and Google Drive.
  * The UCF-101 dataset uploaded to your Google Drive.

### Setup

1.  **Clone the Repository (Optional):**

    ```bash
    git clone <your-repository-url>
    ```

2.  **Upload to Google Drive:**

      * Upload the `Action_Detection_In_Video.ipynb` notebook to your Google Drive.
      * Upload the UCF-101 dataset to a folder in your Google Drive (e.g., `My Drive/Datasets/UCF-101`).

3.  **Install Dependencies:** The notebook uses standard Python libraries. You can install them using `pip`:

    ```bash
    pip install torch torchvision pandas numpy opencv-python scikit-learn decord tqdm
    ```

4.  **Open in Colab:** Open the notebook in Google Colab and ensure the runtime is set to use a GPU accelerator (`Runtime` \> `Change runtime type` \> `GPU`).

### Workflow

The Jupyter Notebook is organized into sequential steps. Run the cells in order to execute the full pipeline:

1.  **Mount Google Drive:** Connect your Google Drive to access the dataset.
2.  **Create Dataset Subset:** A smaller metadata file (`metadata.csv`) is generated for a subset of 25 classes.
3.  **Extract Frame Features:** The script samples 16 frames from each video, passes them through ResNet50, and saves the resulting feature sequences as `.npy` files. *This is the most time-consuming step.*
4.  **Prepare Dataloaders:** The notebook creates PyTorch `Dataset` and `DataLoader` objects for the training, validation, and test sets.
5.  **Train the LSTM Model:** The LSTM classifier is trained on the feature sequences with validation, learning rate scheduling, and early stopping.
6.  **Evaluate the Model:** The trained model is evaluated on the unseen test set to report the final accuracy.

-----

## Results 🏆

After training, the model achieves high accuracy on the test set, demonstrating its effectiveness in recognizing actions from video data.

> **Final Test Accuracy: 97.62%**

This result indicates that the two-stage ResNet50-LSTM architecture is highly effective for this task.

-----

## Inference on a New Video 🎬

The final section of the notebook provides a complete script to test the trained model on a new video file.

1.  Upload any video file (e.g., in `.mp4` format) to your Colab session.
2.  Update the `video_path` variable in the "Model Testing" section to point to your video.
3.  Run the cells. The script will load the models, process your video, and predict the action.

**Example Output:**

```console
[INFO] Processing video: /content/my_test_video.mp4
[INFO] Extracted 16 frames.
[INFO] Extracted features with shape: torch.Size([1, 16, 2048])

==============================
Prediction: BENCHPRESS
Confidence: 98.54%
==============================
```

-----

## Contributing 🙏

Contributions are welcome\! If you have suggestions for improvements, please feel free to create a **pull request** or open an **issue**.
