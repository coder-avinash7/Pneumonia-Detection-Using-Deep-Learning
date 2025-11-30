# Pneumonia Detection Using Deep Learning

## Overview
This project leverages deep learning to detect pneumonia from chest X-ray images using a pre-trained VGG19 model. The dataset, sourced from Kaggle, contains X-ray images categorized as "Pneumonia" or "Normal". The application includes a Flask-based web interface for user interaction.

---

## Video Demo

Here's a video demo of the project:



---

## Features
- Utilizes transfer learning with the VGG19 model for efficient feature extraction.
- Employs advanced data augmentation techniques for robust model training.
- Implements early stopping and learning rate reduction to optimize training.
- Flask web application for user-friendly pneumonia detection.

---

## Tools & Libraries
The project uses the following libraries and tools:

- **Deep Learning Frameworks:**
  - TensorFlow
  - Keras
- **Data Processing:**
  - Pandas
  - NumPy
- **Image Processing:**
  - OpenCV
  - Pillow (PIL)
- **Web Framework:**
  - Flask

---

## Dataset
- **Source:** [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Description:**
  - Contains 5,863 JPEG X-ray images in two categories: Pneumonia and Normal.
  - Images are organized into three folders: `train`, `test`, and `val`.
  - Pediatric patients (1 to 5 years old) from Guangzhou Women and Children’s Medical Center.

---

## What, How, and Why
### What
This project focuses on developing a deep learning model to classify chest X-rays into two categories: "Pneumonia" or "Normal". The goal is to assist healthcare professionals in early diagnosis.

### How
1. Preprocess the dataset using TensorFlow's `ImageDataGenerator`.
2. Fine-tune a pre-trained VGG19 model for feature extraction.
3. Train the model with custom layers for binary classification.
4. Evaluate the model's performance using test data.
5. Deploy the model using a Flask-based web application.

### Why
Pneumonia is a significant global health concern, especially in children. Early detection can drastically improve treatment outcomes. Automating the detection process using AI can:
- Reduce diagnostic errors.
- Save time for healthcare professionals.
- Increase accessibility in low-resource settings.

---

## Challenges and Benefits
### Challenges
- **Data Quality:** Ensuring all images are high-quality and accurately labeled.
- **Class Imbalance:** Addressing the imbalance in the number of images for each category.
- **Model Overfitting:** Preventing overfitting on the training data.
- **Generalization:** Ensuring the model performs well on unseen data.

### Benefits
- **Efficiency:** Speeds up the diagnostic process.
- **Accessibility:** Provides a cost-effective solution for pneumonia screening.
- **Accuracy:** Reduces human errors in diagnosis.
- **Scalability:** Can be deployed widely in hospitals and remote clinics.

---

## Advantages and Disadvantages
### Advantages
- **High Accuracy:** Leveraging pre-trained models improves detection accuracy.
- **Time-Saving:** Automates the diagnostic process.
- **Cost-Effective:** Reduces dependency on expensive manual labor.
- **User-Friendly:** Simple Flask interface for non-technical users.

### Disadvantages
- **Dependence on Data:** Requires large, high-quality datasets for training.
- **Interpretability:** Deep learning models often act as a "black box" with limited explainability.
- **Hardware Requirements:** Demands significant computational resources.
- **Limited Scope:** May not generalize well to datasets from different demographics.

---

## Preprocessing and Augmentation
The dataset was preprocessed using TensorFlow's `ImageDataGenerator` to include:
- Rescaling pixel values.
- Rotation, flipping, zooming, and shearing for data augmentation.

---

## Model Architecture
The project employs the VGG19 architecture with modifications:
- **Base Model:** Pre-trained VGG19 network with weights from ImageNet.
- **Custom Layers:**
  - Flatten layer
  - Fully connected Dense layers
  - Dropout for regularization
  - Output layer with sigmoid activation for binary classification
- **Optimization Algorithms:**
  - SGD
  - RMSprop
  - Adam
- **Callbacks:**
  - ModelCheckpoint
  - EarlyStopping
  - ReduceLROnPlateau

---

## Implementation
### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/thatritikpatel/Pneumonia-Detection-Using-Deep-Learning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd pneumonia-detection
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
2. Extract the dataset and place it in the `data/` folder.

### Training the Model
Run the training script to train the model:
```bash
python train.py
```

### Running the Flask App
Start the Flask server:
```bash
python app.py
```
Access the web application at `http://127.0.0.1:5000/`.

---

## File Structure
```
.
├── app.py              # Flask application
├── train.py            # Model training script
├── data/               # Dataset folder
├── static/             # Static files (CSS, JS)
├── templates/          # HTML templates
├── models/             # Saved models
├── utils/              # Utility scripts
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

### Visualizations
Confusion matrix, training/validation loss, and accuracy plots are included in the report.

---

## References
- [Original Dataset Publication](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)
- [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2)

---

## License
This project is licensed under the CC BY 4.0 License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements
Special thanks to the creators of the dataset and the research team for their valuable contributions to pneumonia detection.

## Contact
- Avinash Raj Anand
