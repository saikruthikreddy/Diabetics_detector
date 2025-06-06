# Diabetics-Predictor
This project aims to classify retinal fundus images to detect Diabetic Retinopathy (DR) using a combination of TensorFlow/Keras for preprocessing and FastAI with ResNet18 for training a deep learning model.

📌 Project Overview
Classifies images into:

No_DR (No Diabetic Retinopathy)

DR (Any level of DR: Mild, Moderate, Severe, Proliferative)

Uses Gaussian-filtered retinal images for better feature extraction.

Trained using a ResNet18 model with FastAI.

📂 Dataset
train.csv:

Contains id_code and diagnosis values (0–4).

Diagnosis labels are mapped:

Binary: No_DR (0) and DR (1–4)

Multi-class (for future use): Mild, Moderate, Severe, Proliferative_DR

Images are stored in:

gaussian_filtered_images/gaussian_filtered_images/<label>/<id_code>.png

⚙️ Workflow Summary
Load and process CSV

Map diagnosis to binary and multi-class categories.

Train/Test Split

Use stratified split to preserve class balance.

Folder Setup

Copy images into structured folders:
train/No_DR, train/DR, test/No_DR, test/DR.

Image Preprocessing

Use ImageDataGenerator from TensorFlow to rescale images.

Model Training with FastAI

Load images using FastAI DataBlock.

Resize to 128x128 and apply transformations.

Use vision_learner() with pretrained resnet18.

Train for 8 epochs using fine_tune().

Evaluation

Plot confusion matrix and top loss images.

Save model as stage-1_resnet.pkl.

Testing

Create a test dataloader with FastAI.

Evaluate performance using learn.validate().

Inference on New Image

Use learn.predict(image) to get class and confidence.
