# IELTS Writing Score Prediction

This project is an **end-to-end multi-class classification model** to predict IELTS Writing scores using **TensorFlow** and **Keras**. The model can take an IELTS question and essay as input and predict the writing score, with visualization of probability distributions and evaluation metrics.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)

---

## Project Overview

The goal of this project is to build a **machine learning pipeline** that predicts IELTS Writing scores (1.0–9.0) from essay text. The model is designed to handle essays and considers the natural imbalance in the dataset using **class weighting**. The model provides both discrete predictions and probability distributions for each possible score.

---

## Dataset

The dataset used in this project comes from Kaggle: [IELTS Writing Scored Essays Dataset](https://www.kaggle.com/datasets/mazlumi/ielts-writing-scored-essays-dataset/data).

It contains the following columns:

- `Task_Type`: 1 or 2
- `Question`: The essay prompt
- `Essay`: The student’s essay
- `Overall`: IELTS writing score (1.0–9.0)
- Other columns like `Examiner_Comment` (not used in the model)

For this project, we only use **Task 2 essays**.

---

## Features

- Combined feature: `text = Question + Essay`
- Cleaned text: lowercased, punctuation removed, digits removed
- Target: `Overall` IELTS score

---

## Preprocessing

- Text cleaning: remove punctuation and numbers
- Train/validation/test split:
  - 60% training
  - 20% validation
  - 20% test
- Label encoding for IELTS scores
- Tokenization and padding sequences for LSTM input
- Handling class imbalance with **class weights**

---

## Model Architecture

- **Embedding layer**: 100-dimensional embeddings
- **LSTM layer**: 128 units
- **Dropout layer**: 0.3
- **Dense layer**: output layer with softmax activation over all IELTS scores

Custom metric: **±0.5 accuracy**, which considers predictions within 0.5 of the actual score as correct.

---

## Training

- Loss function: `sparse_categorical_crossentropy`
- Optimizer: `adam`
- Batch size: 32
- Early stopping on `val_loss` with patience 3
- Trained for up to 20 epochs (stopped early if no improvement)

---

## Evaluation

- **Metrics**:
  - Accuracy
  - ±0.5 accuracy
- **Test set ±0.5 accuracy**: ~52%
- Visualizations:
  - Histogram of predicted vs actual scores
  - Confusion matrix

---

## Prediction

You can predict the IELTS score for any **question + essay** pair using the `predict_ielts_score` function:

```python
from predict import predict_ielts_score

question = "Some people think students in primary or secondary school should be taught how to manage money."
answer = "Many commentators argue that schools should integrate financial management into their curriculum..."

predicted_score, probabilities = predict_ielts_score(question, answer, actual_score=6.0)
