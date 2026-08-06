DeepFER: Facial Emotion Recognition Using Deep Learning

A Convolutional Neural Network (CNN) that classifies facial expressions into 7 emotion categories: angry, disgust, fear, happy, neutral, sad, surprise. Built and trained on the FER2013 dataset, with an interactive web demo for real-time predictions.

-Demo


Upload a face photo and the model predicts the emotion with confidence scores across all 7 classes.

-Project Overview

Facial emotion recognition has applications in human-computer interaction, mental health monitoring, customer sentiment analysis, and accessibility tools. This project builds a CNN from scratch using transfer-learning-informed design principles to classify facial expressions from grayscale images.

Dataset

FER2013 — 35,887 grayscale 48x48 pixel facial images labeled across 7 emotion classes.

Train set: 28,709 images
Test set: 7,178 images

The dataset is imbalanced — "disgust" has significantly fewer samples than classes like "happy" or "neutral," which was addressed using class-weighted loss during training.

Model Architecture

A 4-block CNN with BatchNormalization and Dropout applied consistently at every stage to stabilize training and reduce overfitting:

4x (Conv2D → BatchNormalization → MaxPooling2D → Dropout) blocks, filter sizes 32 → 64 → 128 → 256
Flatten → Dense(512) → BatchNormalization → Dropout(0.5) → Dense(7, softmax)
Optimizer: Adam (learning rate 0.001, with ReduceLROnPlateau)
Loss: categorical cross-entropy
Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
Data Augmentation

To improve generalization on a relatively small, imbalanced dataset, training images were augmented with:

Rotation (15°)
Width/height shift (10%)
Shear (10%)
Zoom (up to 30%)
Horizontal flip
Results
Best validation accuracy: ~60%
Test set accuracy evaluated on the full 7,178-image test set (not a single batch)
For reference: random guessing on 7 classes is ~14%, and human-level agreement on FER2013 is estimated at ~65-70% due to inherent label ambiguity in the dataset
Confusion Matrix

See confusion_matrix.png for the full per-class breakdown. Key observations:

Strongest classes: happy, surprise, neutral (clear, distinct visual features)
Weakest classes: fear and sad, frequently confused with neutral or each other — a well-documented difficulty in FER2013 due to subtle visual differences between these expressions
Development Journey

This project went through several rounds of debugging that meaningfully improved accuracy from ~24% to ~60%:

Class label mismatch — hardcoded emotion labels didn't match Keras' internal alphabetical class-to-index mapping, causing predictions to display as the wrong emotion. Fixed by pulling labels directly from train_generator.class_indices.
Path mismatch bug — image count paths pointed to a different location than the actual dataset, silently setting steps_per_epoch to 0 and preventing real training from happening.
Architecture bottleneck — aggressive downsampling without padding collapsed the feature map to 1x1 before the dense layers, destroying spatial information critical for distinguishing subtle expressions. Fixed with padding='same' throughout.
Inconsistent regularization — BatchNormalization and Dropout were only applied to some layers. Fixed by applying both consistently across every convolutional block.
Learning rate tuning — an overly conservative learning rate (0.0001) slowed early training significantly; raising it to 0.001 combined with ReduceLROnPlateau allowed faster, more stable convergence.
-Tech Stack
Python, TensorFlow / Keras
NumPy, Matplotlib, Seaborn, scikit-learn
Streamlit 
Project Structure
DeepFER/
├── DeepFER.ipynb              # Main training notebook
├── Emotion_detection_model.h5 # Trained model
├── streamlit_app.py           # Web demo (Streamlit)
├── gradio_app.py              # Web demo (Gradio)
├── requirements.txt
└── README.md
