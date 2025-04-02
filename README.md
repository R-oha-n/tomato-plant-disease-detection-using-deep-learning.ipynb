# Tomato-plant-disease-detection.ipynb
In this project, I have designed an AI-powered system for disease detection in the leaves of tomato plants based on deep learning principles. My tasks included data preprocessing, such as resizing images, normalization, and augmentation, for enhancing model generalization. We were involved in training models utilizing CNN architectures like DenseNet121 and assisted with optimizing the classification model to perform 
at high levels of accuracy. Moreover, I helped in performance assessment with metrics such as accuracy, precision, recall, and confusion matrix analysis to ensure sound disease detection and reduce overfitting.

Methodology 
Data Collection & Preprocessing
1. The dataset was collected from publicly available sources that have images of healthy and diseased plant 
leaves.
2. Image Augmentation methods like rotation, flipping, contrast normalization, and zooming were used to 
enhance dataset diversity.
3. Resizing and Normalization: All the images were resized into a standardized size (e.g., 224x224 pixels) and 
pixel values were normalized for uniformity.
Algorithm Selection & Model Development
1. Pre-trained deep learning models were employed, such as DenseNet121, EfficientNetB4, Xception, VGG16, 
and VGG19.
2. Transfer learning was employed, with pre-trained weights utilized and final layers fine-tuned for disease 
classification.
3. The model architecture consisted of:
 Convolutional Layers for feature extraction.
 Batch Normalization & Dropout to minimize overfitting.
 Global Average Pooling to maximize parameter efficiency.
4. Dense Fully Connected Layers with SoftMax activation for multi-class classification.
Model Training & Optimization
1. The data was divided into training (80%) and validation (20%) sets through train-test split.
2. The model was trained with:
 Optimizer: Adam (adaptive learning rate)
 Loss Function: Categorical Cross-Entropy
 Evaluation Metrics: Accuracy, Precision, Recall, and F1-score
3. Training was performed with Early Stopping to avoid overfitting.
Model Testing & Performance Evaluation
Performance was measured with:
1. Confusion Matrix to examine misclassification rates.
2. Precision, Recall, and F1-score to measure classification accuracy.
3. ROC-AUC Curve to quantify model robustness.
Deployment & Future Enhancements
1. The learned model can be utilized as a web or mobile app for in-time disease diagnosis.
2. Potential future updates involve real-time edge AI models to be utilized on IoT or drones for site monitoring.
Algorithms Utilized:
1. Convolutional Neural Networks (CNNs) – Utilized for leaf image feature extraction.
2. Transfer Learning – Trained models (DenseNet121, EfficientNetB4, etc.) were fine-tuned.
3. Image Augmentation – Implemented to artificially boost the dataset for enhanced model generalization.
4. Adam Optimizer – Applied for adaptive and quick learning rate updates.
5. SoftMax Activation Function – Applied in the output layer for multi-class classification.
4.2 Testing OR Verification Plan
Tomato Plant Disease Detection Using Deep Learning - CNN does not have an explicit Testing or Verification 
Plan section. Nevertheless, the model performance evaluation and validation steps act as the verification criteria 
for the project.
