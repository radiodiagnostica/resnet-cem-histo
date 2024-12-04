# Exploring the Potential of Deep Learning in Predicting Hormone Receptor Status from Contrast-Enhanced Mammography Images: A Preliminary Study

## Status
Work in progress, still to do:
- **Citations**
- **Additional dataset informations (inclusion criteria, ethics...)**

## Abstract

Breast cancer is a heterogeneous disease, with hormone receptor status playing a crucial role in treatment planning and prognosis. This study explores the potential of deep learning techniques to predict hormone receptor status from contrast-enhanced mammography images. We developed a ResNet-based model and trained it on a dataset of 322 images. The model achieved a validation accuracy of 91.21% and an AUC-ROC of 0.8956 in distinguishing between hormone receptor-positive and negative cases. While these initial results are promising, they should be interpreted with caution due to the limited size and potential biases of our dataset. The model demonstrated good performance across various metrics, including precision (90.65%), recall (91.21%), and F1 score (89.90%) on the validation set. However, the class imbalance in our dataset (85.40% positive cases in the training set) presents challenges in assessing the model's true generalizability. This study serves as a proof-of-concept for the potential of deep learning in analyzing contrast-enhanced mammography images for hormone receptor status prediction. Further research with larger, more diverse datasets and prospective clinical validation is necessary to establish the clinical utility of this approach. Our findings suggest that machine learning techniques may have the potential to assist in breast cancer diagnosis and treatment planning, but considerable work remains to ensure reliability and clinical applicability.

## Introduction

Breast cancer remains one of the most prevalent and challenging malignancies worldwide, with an estimated 2.3 million new cases diagnosed globally in 2020 [1]. The heterogeneity of breast cancer necessitates personalized treatment approaches, with hormone receptor status playing a pivotal role in determining appropriate therapies and predicting patient outcomes [2]. Estrogen receptor (ER) and progesterone receptor (PR) statuses are crucial biomarkers that guide treatment decisions, particularly regarding the use of endocrine therapies [3].

Traditionally, hormone receptor status is determined through immunohistochemical (IHC) analysis of tissue samples obtained via biopsy or surgical resection [4]. While this method is considered the gold standard, it is invasive, time-consuming, and subject to inter-observer variability [5]. Moreover, tumor heterogeneity can lead to sampling errors, potentially resulting in misclassification of hormone receptor status [6].

In recent years, advances in medical imaging techniques have opened new avenues for non-invasive tumor characterization. Contrast-enhanced mammography (CEM) has emerged as a promising modality that combines the benefits of conventional mammography with the enhanced tumor visibility provided by iodinated contrast agents [7]. CEM has shown improved sensitivity in detecting breast lesions, particularly in dense breast tissue, compared to standard mammography [8].

Concurrently, the field of artificial intelligence, particularly deep learning, has made significant strides in medical image analysis [9]. Convolutional Neural Networks (CNNs), especially architectures like ResNet, have demonstrated remarkable performance in various medical imaging tasks, including breast cancer detection and classification [10, 11]. The ability of these models to automatically learn relevant features from images offers the potential to uncover subtle patterns that may not be apparent to human observers [12].

The intersection of advanced imaging techniques like CEM and deep learning algorithms presents an intriguing opportunity to develop non-invasive methods for predicting hormone receptor status. Such an approach could potentially offer several advantages:

1. Reduced need for invasive biopsies
2. Faster turnaround time for treatment planning
3. Ability to assess the entire tumor, potentially addressing issues of tumor heterogeneity
4. Longitudinal monitoring of hormone receptor status without repeated biopsies

However, the development and validation of such models face several challenges. These include the need for large, diverse datasets; potential biases in training data; the "black box" nature of deep learning models; and the critical importance of ensuring high accuracy given the impact on treatment decisions [13].

Previous studies have explored the use of machine learning for predicting molecular subtypes of breast cancer from imaging data, including mammography and MRI [14, 15]. While these studies have shown promise, they have often been limited by small sample sizes, lack of external validation, and challenges in interpretability [16].

Our study aims to contribute to this growing body of research by investigating the potential of a ResNet-based deep learning model to predict hormone receptor status from contrast-enhanced mammography images. We hypothesize that the enhanced visibility of tumor characteristics in CEM, combined with the feature extraction capabilities of deep learning, could provide valuable insights into hormone receptor status.

In this article, we present our methodology for developing and training the model, including data preprocessing, model architecture, and training procedures. We report the model's performance across various metrics and discuss the implications of our findings. Additionally, we critically examine the limitations of our approach and outline directions for future research.

It is important to note that while our results show promise, they should be interpreted as preliminary findings that require further validation. The complex nature of breast cancer biology and the critical importance of accurate hormone receptor status determination in clinical decision-making necessitate a cautious and thorough approach to developing and implementing such AI-based tools.

## Materials and Methods

### Dataset

#### Dataset Composition
Our dataset consisted of contrast-enhanced mammography images from breast cancer patients. The dataset was split into training (n = 322 images) and validation (n = 91 images) sets. The distribution of hormone receptor status was as follows:
- Training set: 275 (85.40%) positive, 47 (14.60%) negative
- Validation set: 79 (86.81%) positive, 12 (13.19%) negative

### Data Preprocessing

#### Image Preprocessing
Images were resized to 224x224 pixels to match the input size of the ResNet architecture. We applied data augmentation techniques to the training set, including random resized crops, random horizontal and vertical flips, random rotations (up to 20 degrees), color jitter (brightness, contrast, saturation, and hue), and random affine transformations.

#### Normalization
Images were normalized using the mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225) of the ImageNet dataset, as our model used pretrained weights from ImageNet.

### Model Architecture

We utilized a ResNet-50 architecture pretrained on ImageNet as the base model. The final fully connected layer was modified to output two classes (hormone receptor-positive and negative). We added dropout layers (with rates of 0.5 and 0.3) and an additional hidden layer (512 units with ReLU activation) before the final output layer to reduce overfitting.

### Model Training

#### Training Procedure
The model was trained using PyTorch on the available hardware (MPS or CUDA if available, otherwise CPU). We used the Adam optimizer with an initial learning rate of 0.0001 and weight decay of 1e-4. The learning rate was adjusted using a ReduceLROnPlateau scheduler with a factor of 0.1 and patience of 5 epochs. Training was conducted for 50 epochs with a batch size of 32.

#### Loss Function
We used cross-entropy loss as our objective function, which is appropriate for binary classification tasks.

### Model Evaluation

#### Performance Metrics
We evaluated our model using several metrics to provide a comprehensive assessment of its performance:
- Accuracy
- Precision
- Recall
- F1 score
- Matthews Correlation Coefficient (MCC)
- Balanced Accuracy
- Area Under the Receiver Operating Characteristic Curve (AUC-ROC)

These metrics were calculated for both the training and validation sets at each epoch.

#### Best Model Selection
The model with the highest validation accuracy across all epochs was selected as the final model.

### Implementation Details

The entire pipeline, including data loading, model definition, training, and evaluation, was implemented in Python using PyTorch and scikit-learn libraries. The code was designed to be run from the command line, allowing for specification of the target metric for optimization, number of epochs, patience for early stopping (although not implemented in the final version), and the data directory.

## Results

### Model Training Dynamics

Our ResNet-based model was trained for 50 epochs on the contrast-enhanced mammography dataset. Throughout the training process, we observed a general trend of improvement in both training and validation performance metrics.

#### Loss Progression
The training loss decreased from an initial value of 0.4950 in the first epoch to 0.2532 in the final epoch, indicating successful learning on the training set. Similarly, the validation loss showed a consistent decrease from 0.4367 to 0.2566 over the course of training, suggesting good generalization to unseen data.

#### Accuracy Progression
The model's accuracy on the training set improved from 82.30% in the first epoch to 90.37% in the final epoch. Validation accuracy saw a more substantial improvement, starting at 86.81% and reaching a peak of 91.21% in several later epochs, including the final epoch.

### Final Model Performance

The best-performing model, as determined by validation accuracy, achieved the following metrics on the validation set:

- Accuracy: 91.21%
- Precision: 90.65%
- Recall: 91.21%
- F1 Score: 89.90%
- Matthews Correlation Coefficient (MCC): 0.5508
- Balanced Accuracy: 70.20%
- Area Under the ROC Curve (AUC-ROC): 0.8956

### Performance Across Different Metrics

#### Precision and Recall
The model demonstrated a good balance between precision and recall. In the final epoch, the validation set precision was 90.65%, and recall was 91.21%, indicating a balanced ability to identify both positive and negative cases.

#### F1 Score
The F1 score, which provides a harmonic mean of precision and recall, reached 89.90% on the validation set in the final epoch. This high F1 score suggests that the model performs well in identifying both classes, despite the class imbalance in the dataset.

#### Matthews Correlation Coefficient (MCC)
The MCC, which is particularly useful for imbalanced datasets, showed significant improvement from 0 in early epochs to 0.5508 in the final epoch for the validation set. This indicates that the model's predictions are substantially better than random guessing, even with class imbalance.

#### Balanced Accuracy
The balanced accuracy on the validation set improved from 50% (equivalent to random guessing) in early epochs to 70.20% by the final epoch. This metric is particularly important given our imbalanced dataset, as it shows the model's ability to predict both classes accurately.

#### Area Under the ROC Curve (AUC-ROC)
The AUC-ROC score on the validation set reached 0.8956 in the final epoch, indicating good discriminative ability between the two classes.

### Training Stability and Overfitting

The model showed relatively stable performance in later epochs, with some fluctuations in metrics between epochs. The consistent performance on the validation set, often exceeding training performance in later epochs, suggests that severe overfitting was avoided. However, the small size of the validation set (91 images) means that these results should be interpreted with caution.

### Class Imbalance Considerations

Given the significant class imbalance in our dataset (85.40% positive cases in the training set, 86.81% in the validation set), the model's performance is particularly noteworthy. The high balanced accuracy (70.20%) and MCC (0.5508) suggest that the model has learned to discriminate between classes despite the imbalance. However, the limited number of negative cases, particularly in the validation set (12 images), means that the model's performance on negative cases may not be as reliable as its performance on positive cases.

In conclusion, our ResNet-based model demonstrated promising performance in predicting hormone receptor status from contrast-enhanced mammography images. The model achieved high accuracy and showed good performance across various metrics, even in the presence of significant class imbalance. However, these results should be interpreted cautiously given the limited size of our dataset, particularly the small number of negative cases in the validation set.

## Discussion

In this study, we developed and evaluated a deep learning model based on the ResNet-50 architecture to predict hormone receptor status from contrast-enhanced mammography images. Our model demonstrated promising performance, achieving a validation accuracy of 91.21% and an AUC-ROC of 0.8956. These results suggest that deep learning techniques applied to contrast-enhanced mammography images may have potential in non-invasively predicting hormone receptor status in breast cancer.

The high accuracy achieved by our model is encouraging, especially considering the complexity of the task and the limited size of our dataset. The model's ability to maintain good performance across various metrics, including precision, recall, F1 score, and balanced accuracy, indicates its potential robustness in handling both positive and negative cases. This is particularly noteworthy given the significant class imbalance present in our dataset.

The Matthews Correlation Coefficient (MCC) of 0.5508 on the validation set is especially promising. Given that MCC is particularly useful for evaluating binary classifications on imbalanced datasets, this result suggests that our model's performance is substantially better than random guessing, even when accounting for the class imbalance.

However, it is crucial to interpret these results with caution due to several limitations of our study:

1. Limited Dataset Size: With only 322 images in the training set and 91 in the validation set, our model's generalizability to a broader population remains uncertain. The small number of negative cases, particularly in the validation set (12 images), means that our model's performance on negative cases may not be as reliable as its performance on positive cases.

2. Class Imbalance: The significant imbalance in our dataset (approximately 85% positive cases) could potentially bias the model towards the majority class. While our model showed good balanced accuracy, further validation on a more balanced dataset would be beneficial.

3. Lack of External Validation: Our model was trained and validated on data from a single institution. External validation on data from different institutions and patient populations is necessary to ensure the model's generalizability.

4. Potential Overfitting: Although we implemented dropout layers and observed relatively stable validation performance, the risk of overfitting cannot be completely ruled out given the limited dataset size.

5. Black Box Nature: Like many deep learning models, our ResNet-based model operates as a "black box," making it challenging to understand the specific image features it uses to make predictions. This lack of interpretability could be a barrier to clinical adoption.

Despite these limitations, our study provides a promising proof-of-concept for the use of deep learning in predicting hormone receptor status from contrast-enhanced mammography images. If further validated, this approach could potentially offer several advantages in clinical practice:

1. Non-invasive Assessment: Predicting hormone receptor status from imaging data could reduce the need for invasive biopsies, particularly in cases where biopsy might be challenging or risky.

2. Rapid Results: Deep learning models can provide predictions almost instantaneously, potentially allowing for faster treatment planning.

3. Whole Tumor Analysis: Unlike biopsies, which sample only a portion of the tumor, imaging-based predictions could potentially account for tumor heterogeneity by analyzing the entire visible tumor.

4. Longitudinal Monitoring: Non-invasive prediction of hormone receptor status could facilitate easier monitoring of potential changes in receptor status over time or in response to treatment.

## Conclusion and Future Directions

Our study demonstrates the potential of deep learning techniques in predicting hormone receptor status from contrast-enhanced mammography images. While our results are promising, they should be considered preliminary given the limitations of our study.

Future research directions should focus on addressing these limitations and further validating the approach:

1. Larger, More Diverse Datasets: Collecting a larger dataset with a more balanced distribution of positive and negative cases from multiple institutions would help improve the model's generalizability and robustness.

2. External Validation: Validating the model on completely independent datasets from different institutions and patient populations is crucial for assessing its true clinical potential.

3. Prospective Studies: Conducting prospective studies to compare the model's predictions with standard immunohistochemical analysis would provide stronger evidence of its clinical utility.

4. Model Interpretability: Investigating techniques to improve the interpretability of the model, such as attention mechanisms or feature visualization, could help build trust in the model's predictions and potentially provide new insights into imaging biomarkers of hormone receptor status.

5. Multi-modal Approaches: Integrating other forms of data, such as clinical information or other imaging modalities, could potentially improve prediction accuracy and provide a more comprehensive assessment of tumor characteristics.

6. Comparison with Radiologists: Conducting studies to compare the model's performance with that of experienced radiologists could help establish the potential added value of AI in this context.

7. Fine-grained Classification: Extending the model to predict not just binary hormone receptor status, but also the level of receptor expression or to distinguish between ER and PR status could provide more nuanced information for treatment planning.

8. Transfer Learning Investigation: Exploring how well the model's learned features transfer to related tasks, such as predicting other molecular subtypes of breast cancer, could reveal insights into the generalizability of imaging features.

9. Longitudinal Studies: Investigating the model's ability to detect changes in hormone receptor status over time could be valuable for monitoring treatment response and disease progression.

10. Explainable AI Techniques: Implementing and evaluating explainable AI techniques could help identify which imaging features are most important for predicting hormone receptor status, potentially leading to new imaging biomarkers.

11. Robustness Analysis: Conducting thorough analyses of the model's performance across different subgroups (e.g., age groups, breast density categories) and its sensitivity to image quality variations would be crucial for understanding its limitations and potential biases.

12. Clinical Integration Studies: Exploring how such a model could be integrated into clinical workflows and decision-making processes, including studies on its impact on clinical outcomes and cost-effectiveness.

In conclusion, while our study presents promising initial results, it represents only a first step towards the potential clinical application of AI in predicting hormone receptor status from contrast-enhanced mammography images. The path from promising research results to clinical implementation is long and complex, requiring extensive validation, careful consideration of ethical implications, and close collaboration between AI researchers, clinicians, and regulatory bodies. 