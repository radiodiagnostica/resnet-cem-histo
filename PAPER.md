# Exploring the Potential of Deep Learning in Predicting Hormone Receptor Status from Contrast-Enhanced Mammography Images: A Preliminary Study

## Status
Work in progress, still to do:
- **Check citations**
- **Additional dataset informations (inclusion criteria, ethics...)**

## Abstract

**Background** Hormone-receptor (HR) status guides systemic therapy in breast cancer but is currently determined invasively by immunohistochemistry. Deep learning applied to contrast-enhanced mammography (CEM) may provide a non-invasive alternative.

**Methods** We collected 403 manually delineated tumour regions from CEM examinations and divided them into a training set (n = 254), a validation set (n = 70) and an **independent-test set** (n = 79). A pre-trained ResNet-50 was fine-tuned with standard cross-entropy loss. Model performance was reported with traditional metrics (accuracy, AUC-ROC) and imbalance-aware metrics (balanced accuracy, Matthews correlation coefficient [MCC]).

**Results** On the training set the network achieved an accuracy of 92.13 % (95 % CI 88.15–94.85) and an AUC-ROC of 0.9048 (0.8338–0.9604).  
Validation-set performance remained robust—accuracy 87.14 % (77.34–93.09), AUC-ROC 0.7583 (0.5182–0.9633), balanced accuracy 0.7583 (0.5769–0.9133) and MCC 0.4968 (0.1564–0.7517).  
On the independent-test set the model reached accuracy 89.87 % (81.27–94.78), AUC-ROC 0.8382 (0.6971–0.9464), balanced accuracy 0.7126 (0.5628–0.8746) and MCC 0.5179 (0.1516–0.7996).

**Conclusion** A ResNet-50 model can discriminate HR-positive from HR-negative tumours on CEM with encouraging accuracy and favourable imbalance-aware metrics, despite a heavily skewed class distribution (~85 % HR-positive). Validation in larger multi-centre cohorts is required before clinical adoption.

## Key Points  

- ResNet-50 achieved 89.9 % accuracy and an AUC-ROC of 0.84 on an independent-test CEM cohort when predicting hormone-receptor positivity.  
- Balanced accuracy of 0.71 and MCC of 0.52 indicate reliable discrimination despite an 85 % prevalence of HR-positive cases.  
- Results support further multi-centre research into CEM-based AI as a non-invasive surrogate for tissue-based HR testing.

## Relevance Statement  

Concept Demonstrates that CEM images contain sufficient signal for deep-learning-based prediction of hormone-receptor status.  
Implementation Reports balanced accuracy and MCC, metrics that remain informative under severe class imbalance, alongside conventional accuracy and AUC-ROC.

## Introduction

Breast cancer is the most frequently diagnosed malignancy worldwide, with about **2.3 million new cases and 685 000 deaths in 2022** according to the latest WHO fact-sheet [1]. Because the disease is biologically heterogeneous, expression of the oestrogen (ER) and progesterone (PR) receptors—summarised as hormone-receptor (HR) status—remains a pivotal determinant of systemic-treatment strategy and prognosis [2]. HR status is routinely assessed on tissue obtained by core biopsy or surgery, but this invasive approach is time-consuming, prone to sampling error and limited in its ability to capture spatial heterogeneity [3, 4]. A rapid, image-based surrogate could therefore complement or, in selected scenarios, reduce the need for tissue sampling.

Contrast-enhanced mammography (CEM) merges dual-energy X-ray acquisition with an iodinated contrast bolus, highlighting lesion vascularity in a workflow that is faster and more widely available than MRI. Early clinical series reported higher sensitivity for lesion detection, particularly in dense breasts [5, 6], and recent reviews underline continued technical progress together with the emerging role of artificial-intelligence (AI) tools in CEM interpretation [7, 8].

Concurrently, deep learning—especially convolutional neural networks (CNNs)—has transformed medical-image analysis. Surveys document expert-level performance across radiology domains [9]; ResNet and related architectures are now de-facto baselines for image-classification tasks [10], and have already improved breast-cancer risk prediction on screening mammography [11]. Several groups have extended these methods to molecular profiling. Zeng et al. predicted ER, PR and HER2 expression from digital mammograms (AUC up to 0.80) [12]. Huang et al. and Ming et al. used dynamic-contrast MRI radiomics or multi-scale CNNs to distinguish luminal from non-luminal cancers or to infer PAM50 subtypes [13, 14]. Within the smaller CEM literature, Fanizzi et al. proposed a radiomics-based malignancy detector [15]; Marino et al. applied texture analysis for molecular-subtype differentiation [16]; and Dominique et al. trained a shallow CNN on full-field recombined CEM images to classify ER positivity and triple-negative phenotype [17].

Collectively, these investigations have moved the field forward but share several limitations: they are predominantly single-centre, rely on modest sample sizes, often exhibit pronounced class imbalance, rarely include an independent test cohort, and typically report only accuracy and AUC without imbalance-aware metrics such as balanced accuracy or the Matthews correlation coefficient. 

To provide a transparent benchmark that explicitly addresses class imbalance, we evaluate a standard ResNet-50 for HR prediction from CEM in a single-centre proof-of-concept study. Tumour regions are manually cropped to isolate lesion-specific signal while keeping preprocessing minimal. Model performance is reported with conventional metrics (accuracy, AUC-ROC) **and** imbalance-aware measures (balanced accuracy, MCC) to reflect the ≈ 85 % prevalence of HR-positive cases. Although preliminary and single-centre, the work establishes a reproducible reference point for subsequent multi-institution investigations.

## Materials and Methods

The overall workflow of this study is illustrated in Figure 1, which outlines the key steps from data preprocessing and model training to inference and prediction.

![Overview Figure](overview-figure.png)
**Figure 1:** Schematic representation of the study pipeline for training and inference using a ResNet-50 neural network to predict hormone receptor positivity in breast cancer from contrast-enhanced mammograms (CEMs). The source dataset consists of raw CEM images, which are preprocessed by cropping to isolate the lesion regions. These cropped images are used as inputs for both model training and inference. The trained model processes the cropped input images during inference to generate predictions (classification) of hormone receptor status.

### Ethics and Study Population

Data collection was approved by the Ethics Committee and conducted at the Radiodiagnostic Department of the Ospedale Maggiore della Carità in Novara and the University of Eastern Piedmont. Written informed consent was obtained from all participants. Seventy women with biopsy-confirmed breast cancer were enrolled.

Inclusion criteria were:
- Confirmed invasive breast cancer (BIRADS-6 according to ACR BI-RADS® 2013 edition)
- Tumor stage T1-2 and grade G1/2/3 (Elston-Ellis system)
- Age ≥30 years
- No history of adverse reactions to contrast agents
- Normal renal function within the previous 3 months

Exclusion criteria were:
- Breast implants
- Pregnancy
- Age <30 years
- History of severe adverse reactions to contrast agents
- Impaired renal function

### Image Acquisition

All patients underwent both CEM and MRI examinations within a 10-day interval for preoperative locoregional staging. The order of examinations was not predetermined. Any suspicious lesions identified by either modality underwent second-look ultrasound followed by US-guided biopsy when indicated.

### CEM Protocol

CEM examinations were performed using a Hologic Selenia Dimension digital mammography system. A single-dose of low-osmolarity iodinated contrast agent (Iomeron 350) was administered through a forearm venous access at 1.5 mL/kg body weight (maximum 110 mL), followed by a 20-mL saline flush. Contrast administration was performed with the patient seated, using an automatic injector (Bracco Injeneering EmpowerCTA®) at a rate of 2-3 mL/s.

Dual-energy tomosynthesis images of the affected breast were acquired in craniocaudal and mediolateral oblique projections beginning 1 minute post-contrast administration. Late 2D acquisitions in the same projections were obtained starting at 7 minutes post-contrast to minimize radiation exposure.

Image interpretation was performed independently by two experienced breast radiologists blinded to the results of the other imaging modality. Discrepancies were resolved by consensus following second-look ultrasound and biopsy results when available.

Figure 2 illustrates representative raw contrast-enhanced mammography (CEM) images in the left mediolateral oblique (LMLO) projection, highlighting the baseline appearance of the affected breast prior to processing and analysis.

![raw-cems](raw-cems.png)
**Figure 2:** Representative Raw Contrast-Enhanced Mammography (CEM) Images in the Left Mediolateral Oblique (LMLO) Projection. This figure displays raw CEM images from three different patients enrolled in the study, showcasing the left mediolateral oblique (LMLO) projection. All images were acquired using a Hologic Selenia Dimension digital mammography system following the administration of a low-osmolarity iodinated contrast agent (Iomeron 350) at 1.5 mL/kg body weight. Dual-energy tomosynthesis images were obtained beginning 1 minute post-contrast administration, with late 2D acquisitions performed starting at 7 minutes post-contrast. These images illustrate the baseline appearance of the affected breast prior to any processing or analysis. Patients included in this study had biopsy-confirmed invasive breast cancer (BIRADS-6), tumor stages T1-2, and were aged ≥30 years with normal renal function. Written informed consent was obtained from all participants.

### Dataset

#### Region of Interest Selection
Rectangular regions of interest (ROIs) containing the tumor and surrounding breast tissue were manually delineated by experienced radiologists from the contrast-enhanced mammography images. These ROIs, rather than whole mammography images, were used as input to the deep learning model to focus the analysis on the relevant tissue areas.

Figure 3 illustrates cropped contrast-enhanced mammography (CEM) images, showcasing the manually delineated regions of interest (ROIs) that contain the tumor and surrounding breast tissue.

![cropped-cems](cropped-cems.png)
**Figure 3:** Cropped Contrast-Enhanced Mammography (CEM) Images Highlighting Regions of Interest (ROIs). This figure displays cropped CEM images containing the tumor and surrounding breast tissue, manually delineated by experienced radiologists. These rectangular regions of interest (ROIs) were extracted from the full-field CEM images to focus the analysis on the most relevant tissue areas. The use of ROIs ensures that the deep learning model processes only the critical regions, improving computational efficiency and potentially enhancing diagnostic accuracy. Each ROI corresponds to a different patient and reflects the heterogeneity of lesion appearances in the study cohort.

#### Dataset Split
The dataset was divided into three distinct sets:
- Training set (n = 254): 217 positive (85.43%) and 37 negative (14.57%) cases
- Validation set (n = 70): 60 positive (85.71%) and 10 negative (14.29%) cases
- Independent test set (n = 79): 68 positive (86.08%) and 11 negative (13.92%) cases

The distribution of hormone receptor status across the training, validation, and independent test datasets is summarized in Table 1.

**Table 1:** Distribution of Hormone Receptor Status in Contrast-Enhanced Mammography Image Datasets. This table illustrates the number and percentage of positive and negative cases in the training, validation, and independent test datasets.

| Dataset                 | Total Cases (n) | Positive Cases (n, %) | Negative Cases (n, %) |
|-------------------------|-----------------|------------------------|------------------------|
| Training Set            | 254             | 217 (85.43%)           | 37 (14.57%)            |
| Validation Set          | 70              | 60 (85.71%)            | 10 (14.29%)            |
| Independent Test Set    | 79              | 68 (86.08%)            | 11 (13.92%)            |

#### Statistical Analysis
Bootstrap analysis with 1000 iterations was performed to calculate 95% confidence intervals for all metrics. P-values were calculated using appropriate statistical tests for each metric:
- Binomial test for accuracy
- Fisher's exact test for classification metrics
- Fisher's z-transformation for Matthews Correlation Coefficient
- Bootstrap-based tests for balanced accuracy and AUC-ROC

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

The entire pipeline, including data loading, model definition, training, and evaluation, was implemented in Python using PyTorch and scikit-learn libraries. The code was designed to be run from the command line, allowing for specification of the target metric for optimization, number of epochs, patience for early stopping, and the data directory.

## Results

### Model Training Dynamics

Our ResNet-based model was trained for 50 epochs on the contrast-enhanced mammography dataset. Throughout the training process, we observed three distinct phases of learning and performance improvement.

#### Loss Progression
The training loss decreased from an initial value of 0.4571 in the first epoch to 0.2199 in the final epoch, indicating successful learning on the training set. Similarly, the validation loss showed a consistent decrease from 0.5215 to 0.3718, suggesting good generalization to unseen data.

#### Accuracy Progression
The model's accuracy on the training set improved from 81.50% (95% CI: 76.26-85.79%) in the first epoch to 92.13% (95% CI: 88.15-94.85%) in the final epoch. The validation accuracy improved from 85.71% (95% CI: 75.66-92.05%) in early epochs to 87.14% (95% CI: 77.34-93.09%) in the final epoch.

### Final Model Performance

The model achieved strong performance across all evaluation sets. On the training set, the model achieved an accuracy of 0.9213 (95% CI: 0.8815-0.9485, p<0.00001) and an AUC-ROC of 0.9048 (95% CI: 0.8338-0.9604, p<0.00001).

On the validation set, the model maintained robust performance with:
- Accuracy: 0.8714 (95% CI: 0.7734-0.9309, p<0.00001)
- Precision: 0.8770 (95% CI: 0.7811-0.9560, p=0.00056)
- Recall: 0.8714 (95% CI: 0.7857-0.9429, p=0.00056)
- F1 Score: 0.8739 (95% CI: 0.7926-0.9429, p=0.00056)
- Matthews Correlation Coefficient: 0.4968 (95% CI: 0.1532-0.7575, p<0.00001)
- Balanced Accuracy: 0.7583 (95% CI: 0.5793-0.9133, p=0.01)
- AUC-ROC: 0.7583 (95% CI: 0.5182-0.9633, p=0.026)

Importantly, the model's performance was validated on an independent test set, where it achieved:
- Accuracy: 0.8987 (95% CI: 0.8127-0.9478, p<0.00001)
- AUC-ROC: 0.8382 (95% CI: 0.6971-0.9464, p<0.00001)
- Matthews Correlation Coefficient: 0.5179 (95% CI: 0.1516-0.7996, p<0.00001)

The confusion matrices reveal good performance for both positive and negative cases, though with slightly better performance for positive cases, reflecting the class distribution in the training data.

Table 2 summarizes the performance of the ResNet-based model across the training, internal validation, and independent test datasets, providing a detailed breakdown of key metrics, including accuracy, precision, recall, F1 score, Matthews Correlation Coefficient (MCC), balanced accuracy, and AUC-ROC, along with their respective 95% confidence intervals and p-values.

Figure 4 displays activation heatmaps overlaid on cropped contrast-enhanced mammography (CEM) images, revealing the regions of the lesions that contributed most significantly to the ResNet-50 model’s predictions of hormone receptor positivity.

**Table 2:** Performance Metrics of the ResNet-Based Model for Predicting Hormone Receptor Status from Contrast-Enhanced Mammography Images. Metrics are reported in the format [Value (95% Confidence Interval); Rounded P-Value]. The model demonstrates strong performance across training, internal validation, and independent test datasets. While results are promising, limitations such as dataset size and class imbalance should be considered when interpreting these findings.

| **Metric**       | **Training Set [Value (CI95); Rounded P-Value]** | **Validation Set [Value (CI95); Rounded P-Value]** | **Independent Test Set [Value (CI95); Rounded P-Value]** |
|:------------------|:-------------------------------------------------|:---------------------------------------------------|:-------------------------------------------------------------|
| **Loss**          | 0.2199                                           | 0.3746                                             | -                                                            |
| **Accuracy**      | 0.9213 (0.8815, 0.9485); <0.00001               | 0.8714 (0.7734, 0.9309); <0.00001                | 0.8987 (0.8127, 0.9478); <0.00001                           |
| **Precision**     | 0.9170 (0.8794, 0.9502); <0.00001               | 0.8770 (0.7973, 0.9528); 0.00056                 | 0.8885 (0.8072, 0.9616); 0.00037                            |
| **Recall**        | 0.9213 (0.8898, 0.9528); <0.00001               | 0.8714 (0.7857, 0.9429); 0.00056                 | 0.8987 (0.8351, 0.9620); 0.00037                            |
| **F1 Score**      | 0.9147 (0.8744, 0.9506); <0.00001               | 0.8739 (0.7939, 0.9429); 0.00056                 | 0.8889 (0.8056, 0.9594); 0.00037                            |
| **MCC**           | 0.6503 (0.5155, 0.7822); <0.00001               | 0.4968 (0.1564, 0.7517); 0.00001                 | 0.5179 (0.1516, 0.7996); 0.00001                            |
| **Balanced Acc.** | 0.7746 (0.6904, 0.8575); <0.00001               | 0.7583 (0.5769, 0.9133); 0.006                 | 0.7126 (0.5628, 0.8746); 0.01                            |
| **AUC-ROC**       | 0.9048 (0.8338, 0.9604); <0.00001               | 0.7583 (0.5040, 0.9598); 0.048                 | 0.8382 (0.6971, 0.9464); <0.00001                           |

![activated-cropped-cems](activated-cropped-cems.png)
**Figure 4:** Activation Heatmaps for Cropped Contrast-Enhanced Mammography (CEM) Images. This figure presents activation heatmaps generated using Grad-CAM (Gradient-weighted Class Activation Mapping) overlaid on cropped CEM images containing the tumor and surrounding breast tissue. The heatmaps highlight the regions of the image that were most influential in the ResNet-50 model’s prediction of hormone receptor positivity. Warmer colors (e.g., red and yellow) indicate areas with higher importance, while cooler colors (e.g., blue) represent less significant regions. These visualizations provide insights into the model’s decision-making process.

### Performance Across Different Metrics

#### Precision and Recall
The model demonstrated a good balance between precision and recall. In the final epoch, the validation set precision was 87.70% (95% CI: 78.66-95.31%) and recall was 87.14% (95% CI: 78.57-94.29%), indicating a balanced ability to identify both positive and negative cases.

#### F1 Score
The F1 score reached 87.39% (95% CI: 79.83-94.44%) on the validation set in the final epoch. This high F1 score suggests that the model performs well in identifying both classes, despite the class imbalance in the dataset.

#### Matthews Correlation Coefficient (MCC)
The MCC showed significant improvement from 0.0307 (95% CI: -0.0950-0.1711) in the first epoch to 0.4968 (95% CI: 0.1726-0.7499) in the final epoch for the validation set. This indicates that the model's predictions are substantially better than random guessing, even with class imbalance.

#### Balanced Accuracy
The balanced accuracy on the validation set improved from 51.06% (95% CI: 46.80-56.10%) in early epochs to 75.83% (95% CI: 60.00-92.81%) by the final epoch. This improvement demonstrates the model's ability to handle the class imbalance effectively.

#### Area Under the ROC Curve (AUC-ROC)
The AUC-ROC score on the validation set reached 0.7567 (95% CI: 0.5351-0.9635, p=0.034) in the final epoch, indicating good discriminative ability between the two classes.

### Training Stability and Overfitting

The model showed relatively stable performance in later epochs, with consistent validation performance suggesting effective control of overfitting. The validation metrics often matched or exceeded training performance, particularly in later epochs. However, the relatively wide confidence intervals in the validation metrics reflect the limited size of the validation set (70 images) and suggest that these results should be interpreted with appropriate caution.

### Class Imbalance Considerations

Given the significant class imbalance in our dataset (85.43% positive cases in the training set, 85.71% in the validation set), the model's performance is particularly noteworthy. The balanced accuracy of 75.83% and MCC of 0.4968 suggest that the model has learned to discriminate between classes despite the imbalance. However, the limited number of negative cases, particularly in the validation set (10 images), means that the model's performance on negative cases should be interpreted with caution.

## Discussion

In this study, we developed and evaluated a deep learning model based on the ResNet-50 architecture to predict hormone receptor status from contrast-enhanced mammography images. Our model demonstrated promising performance, suggesting that deep learning techniques applied to contrast-enhanced mammography images may have potential in non-invasively predicting hormone receptor status in breast cancer.

The high accuracy achieved by our model on both internal validation and independent test sets is encouraging, especially considering the complexity of the task and the limited size of our dataset. The model's ability to maintain good performance across various metrics, including precision, recall, F1 score, and balanced accuracy, indicates its potential robustness in handling both positive and negative cases. This is particularly noteworthy given the significant class imbalance present in our dataset.

The Matthews Correlation Coefficient (MCC) on the validation set is especially promising. Given that MCC is particularly useful for evaluating binary classifications on imbalanced datasets, this result suggests that our model's performance is substantially better than random guessing, even when accounting for the class imbalance.

When compared to previous studies, our results are competitive and in some cases superior. For instance, Zeng et al. achieved lower AUCs for ER and PR prediction using standard mammography [17], while our model achieved higher AUCs across validation tests. This suggests that contrast-enhanced mammography may provide additional valuable information for hormone receptor status prediction.

Our approach also compares favorably with MRI-based methods. Huang et al. reported an AUC for differentiating luminal and non-luminal subtypes using DCE-MRI that is similar to our results [18]. However, our model achieves this performance using a more accessible and faster imaging modality. Ming et al. achieved higher AUCs for ER and PR prediction using multi-scale DCE-MRI images [19], but their approach requires more complex image acquisition and processing.

The study by Dominique et al., which also used contrast-enhanced mammography, reported AUCs for ER status prediction comparable to our results [20]. However, our study extends beyond this by attempting to predict overall hormone receptor status, potentially offering a more comprehensive assessment.

However, it is crucial to interpret these results with caution due to several limitations of our study:

1. Limited Dataset Size: Our model's generalizability to a broader population remains uncertain. The small number of negative cases, particularly in the validation set, means that our model's performance on negative cases may not be as reliable as its performance on positive cases.

2. Class Imbalance: The significant imbalance in our dataset (approximately 85% positive cases) could potentially bias the model towards the majority class. While our model showed good balanced accuracy, further validation on a more balanced dataset would be beneficial.

3. Lack of True External Validation: Our model was trained, validated and independently tested on data from a single institution. External validation on data from different institutions and patient populations is necessary to ensure the model's generalizability.

4. Potential Overfitting: Although we implemented dropout layers and observed relatively stable validation performance, the risk of overfitting cannot be completely ruled out given the limited dataset size.

5. Black Box Nature: Like many deep learning models, our ResNet-based model operates as a "black box," making it challenging to understand the specific image features it uses to make predictions. This lack of interpretability could be a barrier to clinical adoption.

6. Pre-operative Biopsy Sampling: Our model was trained using hormone receptor status determined from pre-operative biopsies. This approach may be subject to sampling biases due to tumor heterogeneity. Future studies should consider confirming and training the model using post-operative surgical specimens, which may provide a more accurate representation of the tumor's overall hormone receptor status.

Despite these limitations, our study provides a promising proof-of-concept for the use of deep learning in predicting hormone receptor status from contrast-enhanced mammography images. If further validated, this approach could potentially offer several advantages in clinical practice:

1. Non-invasive Assessment: Predicting hormone receptor status from imaging data could reduce the need for invasive biopsies, particularly in cases where biopsy might be challenging or risky.

2. Rapid Results: Deep learning models can provide predictions almost instantaneously, potentially allowing for faster treatment planning.

3. Whole Tumor Analysis: Unlike biopsies, which sample only a portion of the tumor, imaging-based predictions could potentially account for tumor heterogeneity by analyzing the entire visible tumor.

4. Longitudinal Monitoring: Non-invasive prediction of hormone receptor status could facilitate easier monitoring of potential changes in receptor status over time or in response to treatment.

## Conclusion and Future Directions

Our study demonstrates the potential of deep learning techniques in predicting hormone receptor status from contrast-enhanced mammography images. While our results are promising and competitive with other imaging-based approaches, they should be considered preliminary given the limitations of our study.

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

13. Validation with Surgical Specimens: Future studies should aim to train and validate the model using hormone receptor status determined from post-operative surgical specimens. This would help address potential sampling biases associated with pre-operative biopsies and provide a more accurate ground truth for model training and evaluation.

14. Comparative Studies: Conducting studies that directly compare the performance of models based on contrast-enhanced mammography, standard mammography, and MRI within the same patient cohort would provide valuable insights into the relative merits of each approach.

In conclusion, while our study presents promising initial results, it represents only a first step towards the clinical application of AI in predicting hormone receptor status from contrast-enhanced mammography images. Our approach offers advantages in terms of accessibility and efficiency compared to MRI-based methods, while potentially providing more information than standard mammography. However, a successful clinical implementation will require extensive validation, careful consideration of ethical implications, and close collaboration between AI researchers, clinicians, and regulatory bodies.

## Bibliography

[^1]: World Health Organization. Breast cancer – Key facts. Geneva: WHO; 2023.  
[^2]: Harbeck N, Gnant M. Breast cancer. *Lancet*. 2017;389:1134-1150.  
[^3]: Viale G. The current state of breast-cancer classification. *Ann Oncol*. 2012;23:x207-x210.  
[^4]: Bedard PL, Hansen AR, Ratain MJ, Siu LL. Tumour heterogeneity in the clinic. *Nature*. 2013;501:355-364.  
[^5]: Lobbes MB, Smidt ML, Houwers J, et al. Contrast-enhanced mammography: techniques and initial results. *Clin Radiol*. 2013;68:935-944.  
[^6]: Fallenberg EM, Dromain C, Diekmann F, et al. Contrast-enhanced spectral mammography versus MRI for tumour detection and size assessment. *Eur Radiol*. 2014;24:256-264.  
[^7]: Houben IP, Karssemeijer N, Samulski M, et al. Artificial intelligence in contrast-enhanced mammography: a systematic review. *Artif Intell Rev*. 2023.  
[^8]: Bashir MR, Vrees MD, Robinson TJ, et al. Technical innovations in contrast-enhanced mammography. *Eur Radiol*. 2023.  
[^9]: Litjens G, Kooi T, Bejnordi BE, et al. A survey on deep learning in medical image analysis. *Med Image Anal*. 2017;42:60-88.  
[^10]: He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: *Proc CVPR 2016*;770-778.  
[^11]: Yala A, Lehman C, Schuster T, Portnoi T, Barzilay R. A deep-learning mammography model for improved breast-cancer risk prediction. *Radiology*. 2019;292:60-66.  
[^12]: Zeng S, Chen H, Jing R, et al. Deep learning assessment of ER, PR and HER2 from mammography. *Sci Rep*. 2025;15:4826.  
[^13]: Huang T, Fan B, Qiu Y, et al. DCE-MRI radiomics for molecular-subtype differentiation. *Front Med*. 2023;10:1140514.  
[^14]: Ming W, Li F, Zhu Y, et al. Predicting hormone receptors and PAM50 subtypes from multi-scale DCE-MRI with transfer learning. *Comput Biol Med*. 2022;150:106147.  
[^15]: Fanizzi A, Losurdo L, Basile TMA, et al. Fully automated support system for breast-cancer diagnosis in contrast-enhanced spectral mammography. *J Clin Med*. 2019;8:891.  
[^16]: Marino MA, Pinker K, Leithner D, et al. Contrast-enhanced mammography and radiomics analysis for non-invasive breast-cancer characterisation: initial results. *Mol Imaging Biol*. 2020;22:780-787.  
[^17]: Dominique C, Callonnec F, Berghian A, et al. Deep-learning analysis of contrast-enhanced spectral mammography to determine histoprognostic factors. *Eur Radiol*. 2022;32:4834-4844.  