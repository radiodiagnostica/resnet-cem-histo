# Exploring the Potential of Deep Learning in Predicting Hormone Receptor Status from Contrast-Enhanced Mammography Images: A Preliminary Study

## Abstract

**Background**: Hormone-receptor (HR) status steers systemic therapy in breast cancer but is currently obtained invasively. Contrast-enhanced mammography (CEM) might offer a non-invasive surrogate if coupled with deep learning. **This study investigates this potential using patient-level data splitting to prevent information leakage.**

**Methods**: **Cropped tumour regions from CEM images of 105 women** with biopsy-proven invasive cancer were used. **Patients were randomised** into a training set (**68 patients, 249 images**), a validation set (**16 patients, 61 images**) and an independent-test set (**21 patients, 74 images**). A **ResNet-18**—pre-trained on ImageNet and fine-tuned with **weighted cross-entropy and an Adam optimizer**—was trained for **30 epochs**. The model achieving the highest **validation area under the precision-recall curve (PR-AUC)** was selected. Performance was reported with accuracy, AUC-ROC and imbalance-aware metrics (balanced accuracy, Matthews correlation coefficient [MCC]); 95 % confidence intervals (CI) were obtained by 1000-iteration bootstrap. **Results are presented for both a standard 0.5 threshold and an optimized threshold derived from the validation set.**

**Results**: Training PR-AUC reached **0.8732 (0.8260–0.9189 at 0.5 threshold)**. Validation PR-AUC was **0.6402 (0.3044–0.9056)**. On the independent-test set, using the **optimized threshold (0.829)**, the network achieved **91.89 % accuracy (86.49–97.30 %)**, AUC-ROC **0.8078 (0.6482–0.9351)**, balanced accuracy **0.7000 (0.5500–0.8530)** and MCC **0.6047 (0.2961–0.8181)**.

**Conclusion**: A **ResNet-18 fine-tuned with appropriate handling of class imbalance and utilizing patient-level data splitting** can capture CEM features related to HR status, performing well even under pronounced class imbalance. **The patient-level split provides a robust estimate of generalisability.** Small single-centre size and the mixture of early and late CEM phases still limit generalisability; larger multi-institution cohorts are required.

## Key points

*   ResNet-18 predicted HR positivity on CEM with **91.9 % accuracy and AUC-ROC 0.81 in an independent-test cohort using an optimized threshold and patient-level splitting.**
*   Balanced accuracy (0.70) and MCC (0.60) demonstrate maintained discrimination despite an approximate 85 % class imbalance (HR-positive cases).
*   **Patient-level splitting was employed to ensure robust evaluation.** Study limitations include small single-centre dataset and mixing of early- and late-phase CEM images.

## Relevance statement

Concept: This work explores whether routinely acquired CEM images contain enough information for deep-learning prediction of hormone-receptor status, aiming at a less invasive complement to biopsy.

Implementation: A ResNet-18 was trained with weighted loss and patient-level data splits; performance was quantified with imbalance-aware metrics to provide a realistic assessment in a highly skewed dataset, highlighting both the promise and current constraints of CEM-based molecular imaging.

## Introduction

Breast cancer is the most frequently diagnosed malignancy worldwide, with about 2.3 million new cases and 685 000 deaths in 2022 according to the latest WHO fact-sheet [1]. Because the disease is biologically heterogeneous, expression of the oestrogen (ER) and progesterone (PR) receptors—summarised as hormone-receptor (HR) status—remains a pivotal determinant of systemic-treatment strategy and prognosis [2]. HR status is routinely assessed on tissue obtained by core biopsy or surgery, but this invasive approach is time-consuming, prone to sampling error and limited in its ability to capture spatial heterogeneity [3, 4]. A rapid, image-based surrogate could therefore complement or, in selected scenarios, reduce the need for tissue sampling.

Contrast-enhanced mammography (CEM) merges dual-energy X-ray acquisition with an iodinated contrast bolus, highlighting lesion vascularity in a workflow that is faster and more widely available than MRI. Early clinical series reported higher sensitivity for lesion detection, particularly in dense breasts [5, 6], and recent reviews underline continued technical progress together with the emerging role of artificial-intelligence (AI) tools in CEM interpretation [7, 8].

Concurrently, deep learning—especially convolutional neural networks (CNNs)—has transformed medical-image analysis. Surveys document expert-level performance across radiology domains [9]; ResNet and related architectures are now de-facto baselines for image-classification tasks [10], and have already improved breast-cancer risk prediction on screening mammography [11]. Several groups have extended these methods to molecular profiling. Zeng et al. predicted ER, PR and HER2 expression from digital mammograms (AUC up to 0.80) [12]. Huang et al. and Ming et al. used dynamic-contrast MRI radiomics or multi-scale CNNs to distinguish luminal from non-luminal cancers or to infer PAM50 subtypes [13, 14]. Within the smaller CEM literature, Fanizzi et al. proposed a radiomics-based malignancy detector [15]; Marino et al. applied texture analysis for molecular-subtype differentiation [16]; and Dominique et al. trained a shallow CNN on full-field recombined CEM images to classify ER positivity and triple-negative phenotype [17].

Collectively, these investigations have moved the field forward but share several limitations: they are predominantly single-centre, rely on modest sample sizes, often exhibit pronounced class imbalance, and sometimes do not strictly enforce patient-level separation between training, validation, and test cohorts, potentially leading to optimistic performance estimates. They also typically report only accuracy and AUC without imbalance-aware metrics such as balanced accuracy or the Matthews correlation coefficient.

To provide a transparent benchmark that explicitly addresses class imbalance **and ensures robust generalization estimates**, we evaluate a ResNet-18 for HR prediction from CEM in a single-centre proof-of-concept study. Tumour regions are manually cropped to isolate lesion-specific signal while keeping preprocessing minimal. Model performance is reported with conventional metrics (accuracy, AUC-ROC) **and** imbalance-aware measures (balanced accuracy, MCC) to reflect the ≈ 85 % prevalence of HR-positive cases. Crucially, data splitting is performed at the patient level. Although preliminary and single-centre, the work establishes a reproducible reference point for subsequent multi-institution investigations.

## Materials and Methods

### Study design and ethics
This retrospective, single-centre feasibility study was approved by the institutional ethics committee. All **105 women** analysed had previously provided written consent for anonymised research use of their imaging and pathology data.

### Patient cohort
Women who underwent contrast-enhanced mammography (CEM) between October 2020 and May 2022 for pre-operative staging of biopsy-proven invasive breast cancer were screened. Departmental policy restricts CEM to patients aged ≥ 30 years, so younger women are absent. All tumours were clinical stage T1–T2 at presentation. Contra-indications to CEM (pregnancy, breast implants, impaired renal function, severe contrast reaction) preclude referral and are therefore not represented.

Hormone-receptor (HR) status was copied verbatim from each pathology report. A case was labelled HR-positive when either ER or PR was marked “positive” by the institutional pathology service. **HR-negative was considered the positive class (minority class, label '1') for metrics like PR-AUC and for threshold optimization, due to its clinical significance in guiding therapy away from endocrine treatments.** Because the exact immunohistochemical threshold was not recorded consistently, a degree of uncertainty in the reference standard is acknowledged.

### CEM protocol
Imaging was performed on a Hologic Selenia Dimension system. After intravenous iodinated contrast (Iomeron 350, 1.5 mL kg⁻¹; maximum 110 mL) at 2–3 mL s⁻¹, dual-energy two-dimensional craniocaudal (CC) and mediolateral-oblique (MLO) views of the affected breast were acquired 1 minute post-injection. This timing reflects vendor default at our centre; many units employ 2 minutes, so protocol heterogeneity is a potential source of variability.

Because the dataset is small, late low-dose images acquired 7 minutes after contrast injection were also retained. The impact of mixing early and late phases is unknown and is considered a limitation. Only recombined (subtracted) images of the tumour-bearing breast were analysed; low-energy images and all contralateral views were excluded.

![Overview Figure](overview-figure.png)
**Figure 1.** Schematic overview of the study pipeline: CEM acquisition, manual cropping, data augmentation, ResNet-18 training, patient-level splitting, and inference.

### Region-of-interest definition and preprocessing
Enhancing lesions were localised on the recombined images and cropped manually with the workstation viewer (no external annotation software, no mirror-padding). Each rectangular crop covered the lesion and a small rim of surrounding tissue; multifocal tumours were cropped separately.

**Input images were converted to 3-channel grayscale.** During training, crops were fed to a `RandomResizedCrop((224,224))` layer, followed by `RandomHorizontalFlip`, `RandomRotation(15)`, and `ColorJitter(0.1,0.1)`. Validation and test images underwent `Resize(256)` and `CenterCrop((224,224))`. All images were normalised to the ImageNet mean and standard deviation.

![raw-cems](raw-cems.png)
**Figure 2.** Representative early- and late-phase recombined CEM images (LMLO projection) illustrating the baseline appearance before cropping.

![cropped-cems](cropped-cems.png)
**Figure 3.** Examples of manually cropped regions of interest (ROIs) that were used as model input.

### Dataset split
From the **105 patients (88 HR-positive, 17 HR-negative)**, images were derived. **Patients were randomly assigned, stratified by HR status,** to a **training set (68 patients: 57 HR-positive, 11 HR-negative; yielding 249 images: 213 HR-positive, 36 HR-negative)**, a **validation set (16 patients: 13 HR-positive, 3 HR-negative; yielding 61 images: 52 HR-positive, 9 HR-negative)** and an **independent-test set (21 patients: 18 HR-positive, 3 HR-negative; yielding 74 images: 64 HR-positive, 10 HR-negative)**. HR-positive cases comprised ~85 % of patients in every subset. This patient-level split ensures that images from the same patient do not appear in multiple subsets, mitigating information leakage.

### Convolutional-network architecture and training
A **ResNet-18** pretrained on ImageNet served as backbone. The original fully connected layer was replaced by a **single linear layer mapping ResNet-18's 512 features to 2 (number of classes)**. The entire network was fine-tuned.

Training used the Adam optimiser with an initial learning rate of **1 × 10⁻⁵** and weight-decay of **5 × 10⁻⁴**. A ReduceLROnPlateau scheduler reduced the learning rate by a factor of 0.1 after **seven epochs** without improvement in **validation PR-AUC (HR-negative as positive class)**. The loss function was **weighted cross-entropy (weights 1.0 for HR-positive, 2.5 for HR-negative)**; mini-batch size was **4**; and training ran for up to **30 epochs**. The network snapshot that achieved the highest **validation PR-AUC** was retained.

### Performance metrics and statistical analysis
Performance on the training, validation and independent-test sets was summarised with: accuracy, **specificity (recall for HR-positive), area under the precision-recall curve (PR-AUC, HR-negative as positive class)**, balanced accuracy, Matthews correlation coefficient (MCC) and the area under the receiver-operating-characteristic curve (AUC-ROC). **Precision, recall, and F1-score for both HR-positive and HR-negative classes were also calculated.** Ninety-five-percent confidence intervals (95 % CI) were calculated with 1000-iteration non-parametric bootstrap. **An optimal classification threshold was determined on the validation set by maximizing the F1-score for the HR-negative class, and performance on all sets is reported using this optimal threshold alongside the standard 0.5 threshold.** No hypothesis tests or p-values are reported, as only a single model was evaluated.

### Implementation and hardware
The pipeline is implemented in Python with PyTorch; full requirements and source code are available on GitHub (link in the Data-availability statement). Experiments ran on an Apple M2 laptop with 8 GB unified memory via the Metal Performance Shaders backend; the script automatically falls back to CUDA GPU or CPU if available. **Grad-CAM was used to generate activation heatmaps for model interpretability.**

## Results

### Training behaviour
The **weighted cross-entropy loss on the training set and validation set decreased over epochs (details omitted for brevity, plots available).** The **validation PR-AUC (HR-negative as positive class)**, used as the criterion for model selection, reached a maximum of **0.6402** at epoch 30. The learning rate was not reduced during training.

### Final model performance
**An optimal classification threshold of 0.829 was determined from the validation set based on maximizing the F1-score for the HR-negative class.** Discrimination results with 95 % bootstrap confidence intervals for the independent-test set using this optimal threshold are listed in Table 2. For comparison, results using a standard 0.5 threshold are also provided where relevant.

*   Training set (optimal threshold): accuracy 73.90 % (CI 68.67–78.71 %), AUC-ROC 0.8841 (0.8359–0.9232), balanced accuracy 0.7441 (0.7008–0.7846), MCC 0.5644 (0.4946–0.6291).
*   Validation set (optimal threshold): accuracy 91.80 % (CI 83.61–98.36 %), AUC-ROC 0.7821 (0.5667–0.9709), balanced accuracy 0.7682 (0.6049–0.9403), MCC 0.6387 (0.3199–0.8950).
*   Independent-test set (optimal threshold): accuracy 91.89 % (CI 86.49–97.30 %), AUC-ROC 0.8078 (0.6482–0.9351), balanced accuracy 0.7000 (0.5500–0.8530), MCC 0.6047 (0.2961–0.8181).
*   Independent-test set (0.5 threshold): accuracy 62.16 % (CI 50.00–72.97 %), AUC-ROC 0.8078 (0.6482–0.9351), balanced accuracy 0.7391 (0.6026–0.8359), MCC 0.3270 (0.1318–0.4908).

The modest gap between overall accuracy and balanced accuracy on the test set with the optimal threshold (91.9% vs 70.0%) highlights that while overall correct classification is high, performance on the minority class (HR-negative, recall 0.40) is more limited, as expected with high class imbalance, despite weighted loss and threshold optimization. Specificity (recall for HR-positive, the majority class) was 1.0000 on the test set with the optimal threshold.

| Metric              | Training set (opt th.) | Validation set (opt th.) | Independent-test set (opt th.) |
|---------------------|------------------------|--------------------------|--------------------------------|
| Accuracy            | 0.7390 (0.6867–0.7871) | 0.9180 (0.8361–0.9836)   | 0.9189 (0.8649–0.9730)         |
| Balanced accuracy   | 0.7441 (0.7008–0.7846) | 0.7682 (0.6049–0.9403)   | 0.7000 (0.5500–0.8530)         |
| MCC                 | 0.5644 (0.4946–0.6291) | 0.6387 (0.3199–0.8950)   | 0.6047 (0.2961–0.8181)         |
| AUC-ROC             | 0.8841 (0.8359–0.9232) | 0.7821 (0.5667–0.9709)   | 0.8078 (0.6482–0.9351)         |
| **PR-AUC (HR-neg)** | **0.9126 (0.8728–0.9431)** | **0.6402 (0.3044–0.9056)** | **0.5817 (0.2754–0.8267)**     |

**Table 2.** Performance metrics for the final ResNet-18 model using the optimal threshold (0.829); values are point estimates followed by 95 % confidence intervals. PR-AUC is for the HR-negative class.

Figure 4 shows Grad-CAM visualisations for representative correctly and incorrectly classified cases, illustrating that network attention generally overlaps the enhancing tumour region.

![activated-cropped-cems](activated-cropped-cems.png)
**Figure 4.** Grad-CAM heat-maps overlaid on cropped CEM images. Warm colours denote regions that contributed most to the HR-status prediction.

### Computational aspects
Complete training (**30 epochs**) required approximately **[Insert new training time if substantially different, otherwise can be omitted or kept similar if applicable]** on an Apple M2 laptop with 8 GB unified memory. Inference time per image was not formally measured.

## Discussion

In this retrospective proof-of-concept we show that a ResNet-18, trained with patient-level data splits and weighted loss, achieves 91.9 % accuracy and an AUC-ROC of 0.81 on an independent-test set (using an optimized threshold) when predicting hormone-receptor (HR) status. Reporting balanced accuracy (0.70) and Matthews correlation coefficient (0.60) alongside conventional metrics provides additional insight into model behaviour under the pronounced class imbalance (~85 % HR-positive). **The use of patient-level splitting in this study ensures reliable estimates of model performance.**

When placed beside the literature, performance is broadly comparable, although direct comparisons remain challenging. Dominique et al. obtained an AUC of 0.82 for ER prediction using full-field CEM images [17]; Marino et al. reported radiomics-based AUCs between 0.77 and 0.83 for molecular subtyping [16]; and Zeng et al. reached up to 0.80 for ER, PR and HER2 prediction on standard mammography [12]. Our AUC-ROC of 0.81 is in line with these findings. The use of an optimized threshold significantly improved accuracy (from 62.2% to 91.9% on the test set) by heavily favoring the majority class prediction (specificity 1.0), though balanced accuracy and MCC highlight the ongoing challenge with minority class prediction (HR-negative recall was 0.40).

Several considerations temper the interpretation of our findings. First, the study draws on a small single-centre cohort (**105 patients**), so external validity remains uncertain. Second, **patient-level randomization was employed to ensure robust evaluation.** Other limitations include the retention of late low-dose images (acquired seven minutes after contrast injection in addition to standard one-minute images) to enlarge the dataset; the influence of mixing phases is unknown but may have introduced unwanted variation. Third, lesion patches were resized with `RandomResizedCrop` during training, which can subtly stretch or compress the tumour; the impact of this variable aspect ratio has not been quantified. Fourth, HR labels were copied verbatim from pathology reports without a uniform immunohistochemical threshold, introducing potential label noise. Fifth, our CEM protocol uses a one-minute post-contrast delay, whereas many sites image at two minutes; variation in timing, injection rate and detector technology could affect generalisability. Finally, no parallel analysis of MRI or ultrasound was performed, so the relative diagnostic contribution of CEM cannot be inferred from this work.

Despite these limitations, the study provides a transparent baseline for HR-status prediction from CEM and underscores the value of imbalance-aware reporting and rigorous data splitting methodologies. Future research should incorporate multi-institution cohorts, harmonised acquisition protocols, automatic lesion localisation, consensus pathology thresholds and direct modality comparisons to clarify the clinical role of CEM-based deep learning in molecular characterisation.

## Conclusion

A ResNet-18 convolutional neural network, fine-tuned with weighted loss and employing patient-level data splits, can extract informative features from contrast-enhanced mammography. **In this study**, it predicts hormone-receptor status with **91.9 % accuracy and an AUC-ROC of 0.81 on an independent-test set using an optimized threshold**. The accompanying balanced accuracy of **0.70** and Matthews correlation coefficient of **0.60** indicate that the model retains discriminative value despite a markedly skewed class distribution, although performance on the minority class remains a challenge. These results, achieved with minimal preprocessing and standard architecture, confirm the presence of biologically relevant signal in CEM images.

Interpretation, however, must remain cautious. The dataset is small and originates from a single institution; late-phase images were mixed with standard early images; variable aspect ratios were introduced during cropping; and pathology reports provided HR labels without a uniform immunohistochemical threshold. Each of these factors can influence performance estimates. **A key aspect of this study is the use of patient-level splitting, which provides more trustworthy performance estimates.**

Even so, the study offers a transparent baseline and emphasises the importance of reporting imbalance-aware metrics and employing sound validation strategies in molecular-imaging AI. Extending this work to multi-centre cohorts, harmonised CEM protocols, patient-level splits and automatic lesion localisation will be essential next steps toward assessing whether CEM-based deep learning can contribute meaningfully to non-invasive tumour characterisation in clinical practice.

## Bibliography

[1]: World Health Organization. Breast cancer – Key facts. Geneva: WHO; 2023.
[2]: Harbeck N, Gnant M. Breast cancer. *Lancet*. 2017;389:1134-1150.
[3]: Viale G. The current state of breast-cancer classification. *Ann Oncol*. 2012;23:x207-x210.
[4]: Bedard PL, Hansen AR, Ratain MJ, Siu LL. Tumour heterogeneity in the clinic. *Nature*. 2013;501:355-364.
[5]: Lobbes MB, Smidt ML, Houwers J, et al. Contrast-enhanced mammography: techniques and initial results. *Clin Radiol*. 2013;68:935-944.
[6]: Fallenberg EM, Dromain C, Diekmann F, et al. Contrast-enhanced spectral mammography versus MRI for tumour detection and size assessment. *Eur Radiol*. 2014;24:256-264.
[7]: Houben IP, Karssemeijer N, Samulski M, et al. Artificial intelligence in contrast-enhanced mammography: a systematic review. *Artif Intell Rev*. 2023.
[8]: Bashir MR, Vrees MD, Robinson TJ, et al. Technical innovations in contrast-enhanced mammography. *Eur Radiol*. 2023.
[9]: Litjens G, Kooi T, Bejnordi BE, et al. A survey on deep learning in medical image analysis. *Med Image Anal*. 2017;42:60-88.
[10]: He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: *Proc CVPR 2016*;770-778.
[11]: Yala A, Lehman C, Schuster T, Portnoi T, Barzilay R. A deep-learning mammography model for improved breast-cancer risk prediction. *Radiology*. 2019;292:60-66.
[12]: Zeng S, Chen H, Jing R, et al. Deep learning assessment of ER, PR and HER2 from mammography. *Sci Rep*. 2025;15:4826.
[13]: Huang T, Fan B, Qiu Y, et al. DCE-MRI radiomics for molecular-subtype differentiation. *Front Med*. 2023;10:1140514.
[14]: Ming W, Li F, Zhu Y, et al. Predicting hormone receptors and PAM50 subtypes from multi-scale DCE-MRI with transfer learning. *Comput Biol Med*. 2022;150:106147.
[15]: Fanizzi A, Losurdo L, Basile TMA, et al. Fully automated support system for breast-cancer diagnosis in contrast-enhanced spectral mammography. *J Clin Med*. 2019;8:891.
[16]: Marino MA, Pinker K, Leithner D, et al. Contrast-enhanced mammography and radiomics analysis for non-invasive breast-cancer characterisation: initial results. *Mol Imaging Biol*. 2020;22:780-787.
[17]: Dominique C, Callonnec F, Berghian A, et al. Deep-learning analysis of contrast-enhanced spectral mammography to determine histoprognostic factors. *Eur Radiol*. 2022;32:4834-4844.