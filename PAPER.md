# Exploring the Potential of Deep Learning in Predicting Hormone Receptor Status from Contrast-Enhanced Mammography Images: A Preliminary Study

## Abstract  

**Background**: Hormone-receptor (HR) status steers systemic therapy in breast cancer but is currently obtained invasively. Contrast-enhanced mammography (CEM) might offer a non-invasive surrogate if coupled with deep learning.

**Methods**: 403 tumour regions were manually cropped from early (1-min) and late (7-min) recombined CEM images of 70 women with biopsy-proven invasive cancer. Crops were randomised at image level into a training set (n = 254), a validation set (n = 70) and an independent-test set (n = 79). A ResNet-50—pre-trained on ImageNet and fine-tuned with standard cross-entropy—was trained for 50 epochs. Performance was reported with accuracy, AUC-ROC and imbalance-aware metrics (balanced accuracy, Matthews correlation coefficient [MCC]); 95 % confidence intervals (CI) were obtained by 1 000-iteration bootstrap.

**Results**: Training accuracy reached 92.13 % (95 % CI 88.15–94.85) with an AUC-ROC of 0.9048 (0.8338–0.9604). Validation accuracy was 87.14 % (77.34–93.09); balanced accuracy 0.7583 (0.5793–0.9133); MCC 0.4968 (0.1532–0.7575). On the independent-test set the network achieved 89.87 % accuracy (81.27–94.78), AUC-ROC 0.8382 (0.6971–0.9464), balanced accuracy 0.7126 (0.5628–0.8746) and MCC 0.5179 (0.1516–0.7996).

**Conclusion**: A conventional ResNet-50 can capture CEM features related to HR status, performing well even under pronounced class imbalance. Small single-centre size, image-level splitting and the mixture of early and late CEM phases limit generalisability; larger multi-institution cohorts and patient-level experiments are required.

## Key points  

• ResNet-50 predicted HR positivity on CEM with 89.9 % accuracy and AUC-ROC 0.84 in an independent-test cohort.  

• Balanced accuracy (0.71) and MCC (0.52) demonstrate maintained discrimination despite an 85 % class imbalance.  

• Study limitations include small single-centre dataset, image-level rather than patient-level split, and mixing of early- and late-phase CEM images.

## Relevance statement  

Concept: This work explores whether routinely acquired CEM images contain enough information for deep-learning prediction of hormone-receptor status, aiming at a less invasive complement to biopsy.  

Implementation: A benchmark ResNet-50 was trained with minimal preprocessing; performance was quantified with imbalance-aware metrics to provide a realistic assessment in a highly skewed dataset, highlighting both the promise and current constraints of CEM-based molecular imaging.

## Introduction

Breast cancer is the most frequently diagnosed malignancy worldwide, with about **2.3 million new cases and 685 000 deaths in 2022** according to the latest WHO fact-sheet [1]. Because the disease is biologically heterogeneous, expression of the oestrogen (ER) and progesterone (PR) receptors—summarised as hormone-receptor (HR) status—remains a pivotal determinant of systemic-treatment strategy and prognosis [2]. HR status is routinely assessed on tissue obtained by core biopsy or surgery, but this invasive approach is time-consuming, prone to sampling error and limited in its ability to capture spatial heterogeneity [3, 4]. A rapid, image-based surrogate could therefore complement or, in selected scenarios, reduce the need for tissue sampling.

Contrast-enhanced mammography (CEM) merges dual-energy X-ray acquisition with an iodinated contrast bolus, highlighting lesion vascularity in a workflow that is faster and more widely available than MRI. Early clinical series reported higher sensitivity for lesion detection, particularly in dense breasts [5, 6], and recent reviews underline continued technical progress together with the emerging role of artificial-intelligence (AI) tools in CEM interpretation [7, 8].

Concurrently, deep learning—especially convolutional neural networks (CNNs)—has transformed medical-image analysis. Surveys document expert-level performance across radiology domains [9]; ResNet and related architectures are now de-facto baselines for image-classification tasks [10], and have already improved breast-cancer risk prediction on screening mammography [11]. Several groups have extended these methods to molecular profiling. Zeng et al. predicted ER, PR and HER2 expression from digital mammograms (AUC up to 0.80) [12]. Huang et al. and Ming et al. used dynamic-contrast MRI radiomics or multi-scale CNNs to distinguish luminal from non-luminal cancers or to infer PAM50 subtypes [13, 14]. Within the smaller CEM literature, Fanizzi et al. proposed a radiomics-based malignancy detector [15]; Marino et al. applied texture analysis for molecular-subtype differentiation [16]; and Dominique et al. trained a shallow CNN on full-field recombined CEM images to classify ER positivity and triple-negative phenotype [17].

Collectively, these investigations have moved the field forward but share several limitations: they are predominantly single-centre, rely on modest sample sizes, often exhibit pronounced class imbalance, rarely include an independent test cohort, and typically report only accuracy and AUC without imbalance-aware metrics such as balanced accuracy or the Matthews correlation coefficient. 

To provide a transparent benchmark that explicitly addresses class imbalance, we evaluate a standard ResNet-50 for HR prediction from CEM in a single-centre proof-of-concept study. Tumour regions are manually cropped to isolate lesion-specific signal while keeping preprocessing minimal. Model performance is reported with conventional metrics (accuracy, AUC-ROC) **and** imbalance-aware measures (balanced accuracy, MCC) to reflect the ≈ 85 % prevalence of HR-positive cases. Although preliminary and single-centre, the work establishes a reproducible reference point for subsequent multi-institution investigations.

## Materials and Methods  

### Study design and ethics  
This retrospective, single-centre feasibility study was approved by the institutional ethics committee. All 70 women analysed had previously provided written consent for anonymised research use of their imaging and pathology data.

### Patient cohort  
Women who underwent contrast-enhanced mammography (CEM) between October 2020 and May 2022 for pre-operative staging of biopsy-proven invasive breast cancer were screened. Departmental policy restricts CEM to patients aged ≥ 30 years, so younger women are absent. All tumours were clinical stage T1–T2 at presentation. Contra-indications to CEM (pregnancy, breast implants, impaired renal function, severe contrast reaction) preclude referral and are therefore not represented.  

Hormone-receptor (HR) status was copied verbatim from each pathology report. A case was labelled HR-positive when either ER or PR was marked “positive” by the institutional pathology service. Because the exact immunohistochemical threshold was not recorded consistently, a degree of uncertainty in the reference standard is acknowledged.

### CEM protocol  
Imaging was performed on a Hologic Selenia Dimension system. After intravenous iodinated contrast (Iomeron 350, 1.5 mL kg⁻¹; maximum 110 mL) at 2–3 mL s⁻¹, dual-energy two-dimensional craniocaudal (CC) and mediolateral-oblique (MLO) views of the affected breast were acquired 1 minute post-injection. This timing reflects vendor default at our centre; many units employ 2 minutes, so protocol heterogeneity is a potential source of variability.  

Because the dataset is small, **late low-dose images acquired 7 minutes after contrast injection were also retained and treated as independent inputs.** The impact of mixing early and late phases is unknown and is considered a limitation. Only recombined (subtracted) images of the tumour-bearing breast were analysed; low-energy images and all contralateral views were excluded.

![Overview Figure](overview-figure.png)  
**Figure 1.** Schematic overview of the study pipeline: CEM acquisition, manual cropping, data augmentation, ResNet-50 training and inference.

### Region-of-interest definition and preprocessing  
Enhancing lesions were localised on the recombined images and **cropped manually** with the workstation viewer (no external annotation software, no mirror-padding). Each rectangular crop covered the lesion and a small rim of surrounding tissue; multifocal tumours were cropped separately.

During training, crops were fed to a `RandomResizedCrop(224)` layer, which rescales them to 224 × 224 pixels while allowing modest aspect-ratio changes—potentially distorting tumour geometry, a further limitation. Validation and test images underwent centre cropping to 224 pixels. All images were normalised to the ImageNet mean and standard deviation.

![raw-cems](raw-cems.png)  
**Figure 2.** Representative early- and late-phase recombined CEM images (LMLO projection) illustrating the baseline appearance before cropping.

![cropped-cems](cropped-cems.png)  
**Figure 3.** Examples of manually cropped regions of interest (ROIs) that were used as model input.

### Dataset split  
From the 70 patients, 403 cropped images (each CC, MLO and late-phase projection counted separately) were derived and randomly assigned to a **training set** (n = 254), a **validation set** (n = 70) and an **independent-test set** (n = 79). Allocation was performed at **image level**; consequently, a single patient may contribute to multiple subsets, raising the possibility of information leakage. HR-positive cases comprised ~85 % of every subset.

### Convolutional-network architecture and training  
A ResNet-50 pretrained on ImageNet served as backbone; all convolutional layers were frozen. The original fully connected layer was replaced by a classifier that applies dropout (rate 0.5), a 2 048 → 512 linear layer, ReLU activation, another dropout layer (rate 0.3) and a final 512 → 2 linear layer to output class logits.  

Training used the Adam optimiser with an initial learning rate of 1 × 10⁻⁴ and weight-decay of 1 × 10⁻⁴. A ReduceLROnPlateau scheduler reduced the learning rate by a factor of 0.1 after five epochs without improvement in validation loss. The loss function was standard, unweighted cross-entropy; mini-batch size was 32; and training ran for up to 50 epochs. The network snapshot that achieved the highest **validation accuracy** was retained. Because accuracy may favour the majority class, imbalance-aware metrics are also reported.

### Performance metrics and statistical analysis  
Performance on the training, validation and independent-test sets was summarised with: accuracy, precision, recall, F1-score, balanced accuracy, Matthews correlation coefficient (MCC) and the area under the receiver-operating-characteristic curve (AUC-ROC). Ninety-five-percent confidence intervals (95 % CI) were calculated with 1 000-iteration non-parametric bootstrap. No hypothesis tests or p-values are reported, as only a single model was evaluated.

### Implementation and hardware  
The pipeline is implemented in Python with PyTorch; full requirements and source code are available on GitHub (link in the Data-availability statement). Experiments ran on an Apple M2 laptop with 8 GB unified memory via the Metal Performance Shaders backend; the script automatically falls back to CUDA GPU or CPU if available.

## Results  

### Training behaviour  
The cross-entropy loss declined from 0.46 at epoch 1 to 0.22 at epoch 50. Correspondingly, accuracy rose from 81 % to 92 % on the training set, while validation accuracy stabilised between 85 % and 88 % after epoch 35, suggesting limited over-fitting.

### Final model performance  
Discrimination results with 95 % bootstrap confidence intervals are listed in Table 2.  

* Training set: accuracy 92.1 % (CI 88.2–94.9 %) and AUC-ROC 0.9048 (0.8338–0.9604).  
* Validation set: accuracy 87.1 % (77.3–93.1 %), balanced accuracy 0.7583 (0.5769–0.9133) and MCC 0.4968 (0.1564–0.7517).  
* Independent-test set: accuracy 89.9 % (81.3–94.8 %), AUC-ROC 0.8382 (0.6971–0.9464), balanced accuracy 0.7126 (0.5628–0.8746) and MCC 0.5179 (0.1516–0.7996).  

The modest gap between overall accuracy and balanced accuracy reflects the pronounced class imbalance (~85 % HR-positive).

| Metric | Training set | Validation set | Independent-test set |
|---|---|---|---|
| Accuracy | 0.9213 (0.8815–0.9485) | 0.8714 (0.7734–0.9309) | 0.8987 (0.8127–0.9478) |
| Precision | 0.9170 (0.8794–0.9502) | 0.8770 (0.7973–0.9528) | 0.8885 (0.8072–0.9616) |
| Recall | 0.9213 (0.8898–0.9528) | 0.8714 (0.7857–0.9429) | 0.8987 (0.8351–0.9620) |
| F1-score | 0.9147 (0.8744–0.9506) | 0.8739 (0.7939–0.9429) | 0.8889 (0.8056–0.9594) |
| Balanced accuracy | 0.7746 (0.6904–0.8575) | 0.7583 (0.5769–0.9133) | 0.7126 (0.5628–0.8746) |
| MCC | 0.6503 (0.5155–0.7822) | 0.4968 (0.1564–0.7517) | 0.5179 (0.1516–0.7996) |
| AUC-ROC | 0.9048 (0.8338–0.9604) | 0.7583 (0.5182–0.9633) | 0.8382 (0.6971–0.9464) |

**Table 2.** Performance metrics for the final ResNet-50 model; values are point estimates followed by 95 % confidence intervals.

Figure 4 shows Grad-CAM visualisations for representative correctly and incorrectly classified cases, illustrating that network attention generally overlaps the enhancing tumour region.

![activated-cropped-cems](activated-cropped-cems.png)  
**Figure 4.** Grad-CAM heat-maps overlaid on cropped CEM images. Warm colours denote regions that contributed most to the HR-status prediction.

### Computational aspects  
Complete training (50 epochs) required approximately **30 minutes** on an Apple M2 laptop with 8 GB unified memory. Inference time per image was not formally measured.

## Discussion  

In this retrospective proof-of-concept we show that a standard ResNet-50, trained on manually cropped regions of contrast-enhanced mammograms, achieves 89.9 % accuracy and an AUC-ROC of 0.84 on an independent-test set when predicting hormone-receptor (HR) status. Reporting balanced accuracy (0.71) and Matthews correlation coefficient (0.52) alongside conventional metrics provides additional insight into model behaviour under the pronounced class imbalance (~85 % HR-positive). Few previous CEM studies have supplied such imbalance-aware figures, so our results extend the quantitative picture offered in earlier work.

When placed beside the literature, performance is broadly comparable. Dominique et al. obtained an AUC of 0.82 for ER prediction using full-field CEM images [17]; Marino et al. reported radiomics-based AUCs between 0.77 and 0.83 for molecular subtyping [16]; and Zeng et al. reached up to 0.80 for ER, PR and HER2 prediction on standard mammography [12]. Differences in cohort size, imaging protocol, input representation and HR definition prevent direct head-to-head comparison, yet the collective evidence suggests that CEM harbours imaging surrogates of tumour biology that CNNs can exploit.

Several considerations temper the interpretation of our findings. First, the study draws on a small single-centre cohort, so external validity remains uncertain. Second, images—not patients—were randomised across the training, validation and test subsets; the possibility that multiple projections from the same woman appear in different splits raises a risk of information leakage.  Third, to enlarge the dataset we retained late low-dose images acquired seven minutes after contrast injection in addition to standard one-minute images. The influence of mixing phases is unknown but may have introduced unwanted variation. Fourth, lesion patches were resized with `RandomResizedCrop`, which can subtly stretch or compress the tumour; the impact of this variable aspect ratio has not been quantified. Fifth, HR labels were copied verbatim from pathology reports without a uniform immunohistochemical threshold, introducing potential label noise. Sixth, although evaluation used balanced accuracy and MCC, the network was optimised with unweighted cross-entropy and the best epoch selected by validation accuracy—decisions that can favour the majority class. Seventh, our CEM protocol uses a one-minute post-contrast delay, whereas many sites image at two minutes; variation in timing, injection rate and detector technology could affect generalisability. Finally, no parallel analysis of MRI or ultrasound was performed, so the relative diagnostic contribution of CEM cannot be inferred from this work.

Despite these limitations, the study provides a transparent baseline for HR-status prediction from CEM and underscores the value of imbalance-aware reporting. Future research should incorporate patient-level splits, harmonised acquisition protocols, automatic lesion localisation, consensus pathology thresholds and direct modality comparisons to clarify the clinical role of CEM-based deep learning in molecular characterisation.

## Conclusion  

A convolutional neural network as conventional as ResNet-50 can extract informative features from contrast-enhanced mammography and, in this preliminary study, predicts hormone-receptor status with 89.9 % accuracy and an AUC-ROC of 0.84 on an independent-test set. The accompanying balanced accuracy of 0.71 and Matthews correlation coefficient of 0.52 indicate that the model retains discriminative value despite a markedly skewed class distribution. These results, achieved with minimal preprocessing and without architectural customisation, confirm the presence of biologically relevant signal in CEM images.

Interpretation, however, must remain cautious. The dataset is small, originates from a single institution, and was split at image rather than patient level; late-phase images were mixed with standard early images; variable aspect ratios were introduced during cropping; and pathology reports provided HR labels without a uniform immunohistochemical threshold. Each of these factors can influence performance estimates.

Even so, the study offers a transparent baseline and emphasises the importance of reporting imbalance-aware metrics in molecular-imaging AI. Extending this work to multi-centre cohorts, harmonised CEM protocols, patient-level splits and automatic lesion localisation will be essential next steps toward assessing whether CEM-based deep learning can contribute meaningfully to non-invasive tumour characterisation in clinical practice.

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