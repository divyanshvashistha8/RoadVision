# RoadVision — Road Damage Analysis Using Machine Learning and Computer Vision

An end-to-end Machine Learning and Computer Vision project for road damage classification, object detection, unsupervised learning, anomaly scoring, and visual interpretation using the RDD2022 India dataset.

## 1. Project Overview

RoadVision is an experimental road damage analysis system that combines classical machine learning and deep learning approaches to explore automated road damage identification.

The project implements multiple approaches, including:

- HOG feature extraction with SVM and Random Forest.
- Custom CNN-based image classification.
- ResNet18 transfer learning.
- Deep feature extraction and dimensionality reduction.
- K-Means and DBSCAN clustering.
- Convolutional autoencoder-based anomaly scoring.
- YOLO11n object detection.
- Grad-CAM model interpretability.
- Bounding-box-based damage extent analysis.
- Interactive Streamlit application.

## 2. Project Pipeline

```text
RDD2022 India Dataset
        |
        v
Dataset Preparation
        |
        +---- HOG + SVM
        |
        +---- Random Forest
        |
        +---- CNN Classification
        |
        +---- ResNet18 Transfer Learning
        |          |
        |          +---- Grad-CAM
        |
        +---- Deep Feature Extraction
        |          |
        |          v
        |         PCA
        |          |
        |          +---- K-Means
        |          +---- DBSCAN
        |
        +---- Convolutional Autoencoder
        |          |
        |          v
        |     Anomaly Scoring
        |
        +---- YOLO11n Object Detection
                   |
                   +---- Damage Extent Analysis

All components feed into:

        Streamlit Application
```

## 3. Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch, Torchvision |
| Computer Vision | OpenCV, Pillow |
| Classical ML | Scikit-learn |
| Object Detection | Ultralytics YOLO11n |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Application | Streamlit |
| Configuration | YAML |

## 4. Dataset

**Dataset:** Road Damage Detection 2022 (RDD2022) — India subset.

The project exclusively uses the India subset and does not combine data from other countries.

RDD2022 provides road images and Pascal VOC XML annotations.

### Damage Categories

| Original Label | Mapped Category |
|---|---|
| D00 | Longitudinal Crack |
| D10 | Transverse Crack |
| D20 | Alligator Crack |
| D40 | Pothole |

Other annotation categories are excluded from the configured four-class mapping.

### Processed Dataset Distribution

The corrected, deduplicated classification dataset contains:

| Dataset Split | Images |
|---|---:|
| Training | 2,257 |
| Validation | 483 |
| Testing | 483 |
| **Total** | **3,223** |

The classification labels are derived from bounding-box annotations. For images containing multiple damage categories, the category with the largest total annotated bounding-box area is selected as the dominant image-level label.

This provides an image-classification baseline from a dataset originally designed for object detection.

> **Dataset note:** The raw dataset is not included in this repository. Users must download RDD2022 separately and follow its applicable terms of use.

## 5. Installation and Environment Setup

Recommended Python version: 3.10 or 3.11.

Create a Conda environment:

```bash
conda create -n roadvision python=3.10 -y
conda activate roadvision
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU acceleration, install a compatible CUDA-enabled PyTorch and Torchvision build using the official PyTorch installation instructions.

The project was developed and tested using CPU-only PyTorch.

## 6. Project Configuration

Project parameters are maintained in:

```text
configs/config.yaml
```

Important configuration settings:

- Image size: 224
- Classification batch size: 32
- Number of data-loader workers: 0
- Validation split: 15%
- Test split: 15%
- Random seed: 42
- YOLO image size: 416
- YOLO batch size: 4

## 7. Dataset Preparation

Place the extracted India dataset inside:

```text
data/raw/RDD2022/India/
```

Run the preparation script:

```bash
python scripts/prepare_data.py
```

This generates the processed classification and YOLO datasets, along with the dataset manifest and class mapping.

Expected output:

```text
data/processed/
├── classification/
├── yolo/
├── manifest.csv
└── classes.json
```

## 8. Model Training

### 8.1 Classical Machine Learning

HOG features are extracted from road images and used with:

- Support Vector Machine (SVM)
- Random Forest

Run:

```bash
python scripts/train_classical.py
```

### 8.2 Custom CNN

Train the custom Convolutional Neural Network:

```bash
python scripts/train_cnn.py
```

### 8.3 ResNet18 Transfer Learning

Train the ResNet18-based transfer learning model:

```bash
python scripts/train_transfer.py
```

### 8.4 Deep Feature Extraction

Extract feature embeddings:

```bash
python scripts/extract_embeddings.py
```

### 8.5 Unsupervised Learning

Apply PCA, K-Means, and DBSCAN to the extracted embeddings:

```bash
python scripts/run_clustering.py
```

### 8.6 Autoencoder-Based Anomaly Detection

Train the convolutional autoencoder:

```bash
python scripts/train_anomaly.py
```

Evaluate reconstruction errors:

```bash
python scripts/evaluate_anomaly.py
```

Images with reconstruction errors exceeding the training-derived threshold are flagged for further inspection.

### 8.7 YOLO11n Object Detection

Train the YOLO11n object detector:

```bash
python scripts/train_detector.py
```

Training configuration used:

- Pretrained YOLO11n weights.
- 10 epochs.
- Image size: 416.
- Batch size: 4.
- Patience: 5.

The trained model is saved under:

```text
models/detection/roadvision_clean_split/weights/best.pt
```

## 9. Model Interpretability and Damage Analysis

### Grad-CAM

Generate Grad-CAM visualizations for ResNet18:

```bash
python scripts/gradcam.py
```

Grad-CAM highlights image regions contributing to the model's predicted class.

### Damage Extent Analysis

Run the bounding-box-based damage extent analysis:

```bash
python scripts/analyze_severity.py
```

The heuristic categorizes detections using bounding-box area relative to the complete image area:

| Category | Bounding-Box Area |
|---|---|
| Low | Less than 1% |
| Moderate | 1% to 5% |
| High | 5% or more |

These categories represent visual extent only. They are not engineering standards or validated structural severity assessments.

## 10. Model Evaluation

Display available classification evaluation results:

```bash
python scripts/evaluate_all.py
```

This script lists existing JSON evaluation files and displays available accuracy and macro F1 values.

It does not independently retrain or evaluate the models.

## 11. Single Image Prediction

Run inference on an individual road image:

```bash
python predict.py --image path/to/road.jpg
```

The inference pipeline combines classification, object detection, anomaly scoring, and damage extent analysis.

## 12. Streamlit Application

Launch the interactive application:

```bash
streamlit run app/streamlit_app.py
```

The application provides an interface for exploring road damage predictions and model outputs.

## 13. Experimental Results

The following classification results were obtained using the corrected test split containing 483 images.

### Classification Performance

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| HOG + SVM | 65.01% | 0.4247 |
| Random Forest | 54.66% | 0.2238 |
| Custom CNN | 64.39% | 0.3431 |
| ResNet18 | 60.46% | 0.3732 |

HOG + SVM achieved the highest overall accuracy and macro F1 among the evaluated classification models.

However, the dataset has substantial class imbalance. The test split contains only seven Transverse Crack images, which limits the reliability of minority-class performance estimates.

### YOLO11n Object Detection

| Metric | Result |
|---|---:|
| Precision | 0.5315 |
| Recall | 0.2789 |
| mAP@50 | 0.2474 |
| mAP@50–95 | 0.0993 |

The detector provides an experimental object-detection baseline. Its recall and mAP indicate that additional optimization is required before practical deployment.

### Clustering Results

| Metric | Result |
|---|---|
| Best K-Means clusters | 3 |
| K-Means silhouette score | 0.0680 |
| DBSCAN silhouette score | Not available |

The low silhouette score indicates weak cluster separation. Therefore, clustering is treated as exploratory unsupervised analysis.

### Anomaly Detection Results

Evaluation was performed on 483 validation images.

| Metric | Result |
|---|---:|
| Training-derived threshold | 0.034526 |
| Mean reconstruction error | 0.016946 |
| Median reconstruction error | 0.013739 |
| Flagged images | 33 |
| Flagged percentage | 6.83% |

The autoencoder uses reconstruction error to identify images with unusually high reconstruction loss relative to the training-derived threshold.

This is experimental anomaly scoring, not a validated damaged-versus-normal road classifier.

## 14. Project Structure

```text
RoadVision/
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── classical/
│   ├── clustering/
│   └── detection/
│
├── checkpoints/
│   ├── cnn/
│   ├── resnet18/
│   └── anomaly/
│
├── outputs/
│   ├── anomaly/
│   ├── clustering/
│   ├── embeddings/
│   ├── gradcam/
│   ├── metrics/
│   ├── predictions/
│   └── severity/
│
├── scripts/
│   ├── analyze_severity.py
│   ├── evaluate_all.py
│   ├── evaluate_anomaly.py
│   ├── evaluate_autoencoder_test.py
│   ├── extract_embeddings.py
│   ├── gradcam.py
│   ├── prepare_data.py
│   ├── rebuild_yolo.py
│   ├── run_clustering.py
│   ├── train_anomaly.py
│   ├── train_classical.py
│   ├── train_cnn.py
│   ├── train_detector.py
│   └── train_transfer.py
│
├── src/
│
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 15. Limitations and Future Improvements

### Current Limitations

- Significant class imbalance, especially for Transverse Crack.
- Moderate image-classification performance.
- Limited YOLO detection recall and localization performance.
- Weak separation in unsupervised clustering.
- Anomaly scoring requires further validation using appropriate normal-road and damaged-road data.
- Bounding-box-based extent categories do not establish actual structural road severity.

### Future Improvements

- Apply class balancing and more targeted data augmentation.
- Experiment with stronger pretrained vision architectures.
- Improve object detection through hyperparameter tuning and longer training.
- Explore more effective representation learning for clustering.
- Calibrate anomaly thresholds using suitable normal-road and damaged-road datasets.
- Validate damage severity against expert engineering assessments.

## 16. Disclaimer

RoadVision is an educational and experimental Machine Learning and Computer Vision project.

Its predictions, bounding boxes, anomaly scores, and visual-extent categories are not certified road-condition assessments. They should not be used as the sole basis for infrastructure maintenance or safety decisions.

---

**RoadVision — Exploring Road Damage Analysis Through Machine Learning and Computer Vision.**
