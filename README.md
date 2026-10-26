# Efficient Visual Classification for Resource-Constrained Systems

A reproducible computer vision project for classifying everyday objects using a lightweight deep learning model, with emphasis on **dataset quality, validation, model improvement, and inference efficiency**.

The project follows a complete machine learning workflow: customized dataset collection, data cleaning, leakage-free train/validation/test splitting, model training, fine-tuning, evaluation, and efficiency benchmarking.

---

## Project Overview

This project investigates whether a compact image classification model can provide reliable performance while maintaining low computational and storage requirements.

The classifier recognizes six common object categories:

- Book
- Bottle
- Cup
- Keyboard
- Mouse
- Phone

The project was designed around practical machine learning engineering principles rather than simply training a model. Particular attention was given to:

- Customized dataset preparation
- Data quality inspection
- Prevention of data leakage
- Reproducible train/validation/test splitting
- Baseline model development
- Partial fine-tuning
- Held-out test evaluation
- Model efficiency benchmarking
- Dataset provenance and licensing

---

## Key Results

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Best validation accuracy | 44.44% | **50.00%** |
| Test accuracy | 55.56% | **66.67%** |
| Test improvement | — | **+11.11 percentage points** |
| Total parameters | 1.524M | 1.524M |
| Trainable parameters | 1.524M | 1.242M |
| Checkpoint size | — | **5.94 MB** |
| CPU inference time | — | **8.94 ms/image** |
| CPU throughput | — | **111.88 images/sec** |

The fine-tuned model improved held-out test accuracy from **55.56% to 66.67%**, an absolute improvement of **11.11 percentage points**.

The final model contains approximately **1.52 million parameters** and produced a **5.94 MB checkpoint**, demonstrating a relatively compact model suitable for investigating resource-constrained inference.

> Inference performance was measured on the local CPU environment using batch size 1 and should not be interpreted as device-independent performance.

---

## 1. Project Motivation

Many computer vision applications need to balance prediction accuracy with computational cost.

Large deep learning models can provide strong performance but may require significant memory, storage, and computational resources. This project therefore explores a lightweight image classification approach using **MobileNetV3-Small**.

The broader goal is to understand the complete workflow required to build an efficient computer vision system:

**Dataset → Cleaning → Splitting → Training → Fine-tuning → Evaluation → Efficiency Analysis**

This workflow is also relevant to applications involving **edge AI, embedded systems, mobile inference, and resource-constrained machine learning**.

---

## 2. Dataset

### Classes

The customized dataset contains six object categories:

```text
book
bottle
cup
keyboard
mouse
phone
```

### Dataset collection

Images were collected from **Wikimedia Commons** using publicly available images with compatible open licenses.

The project stores provenance information for the collected images, including:

- Image filename
- Class
- Source
- Source title
- Source URL
- Author
- License
- Image dimensions
- Search term

This information is stored in:

```text
data/image_metadata.csv
```

The original collection contained **263 valid images**.

After visual quality inspection and class balancing, a final dataset of **120 images** was created.

### Final dataset

Each class contains 20 images:

| Class | Images |
|---|---:|
| Book | 20 |
| Bottle | 20 |
| Cup | 20 |
| Keyboard | 20 |
| Mouse | 20 |
| Phone | 20 |
| **Total** | **120** |

---

## 3. Data Cleaning and Quality Control

The collected images were inspected before model training.

Images with unsuitable visual content, ambiguous object representation, or other quality problems were excluded from the final dataset.

The cleaning process is documented in:

```text
data/cleaning_log.csv
```

The final dataset was then divided using a fixed random seed:

```text
Random seed: 42
```

### Dataset split

| Split | Images | Percentage |
|---|---:|---:|
| Training | 84 | 70% |
| Validation | 18 | 15% |
| Test | 18 | 15% |
| **Total** | **120** | **100%** |

Each class contains:

```text
Training:   14 images
Validation:  3 images
Test:        3 images
```

The split information is recorded in:

```text
data/split_metadata.csv
```

### Data leakage check

Before training, the final dataset was automatically verified for:

- Invalid images
- Class imbalance
- Duplicate filenames across splits
- Split overlap
- Image integrity
- Data leakage

Verification result:

```text
Total images checked: 120
Invalid images: 0
Split overlap files: 0
Image integrity: PASS
Data leakage check: PASS
```

---

## 4. Methodology

The project follows a structured machine learning pipeline.

```text
Wikimedia Commons
       │
       ▼
Image Collection
       │
       ▼
Metadata + Provenance
       │
       ▼
Visual Quality Inspection
       │
       ▼
Dataset Cleaning
       │
       ▼
Leakage-Free Dataset Split
       │
       ├── Training Set
       ├── Validation Set
       └── Test Set
              │
              ▼
      MobileNetV3-Small
              │
       ┌──────┴──────┐
       ▼             ▼
   Baseline      Fine-tuning
       │             │
       └──────┬──────┘
              ▼
       Held-out Test
          Evaluation
              │
              ▼
     Efficiency Benchmark
```

---

## 5. Model

### MobileNetV3-Small

The project uses **MobileNetV3-Small** as the main architecture.

MobileNetV3-Small was selected because it provides a practical lightweight architecture for image classification and is appropriate for investigating resource-constrained inference.

The final classification layer was adapted for the six target classes.

### Baseline

The baseline experiment trained the model using the customized dataset with:

```text
Model:          MobileNetV3-Small
Epochs:         15
Batch size:     16
Learning rate:  0.0001
Device:         CPU
Random seed:    42
```

Results:

```text
Best validation accuracy: 44.44%
Test accuracy:            55.56%
```

---

## 6. Fine-Tuning

A second experiment used partial fine-tuning.

The model's feature extractor was partially fine-tuned while using a higher learning rate for the classification layer.

Configuration:

```text
Model:                    MobileNetV3-Small
Epochs:                   15
Batch size:               16
Feature learning rate:    0.00001
Classifier learning rate: 0.00010
Device:                   CPU
Random seed:              42
```

The best checkpoint was selected according to validation accuracy rather than the final training epoch.

```text
Best epoch:               8
Best validation accuracy: 50.00%
Test accuracy:            66.67%
```

This produced an improvement of:

```text
+11.11 percentage points
```

on the held-out test set compared with the baseline.

---

## 7. Test Evaluation

The fine-tuned model was evaluated on the held-out test set.

### Classification performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Book | 0.67 | 0.67 | 0.67 |
| Bottle | 0.67 | 0.67 | 0.67 |
| Cup | 0.50 | 0.33 | 0.40 |
| Keyboard | 0.67 | 0.67 | 0.67 |
| Mouse | 0.60 | 1.00 | 0.75 |
| Phone | 1.00 | 0.67 | 0.80 |

Overall:

```text
Test accuracy: 66.67%
Macro F1:      0.66
```

The strongest F1-score was obtained for **phone (0.80)**, followed by **mouse (0.75)**.

The weakest class was **cup (0.40 F1-score)**, indicating that this class requires additional data or improved visual representation.

> The test set contains only 3 images per class. Therefore, these class-level metrics are useful for evaluating this experiment but should not be interpreted as statistically stable estimates of real-world generalization.

---

## 8. Efficiency Benchmark

The final model was also evaluated from an inference-efficiency perspective.

### Model size

```text
Total parameters:       1,524,006
Trainable parameters:   1,241,638
Checkpoint size:        5.94 MB
```

### CPU inference

Using batch size 1:

```text
Average inference time: 8.94 ms/image
Throughput:             111.88 images/second
```

These measurements were obtained in the local CPU environment using the project's benchmarking script.

The efficiency results provide an additional perspective beyond classification accuracy and help demonstrate the trade-off between model performance and computational requirements.

---

## 9. Project Structure

```text
efficient-visual-classification/
│
├── data/
│   ├── raw/
│   │   ├── book/
│   │   ├── bottle/
│   │   ├── cup/
│   │   ├── keyboard/
│   │   ├── mouse/
│   │   └── phone/
│   │
│   ├── processed/
│   │
│   ├── cleaning_log.csv
│   ├── image_metadata.csv
│   └── split_metadata.csv
│
├── notebooks/
│
├── reports/
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
├── src/
│   ├── benchmark_model.py
│   ├── clean_and_split.py
│   ├── compare_models.py
│   ├── create_splits.py
│   ├── download_dataset.py
│   ├── evaluate_model.py
│   ├── inspect_dataset.py
│   ├── train_finetuned.py
│   ├── train_model.py
│   ├── validate_dataset.py
│   └── verify_final_dataset.py
│
├── tests/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 10. Reproducibility

### Requirements

The project was developed and tested with Python and PyTorch.

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

### Verify the dataset

Run:

```bash
python src/verify_final_dataset.py
```

### Train the baseline model

```bash
python src/train_model.py
```

### Train the fine-tuned model

```bash
python src/train_finetuned.py
```

### Evaluate the fine-tuned model

```bash
python src/evaluate_model.py
```

### Compare the experiments

```bash
python src/compare_models.py
```

### Run the efficiency benchmark

```bash
python src/benchmark_model.py
```

The scripts generate metrics, model checkpoints, and figures under:

```text
results/
```

---

## 11. Results and Artifacts

Important generated artifacts include:

### Training metrics

```text
results/metrics/training_metrics.csv
results/metrics/finetuned_training_metrics.csv
```

### Classification results

```text
results/metrics/finetuned_classification_report.csv
results/metrics/finetuned_confusion_matrix.csv
```

### Dataset verification

```text
results/metrics/dataset_verification.csv
```

### Training curves

```text
results/figures/training_validation_loss.png
results/figures/training_validation_accuracy.png
results/figures/finetuned_training_validation_loss.png
results/figures/finetuned_training_validation_accuracy.png
```

### Evaluation

```text
results/figures/finetuned_confusion_matrix.png
```

### Model checkpoints

```text
results/models/mobilenet_v3_small_best.pth
results/models/mobilenet_v3_small_finetuned_best.pth
```

---

## 12. Dataset Provenance and Licensing

The image collection uses publicly available Wikimedia Commons content with compatible open licenses.

For reproducibility and attribution, the project preserves source information in:

```text
data/image_metadata.csv
```

The metadata records the source title, URL, author, license, and other relevant information for the collected images.

The raw dataset is excluded from version control through `.gitignore`. This keeps the Git repository lightweight while preserving the information necessary to identify the original sources.

Users reproducing the project should verify the applicable license and attribution requirements for individual images before redistribution.

---

## 13. Limitations

This project has several important limitations.

### Small dataset

The final dataset contains only 120 images. This is sufficient for demonstrating the machine learning workflow but is not large enough to establish strong real-world generalization.

### Small test set

Only 18 images are used for final testing, with 3 images per class. Therefore, a single misclassification can noticeably change class-level metrics.

### Limited visual diversity

The selected images do not represent the full variation encountered in real-world environments, such as:

- Different lighting conditions
- Object rotations
- Occlusion
- Background clutter
- Different camera qualities
- Different object shapes and designs

### CPU-only experiment

The experiments were performed using a CPU environment. GPU-based training and inference were not required for this project.

---

## 14. Future Improvements

Several improvements could strengthen the system:

1. Expand the dataset with more images per class.
2. Increase variation in lighting, orientation, scale, and background.
3. Perform stronger data augmentation.
4. Compare additional lightweight architectures.
5. Evaluate model quantization.
6. Investigate pruning and knowledge distillation.
7. Test inference on an actual edge or embedded device.
8. Perform cross-validation with a larger dataset.
9. Add confidence calibration and uncertainty analysis.
10. Explore efficient AI approaches relevant to neuromorphic and resource-constrained computing.

These directions could extend the project toward **edge AI, efficient computer vision, embedded inference, and hardware-aware machine learning**.

---

## 15. Technologies

### Programming

- Python

### Machine Learning

- PyTorch
- Torchvision
- Scikit-learn
- NumPy
- Pandas

### Computer Vision

- Pillow
- Image preprocessing
- Image classification
- Confusion matrix analysis

### Experimentation

- Matplotlib
- Seaborn
- CSV-based experiment tracking

### Engineering

- Git
- GitHub
- Reproducible training scripts
- Dataset provenance tracking
- Automated dataset verification

---

## 16. Skills Demonstrated

This project demonstrates practical experience in:

- Machine learning pipeline development
- Computer vision
- Dataset collection and preparation
- Data cleaning
- Dataset quality validation
- Data leakage prevention
- Train/validation/test methodology
- Transfer learning and fine-tuning
- Model evaluation
- Experiment comparison
- Performance benchmarking
- Reproducible ML workflows
- Git/GitHub project organization
- Technical documentation

---

## 17. Conclusion

This project demonstrates an end-to-end approach to building a lightweight image classification system.

Starting from a customized dataset collected from open sources, the project applies systematic data cleaning, leakage-free splitting, model training, fine-tuning, held-out evaluation, and efficiency benchmarking.

The fine-tuned MobileNetV3-Small model achieved:

```text
66.67% test accuracy
5.94 MB checkpoint size
1.524M total parameters
8.94 ms/image CPU inference
```

Compared with the baseline, fine-tuning improved test accuracy by **11.11 percentage points**.

The project provides a foundation for further work in **efficient computer vision, edge AI, resource-constrained inference, and hardware-aware machine learning**.

---

## Author

**Harsh Singh**

Master's Student in Computer Science  
University of Aizu, Japan

---

## License

See the `LICENSE` file for the license applicable to the source code.

Dataset images remain subject to their respective original licenses. See `data/image_metadata.csv` for image-level provenance and licensing information.