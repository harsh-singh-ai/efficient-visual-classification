# Efficient Visual Classification for Resource-Constrained Systems

## Final Project Report

**Author:** Harsh Singh  
**Program:** Master of Computer Science  
**University:** University of Aizu  
**Project Type:** Machine Learning / Computer Vision

---

## Abstract

This project investigates the use of a lightweight deep learning model for image classification under resource constraints. A customized image dataset containing six everyday object categories—book, bottle, cup, keyboard, mouse, and phone—was collected and prepared for experimentation.

The project follows a complete machine learning workflow consisting of dataset collection, data validation, cleaning, leakage-free dataset splitting, model training, fine-tuning, held-out evaluation, and efficiency benchmarking.

MobileNetV3-Small was selected as the classification model because of its compact architecture and suitability for resource-constrained environments. A baseline model was first trained on the customized dataset, followed by partial fine-tuning to improve classification performance.

The baseline model achieved a test accuracy of 55.56%, while the fine-tuned model achieved 66.67%, representing an improvement of 11.11 percentage points. The final model contains approximately 1.524 million parameters and achieved an average CPU inference time of 8.94 ms per image in the local benchmark environment.

The results demonstrate that fine-tuning a lightweight convolutional neural network can improve classification performance while maintaining a relatively small computational footprint. However, the small dataset and limited test-set size remain important limitations of the current study.

---

## 1. Introduction

Image classification is a fundamental computer vision task in which a machine learning model assigns an input image to one of several predefined categories. Modern deep learning models can achieve strong classification performance, but many high-capacity architectures require substantial computational resources.

For applications running on resource-constrained systems, such as edge devices, embedded computers, or systems with limited computational capacity, lightweight neural network architectures are particularly useful. These models aim to provide a practical balance between classification accuracy, model size, computational requirements, and inference speed.

This project focuses on developing and evaluating a lightweight image classification system using MobileNetV3-Small. Instead of relying entirely on an existing benchmark dataset, a customized dataset was collected for six everyday object categories. The dataset was then systematically validated, cleaned, and divided into training, validation, and test subsets.

The project also compares a baseline training approach with partial fine-tuning and evaluates not only predictive performance but also model efficiency.

---

## 2. Project Objectives

The main objectives of this project are:

1. To collect and prepare a customized image classification dataset.
2. To define clear image classes and ensure consistent labeling.
3. To validate and clean the collected images.
4. To create training, validation, and test sets while preventing data leakage.
5. To train a lightweight MobileNetV3-Small baseline model.
6. To improve the model using partial fine-tuning.
7. To evaluate the final model using a held-out test set.
8. To analyze class-wise classification performance.
9. To measure the computational efficiency of the trained model.
10. To maintain a reproducible project structure containing code, results, and documentation.

## 3. Dataset Collection and Preparation

### 3.1 Dataset Definition

The task was defined as a multi-class image classification problem involving six categories of common everyday objects:

- Book
- Bottle
- Cup
- Keyboard
- Mouse
- Phone

The objective was to develop a compact classifier capable of distinguishing between these six object categories from RGB images.

### 3.2 Dataset Collection

The customized dataset was collected using images from Wikimedia Commons. Images were selected based on compatible open licenses, including Creative Commons and public-domain licenses.

The dataset collection process used automated retrieval through the Wikimedia Commons API. Downloaded images were converted to RGB JPEG format and constrained to a maximum thumbnail resolution of approximately 640 pixels while maintaining a minimum image size requirement.

For dataset provenance and reproducibility, metadata was recorded for each image. The metadata includes information such as:

- Image filename
- Class label
- Source
- Source title
- Source URL
- Author
- License
- Image width and height
- Search term used during collection

This information was stored in:

```text
data/image_metadata.csv


### 3.3 Initial Dataset

The initial collection contained 263 accepted images distributed across the six classes:

| Class | Initial Images |
|---|---:|
| Book | 47 |
| Bottle | 47 |
| Cup | 50 |
| Keyboard | 43 |
| Mouse | 38 |
| Phone | 38 |
| **Total** | **263** |

Before creating the final dataset, the collected images were validated and inspected for image integrity and duplication.

### 3.4 Data Cleaning

The collected images were processed to remove unsuitable samples and create a balanced dataset.

The validation process checked whether images could be successfully opened and processed. Exact duplicate checks were also performed to reduce the possibility of duplicated samples entering the dataset.

After cleaning and selection, the final dataset was balanced to contain exactly 20 images for each class.

The resulting dataset contained:

**120 images across 6 classes.**

The cleaning process and associated information were recorded in:

```text
data/cleaning_log.csv
```

### 3.5 Final Dataset Distribution

The final dataset contains:

| Class | Images |
|---|---:|
| Book | 20 |
| Bottle | 20 |
| Cup | 20 |
| Keyboard | 20 |
| Mouse | 20 |
| Phone | 20 |
| **Total** | **120** |

Balancing the dataset ensured that each class contributed the same number of samples to the subsequent experiments.

### 3.6 Dataset Splitting

The final dataset was divided into three subsets using a fixed random seed of 42:

| Dataset Split | Images | Percentage |
|---|---:|---:|
| Training | 84 | 70% |
| Validation | 18 | 15% |
| Test | 18 | 15% |
| **Total** | **120** | **100%** |

Each class contributes:

- 14 training images
- 3 validation images
- 3 test images

The split information was recorded in:

```text
data/split_metadata.csv
```

### 3.7 Data Leakage Prevention

Data leakage prevention was treated as an important part of the dataset preparation process.

The final dataset was checked for:

- Invalid images
- Duplicate samples
- Overlap between training, validation, and test sets
- Image integrity
- Consistent class distribution

The final verification reported:

- Invalid images: **0**
- Split overlap: **0**
- Image integrity: **PASS**
- Data leakage check: **PASS**

The verification results were saved in:

```text
results/metrics/dataset_verification.csv
```

A class-distribution visualization was also generated and saved as:

```text
results/figures/dataset_class_distribution.png
```

## 4. Methodology and Model Configuration

### 4.1 Overall Workflow

The project follows a complete machine learning workflow from dataset preparation to final model evaluation and efficiency analysis.

The main workflow consists of the following stages:

1. Customized dataset collection
2. Dataset validation and cleaning
3. Leakage-free dataset splitting
4. Baseline model training
5. Partial fine-tuning
6. Held-out test evaluation
7. Class-wise performance analysis
8. Efficiency benchmarking

The implementation is organized into separate Python scripts under the `src/` directory.

### 4.2 Model Selection

MobileNetV3-Small was selected as the primary classification model because it is designed as a lightweight convolutional neural network architecture suitable for applications where computational resources are limited.

The model provides a practical balance between classification capability and computational efficiency. This makes it appropriate for investigating lightweight visual classification rather than focusing only on maximum predictive accuracy.

### 4.3 Input and Classification Configuration

The model was configured for six-class image classification corresponding to the customized dataset:

```text
book
bottle
cup
keyboard
mouse
phone
```

Input images were processed using the image transformations required by the MobileNetV3-Small training pipeline.

The model's final classification layer was configured to produce six output classes.

### 4.4 Baseline Training

A baseline MobileNetV3-Small model was trained using the training subset of 84 images.

The validation subset containing 18 images was used to monitor model performance during training and select the best-performing checkpoint.

The baseline configuration was:

| Parameter | Configuration |
|---|---|
| Model | MobileNetV3-Small |
| Training images | 84 |
| Validation images | 18 |
| Test images | 18 |
| Number of classes | 6 |
| Epochs | 15 |
| Batch size | 16 |
| Learning rate | 0.0001 |
| Random seed | 42 |
| Device | CPU |

Training and validation loss and accuracy were recorded throughout the experiment.

The training results were saved in:

```text
results/metrics/training_metrics.csv
```

Training curves were saved as:

```text
results/figures/training_validation_loss.png
results/figures/training_validation_accuracy.png
```

The best baseline model checkpoint was saved as:

```text
results/models/mobilenet_v3_small_best.pth
```

### 4.5 Partial Fine-Tuning

After establishing the baseline, a second experiment was performed using partial fine-tuning of MobileNetV3-Small.

The purpose of fine-tuning was to allow the pretrained feature representation to adapt more effectively to the customized object dataset while maintaining the lightweight architecture.

Different learning rates were used for the feature extraction layers and the classifier:

| Parameter | Configuration |
|---|---|
| Model | MobileNetV3-Small |
| Feature learning rate | 0.00001 |
| Classifier learning rate | 0.0001 |
| Epochs | 15 |
| Batch size | 16 |
| Random seed | 42 |
| Device | CPU |
| Trainable parameters | 1,241,638 |
| Total parameters | 1,524,006 |

The best fine-tuned model checkpoint was saved as:

```text
results/models/mobilenet_v3_small_finetuned_best.pth
```

The corresponding training metrics were saved in:

```text
results/metrics/finetuned_training_metrics.csv
```

Training curves were saved as:

```text
results/figures/finetuned_training_validation_loss.png
results/figures/finetuned_training_validation_accuracy.png
```

### 4.6 Evaluation Strategy

The test dataset was kept separate from model training and validation.

The validation set was used during training to monitor performance and select the best model checkpoint, while the test set was reserved for the final evaluation.

This separation reduces the risk of reporting performance based on data that was directly used for model selection.

The final evaluation included:

- Test loss
- Test accuracy
- Class-wise precision
- Class-wise recall
- Class-wise F1-score
- Confusion matrix

The class-wise evaluation results were saved in:

```text
results/metrics/finetuned_classification_report

## 5. Experimental Results and Performance Analysis

### 5.1 Baseline Model Results

The baseline MobileNetV3-Small model was trained for 15 epochs using the customized training and validation datasets.

The best validation accuracy achieved by the baseline model was **44.44% at epoch 15**.

After training, the model was evaluated on the held-out test dataset. The baseline model achieved a test loss of **1.4181** and a test accuracy of **55.56%**.

The baseline results are summarized below:

| Metric | Baseline |
|---|---:|
| Best validation accuracy | 44.44% |
| Best epoch | 15 |
| Test loss | 1.4181 |
| Test accuracy | 55.56% |

The relatively modest baseline performance indicates that the small customized dataset presented a challenging classification problem for the initial model configuration.

### 5.2 Fine-Tuned Model Results

A partial fine-tuning experiment was subsequently performed to improve the adaptation of the pretrained MobileNetV3-Small representation to the customized dataset.

The fine-tuned model achieved its best validation accuracy of **50.00% at epoch 8**.

When evaluated on the held-out test set, the fine-tuned model achieved a test loss of **1.2366** and a test accuracy of **66.67%**.

The fine-tuned results are summarized below:

| Metric | Fine-Tuned Model |
|---|---:|
| Best validation accuracy | 50.00% |
| Best epoch | 8 |
| Test loss | 1.2366 |
| Test accuracy | 66.67% |

Compared with the baseline model, the fine-tuned model achieved a test accuracy improvement of **11.11 percentage points**.

### 5.3 Baseline vs Fine-Tuned Comparison

The main experimental results are summarized below:

| Metric | Baseline | Fine-Tuned |
|---|---:|---:|
| Best validation accuracy | 44.44% | 50.00% |
| Test accuracy | 55.56% | 66.67% |
| Test improvement | — | +11.11 pp |
| Total parameters | 1,524,006 | 1,524,006 |

The results show that fine-tuning improved both validation and held-out test performance while retaining the same underlying MobileNetV3-Small architecture and parameter count.

The validation accuracy increased by **5.56 percentage points**, while the test accuracy increased by **11.11 percentage points**.

This suggests that adapting the pretrained representation to the customized dataset was beneficial for the classification task.

### 5.4 Class-Wise Performance

The fine-tuned model was further evaluated using class-wise precision, recall, and F1-score.

The class-wise F1-scores were:

| Class | F1-Score |
|---|---:|
| Book | 0.67 |
| Bottle | 0.67 |
| Cup | 0.40 |
| Keyboard | 0.67 |
| Mouse | 0.75 |
| Phone | 0.80 |

The **phone** class achieved the highest F1-score at **0.80**, followed by **mouse** at **0.75**.

The **cup** class had the lowest F1-score at **0.40**, indicating that this category was more difficult for the model to distinguish from the other object categories.

The complete classification report is available at:

```text
results/metrics/finetuned_classification_report.csv
```

### 5.5 Confusion Matrix Analysis

A confusion matrix was generated to examine the prediction behavior of the fine-tuned model across the six classes.

The confusion matrix provides a more detailed view of which object categories were correctly classified and which categories were confused with one another.

The visualization is available at:

```text
results/figures/finetuned_confusion_matrix.png
```

Because the test set contains only three images per class, individual class-level results should be interpreted cautiously. A single incorrect prediction can substantially change the precision, recall, or F1-score for a class.

### 5.6 Interpretation of Results

The experimental results demonstrate that the fine-tuned model performed better than the baseline model on the held-out test set.

The improvement from **55.56% to 66.67% test accuracy** indicates that fine-tuning helped the pretrained model adapt to the customized dataset.

However, the results should not be interpreted as a definitive estimate of real-world classification performance. The dataset contains only 120 images, and the test set contains only 18 images in total. Therefore, the measured accuracy is sensitive to the specific images selected for the test set.

The main conclusion from this experiment is that lightweight transfer learning and fine-tuning can improve performance on a small customized dataset without increasing the model's total parameter count.

## 6. Efficiency Benchmark

### 6.1 Purpose of the Benchmark

In addition to classification accuracy, the computational efficiency of the final model was evaluated.

For resource-constrained applications, model size, parameter count, and inference speed are important considerations. Therefore, the fine-tuned MobileNetV3-Small model was benchmarked in the local CPU environment used for this project.

The benchmark was performed using a batch size of 1 to measure single-image inference performance.

### 6.2 Benchmark Configuration

The efficiency benchmark used the following configuration:

| Parameter | Configuration |
|---|---|
| Model | MobileNetV3-Small |
| Input resolution | 224 × 224 |
| Device | CPU |
| Batch size | 1 |
| Warm-up runs | 10 |
| Timed runs | 50 |

Warm-up runs were performed before timing to reduce the effect of initial execution overhead.

The benchmark implementation is available in:

```text id="7e5v9j"
src/benchmark_model.py
```

### 6.3 Efficiency Results

The benchmark produced the following results:

| Efficiency Metric | Result |
|---|---:|
| Total parameters | 1,524,006 |
| Trainable parameters | 1,524,006 |
| Checkpoint size | 5.94 MB |
| Average inference time | 8.94 ms/image |
| Throughput | 111.88 images/sec |

The model contains approximately **1.524 million parameters** and the saved checkpoint occupies approximately **5.94 MB**.

In the local CPU benchmark environment, the average inference time was **8.94 milliseconds per image**, corresponding to a measured throughput of approximately **111.88 images per second**.

### 6.4 Interpretation

The benchmark indicates that the selected MobileNetV3-Small architecture provides a relatively compact model suitable for investigating resource-constrained image classification.

The combination of a small parameter count, approximately 5.94 MB checkpoint size, and low measured CPU inference latency supports the motivation for selecting a lightweight architecture.

However, the reported inference time and throughput are specific to the local CPU environment used during the experiment. They should not be interpreted as universal performance values for all computers or edge devices.

For deployment on a specific target device, additional benchmarking would be required using the actual hardware and deployment framework.

## 7. Limitations and Future Improvements

### 7.1 Dataset Size

The main limitation of this project is the relatively small dataset size.

The final dataset contains only **120 images**, with 20 images per class. Although the dataset was carefully cleaned and balanced, this number of samples is limited compared with typical computer vision datasets.

A larger dataset would provide more visual diversity and could help the model learn more robust representations of each object category.

### 7.2 Limited Test Set

The test set contains only **18 images**, with three images per class.

Because of the small test-set size, the measured test accuracy and class-wise metrics can change substantially based on individual predictions. Therefore, the reported **66.67% test accuracy** should be interpreted as an experimental result for this dataset rather than a definitive estimate of generalization performance.

Future experiments should use a larger test set to obtain more stable and statistically meaningful performance measurements.

### 7.3 Dataset Diversity

The images were collected from Wikimedia Commons and may not fully represent the range of conditions encountered in real-world applications.

Differences in:

- Object appearance
- Backgrounds
- Lighting
- Camera viewpoint
- Object orientation
- Image quality
- Scale

may affect classification performance.

Increasing the diversity of images within each class would improve the robustness of the classifier.

### 7.4 Model Performance

Although fine-tuning improved test accuracy from **55.56% to 66.67%**, the final accuracy remains moderate.

The relatively low validation accuracy also suggests that the model has difficulty generalizing from the small customized dataset.

Additional experiments with stronger data augmentation, hyperparameter tuning, and alternative lightweight architectures could potentially improve performance.

### 7.5 Hardware Benchmark Limitation

The reported inference time of **8.94 ms/image** and throughput of **111.88 images/sec** were measured on the local CPU environment used for this project.

These measurements are hardware-dependent and therefore cannot be directly generalized to other computers or embedded devices.

Future work should benchmark the model on representative edge or embedded hardware if deployment on resource-constrained systems is intended.

### 7.6 Future Improvements

Several improvements can be explored in future work:

1. **Increase dataset size**  
   Collect more images for every class to improve generalization.

2. **Increase dataset diversity**  
   Include more variations in lighting, viewpoint, background, object orientation, and scale.

3. **Apply stronger data augmentation**  
   Use techniques such as random cropping, horizontal flipping, rotation, and color transformations where appropriate.

4. **Perform systematic hyperparameter tuning**  
   Experiment with learning rates, batch sizes, optimizers, and training schedules.

5. **Evaluate additional lightweight architectures**  
   Compare MobileNetV3-Small with other efficient models to determine the best accuracy-efficiency trade-off.

6. **Use cross-validation or repeated experiments**  
   Repeated evaluation could provide more reliable estimates when the available dataset is small.

7. **Benchmark on target hardware**  
   Measure latency, throughput, memory usage, and power consumption on an actual resource-constrained or edge device.

8. **Explore model optimization**  
   Techniques such as quantization, pruning, and knowledge distillation could be investigated to further reduce computational requirements.

9. **Expand the classification task**  
   Additional object categories and more challenging visual conditions could be introduced to evaluate the scalability of the approach.

   ## 8. Reproducibility and Project Structure

### 8.1 Reproducibility

Reproducibility was considered throughout the project so that the dataset preparation, model training, evaluation, and benchmarking procedures can be repeated using the provided source code.

A fixed random seed of **42** was used for the dataset splitting and model experiments.

The project also stores important intermediate information and experimental outputs, including:

- Dataset metadata
- Cleaning information
- Dataset split information
- Dataset verification results
- Training metrics
- Classification metrics
- Confusion matrix results
- Training and validation curves
- Model benchmarking results

The source code required to reproduce the main stages of the project is available in the `src/` directory.

### 8.2 Software Environment

The project was implemented using Python and the following main libraries:

- PyTorch
- Torchvision
- NumPy
- Pandas
- Pillow
- Matplotlib
- Scikit-learn

The required Python packages are listed in:

```text
requirements.txt
```

The experiments were performed using a CPU-based environment.

### 8.3 Reproduction Workflow

The general workflow for reproducing the project is:

#### Step 1 — Clone the repository

```bash
git clone https://github.com/harsh-singh-ai/efficient-visual-classification.git
cd efficient-visual-classification
```

#### Step 2 — Create a Python virtual environment

```bash
python -m venv .venv
```

#### Step 3 — Install the required packages

```bash
python -m pip install -r requirements.txt
```

#### Step 4 — Validate the dataset

```bash
python src/validate_dataset.py
```

#### Step 5 — Create the dataset splits

```bash
python src/create_splits.py
```

#### Step 6 — Verify the final dataset

```bash
python src/verify_final_dataset.py
```

#### Step 7 — Train the baseline model

```bash
python src/train_model.py
```

#### Step 8 — Train the fine-tuned model

```bash
python src/train_finetuned.py
```

#### Step 9 — Evaluate the model

```bash
python src/evaluate_model.py
```

#### Step 10 — Run the efficiency benchmark

```bash
python src/benchmark_model.py
```

The exact outputs depend on the available dataset, software environment, and hardware. The reported benchmark values in this report correspond to the experimental environment used during this project.

### 8.4 Project Structure

The main project structure is:

```text
efficient-visual-classification/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── cleaning_log.csv
│   ├── image_metadata.csv
│   └── split_metadata.csv
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
├── src/
│   ├── benchmark_model.py
│   ├── clean_and_split.py
│   ├── create_splits.py
│   ├── download_dataset.py
│   ├── evaluate_model.py
│   ├── inspect_dataset.py
│   ├── train_finetuned.py
│   ├── train_model.py
│   ├── validate_dataset.py
│   └── verify_final_dataset.py
│
├── reports/
│   └── final_report.md
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

The repository separates data, source code, experimental results, and documentation to make the project easier to understand and maintain.

## 9. Dataset Provenance and Academic Integrity

### 9.1 Dataset Provenance

The images used in this project were collected from Wikimedia Commons using publicly available images with compatible licenses.

Source and licensing information was recorded during dataset collection and stored in:

```text
data/image_metadata.csv
```

The metadata provides information about the source, image title, author where available, license, source URL, and other image attributes.

This provenance information was retained to improve transparency and reproducibility of the dataset preparation process.

### 9.2 Responsible Dataset Usage

The project does not redistribute the original collected image files through the public GitHub repository. Instead, the repository contains the dataset preparation code, metadata, processing information, and experimental results required to document the workflow.

The project therefore separates the reproducible methodology and provenance information from the original external image files.

### 9.3 Academic Integrity

The implementation, experiments, evaluation, and analysis presented in this report correspond to the work performed for this project.

External resources and software libraries were used where appropriate, and dataset provenance information was recorded to maintain transparency regarding the source of the images.

The project repository provides the source code and documentation required to understand the experimental methodology and reproduce the main workflow.

## 10. Conclusion

This project developed and evaluated a lightweight image classification system using a customized dataset of six everyday object categories: book, bottle, cup, keyboard, mouse, and phone.

The project followed a complete machine learning workflow, including dataset collection, validation, cleaning, leakage-free splitting, model training, fine-tuning, held-out evaluation, and efficiency benchmarking.

The initial MobileNetV3-Small baseline achieved a test accuracy of **55.56%**. After partial fine-tuning, the test accuracy increased to **66.67%**, representing an improvement of **11.11 percentage points**.

The final model contains **1,524,006 parameters** and has a checkpoint size of approximately **5.94 MB**. In the local CPU benchmark environment, it achieved an average inference time of **8.94 ms per image** and a measured throughput of **111.88 images per second**.

These results demonstrate that fine-tuning a lightweight pretrained model can improve classification performance on a small customized dataset while retaining a relatively compact model architecture.

However, the small dataset and test set limit the generalizability of the reported results. Future work should investigate larger and more diverse datasets, stronger data augmentation, systematic hyperparameter tuning, alternative lightweight architectures, and deployment on actual resource-constrained hardware.

Overall, the project provides a complete and reproducible example of an efficient visual classification pipeline, from customized dataset preparation through model evaluation and computational benchmarking.

---

## 11. References

### Dataset Source

Wikimedia Commons. Publicly available image collection used for the customized dataset. Source and licensing information for individual images is recorded in `data/image_metadata.csv`.

### Software and Libraries

- PyTorch — Deep learning framework used for model training and evaluation.
- Torchvision — Computer vision models and image processing utilities.
- NumPy — Numerical computing.
- Pandas — Data processing and experiment tracking.
- Pillow — Image loading and processing.
- Matplotlib — Visualization and training curves.
- Scikit-learn — Evaluation metrics and classification analysis.

### Project Repository

The complete source code, documentation, experimental results, and reproducibility information are available in the project repository:

```text
https://github.com/harsh-singh-ai/efficient-visual-classification
```

---

## 12. Project Completion Summary