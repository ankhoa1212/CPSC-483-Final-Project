# CPSC-483-Final-Project
## Problem Statement
In general, machine learning can be applied in a wide variety of fields, especially when it comes to enhancing human health in environments such as clinical settings, medical research, healthcare, etc. In these areas, it is especially important to be able to analyze large amounts of data quickly and accurately. As such, training a machine learning model to quickly garner insights from medical images can help reduce human error, streamline the process of analysis, and improve patient outcomes.

## Table of Contents
1. [Dataset](#medmnist-dataset)
2. [Data Analysis](#data-analysis)
3. [Convolutional Neural Network](#convolutional-neural-network)
4. [Autokeras](#autokeras)
5. [Evaluation](#evaluation)
6. [Conclusion](#conclusion)
7. [Resources](#resources)

## MedMNIST Dataset
The [MedMNIST dataset](https://github.com/MedMNIST) was used because it is a standardized, diverse dataset that can be adapted for machine learning use cases. For this project, the BreastMNIST dataset was used due to it having the least amount of data points, which would make the training process a lot quicker.

## Data Analysis
To better understand the dataset, exploratory data analysis was conducted. Data was visualized and inspected to confirm data quality.

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ankhoa1212/CPSC-483-Final-Project/blob/main/data_analysis_and_modeling.ipynb)

## Convolutional Neural Network
For a proof-of-concept, a convolutional neural network was developed. This particular model was chosen since convolutional layers are typically well-suited for image classification tasks.

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ankhoa1212/CPSC-483-Final-Project/blob/main/cpsc483_final_project_cnn.ipynb)

## Autokeras
The CNN showed promising results, so an automated machine learning library was used to find an ideal architecture for that can perform well on the data.
Script for training and evaluating with Autokeras was adapted from [this code](https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/train_and_eval_autokeras.py).

1. Install Python 3.11 ([pyenv](https://github.com/pyenv/pyenv) can be installed and used for multiple Python versions)
2. Create a [Python Virtual Environment](https://docs.python.org/3/library/venv.html) (recommended)
3. Install requirements (can install [TensorFlow with GPU acceleration](https://www.tensorflow.org/install/pip))
```
pip install -r requirements.txt
```
4. Run the Autokeras script
```
python3 train_and_eval_autokeras.py --input_root medmnist
```

## Evaluation
Regarding metrics, both implementation using CNN and Aurokeras on the model were evaluated based on two criteria:
- Area under the curve (AUC): generally, the greater this value the better the model
- Accuracy: measures how often the model predicts the outcome

### CNN Results
Train: AUC: 0.850 Accuracy: 0.736

Test: AUC: 0.771 Accuracy: 0.731

### Autokeras Results
Train: AUC: 0.872 Accuracy: 0.844

Test: AUC: 0.965 Accuracy: 0.926

Validation: AUC: 0.953 Accuracy: 0.902

A total of 3 models were executed and averaged to make sure that the results were accurate.
The architecture of the best model is as follows:

![breastmnist_autokeras_model3_arch_visualization](https://github.com/user-attachments/assets/949f15f4-cf9e-465c-8c6d-ad0fb0e97101)

Model architectures were visualized using [this script](https://github.com/ankhoa1212/CPSC-483-Final-Project/blob/main/visualize_model.py).

## Conclusion
We developed a machine learning model with an AUC of 0.956 and an accuracy of 0.936, which exceeded the original BreastMNIST 2D [benchmark results](https://medmnist.com/) (AUC: 0.871, accuracy: 0.831).

### Takeaways
For image classification problems, having convolutional layers can be effective at extracting features from images. Managing multiple libraries and dependencies in Python can be complicated, so installing them in an isolated environment can help with avoiding conflicts between dependencies. As a result, Google Colab can be useful to quickly get started on a machine learning project, but has the downside of usage limitations, especially on problems that require a lot of computation. To set up and use certain code libraries, it is extremely useful to read existing documentation.

## Resources
- https://www.nature.com/articles/s41597-022-01721-8
- https://github.com/MedMNIST
- https://autokeras.com/
- https://www.tensorflow.org/install/pip
