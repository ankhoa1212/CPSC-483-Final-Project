# CPSC-483-Final-Project
This the final project for CPSC 483 (Introduction to Machine Learning).

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ankhoa1212/CPSC-483-Final-Project/blob/main/data_analysis_and_modeling.ipynb)

# Running Autokeras script

Script for training and evaluating with Autokeras was adapted from code used in [MedMNIST](https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/train_and_eval_autokeras.py)

1. Install Python 3.11 ([pyenv](https://github.com/pyenv/pyenv) can be installed and used for multiple Python versions)

2. Install requirements (can also install [TensorFlow with GPU acceleration](https://www.tensorflow.org/install/pip))

    ```pip install -r requirements.txt```

3. Run the Autokeras script

    ```python3 train_and_eval_autokeras.py --input_root medmnist```

# Resources
- https://www.nature.com/articles/s41597-022-01721-8
- https://github.com/MedMNIST
- https://autokeras.com/
- https://www.tensorflow.org/install/pip
