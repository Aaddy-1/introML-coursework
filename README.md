# Decision Tree Coursework (COMP70050)

This repository contains the implementation of a **Decision Tree learning algorithm** from scratch, designed to classify indoor locations based on WIFI signal strengths. The implementation covers tree creation, evaluation with 10-fold cross-validation, and post-pruning based on validation error

## Setup and Prerequisites

This project is built using standard Python libraries, with a strict limitation to **NumPy** and **Matplotlib** for external dependencies.

You can use either the script (`main.py`) or the notebook (`main.ipynb`) to evaluate the project

### **1. Setup the Data**
1) The code is assuming that the data is being stored in a folder called 'Dataset', with filenames 'clean_dataset.txt', 'noisy_dataset.txt'.
2) In order to test the code using your own data, please change the lines 
   ```python
   clean_dataset = np.loadtxt('Dataset/clean_dataset.txt')
   noisy_dataset = np.loadtxt('Dataset/noisy_dataset.txt')
   ```
   To: 
   ```python
   your_data = np.loadtxt('path-to-your-data')
   ```
3) In order to evaluate your data, just replace the lines:
   ```python
   datasets = kfold_datasets_generator(clean_dataset)
   kfold_evaluator(datasets, prune=True)
   ```
   To:
   ```python
   datasets = kfold_datasets_generator(your_data)
   kfold_evaluator(datasets, prune=True)
   ```
   If you want to evaluate with pruned decision trees, set `prune = True` in the call to `kfold_evaluator`.
   prune remains `False` by default in the function.
   

The code is confirmed to run on the DoC lab machines.

### **2. Dependencies**

Ensure you have the following libraries installed in your Python environment:

* **Python 3**
* **NumPy** 
* **Matplotlib**

You can install the required libraries using `pip`:

```bash
pip3 install numpy matplotlib
```
