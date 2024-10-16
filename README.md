# MLDA EEE Datathon - Credit Card Fraud Detection

## Team Members
- Kalaiselvan Shanmugapriyan
- Gunasekaran Abineshkumar
- Naden Jarel Anthony Koh
- Vivagananthan Muhekarthik
- Darren Ng
- Khaw Boon Kiat

## Overview

This project was developed as part of the MLDA EEE Datathon, focusing on the detection of fraudulent credit card transactions. The dataset used was the **Credit Card Fraud Prediction Dataset by Kelvin Kelue**. The key challenge addressed is the highly imbalanced nature of the dataset, with fraudulent transactions making up only a small fraction of the data. 

## Project Workflow

1. **Data Preprocessing**
    - Dropped the unnecessary `'Unnamed: 0'` column.
    - Split the data into training and test sets, using a 2:1 ratio.
    - Used a correlation matrix to analyze relationships between numerical features.

2. **Feature Engineering**
    - **Numerical Features**: Processed using Random Forest. 
        - Chose Random Forest for its flexibility and ability to handle various tasks.
        - Adjusted thresholds to prioritize minimizing false negatives.
        - Set class weight to "balanced" due to the imbalanced nature of the dataset.
        - Utilized Randomized Search CV to tune hyperparameters.
    - **Categorical Features**: Processed using Natural Language Processing (NLP).
        - NLP was used to reduce high dimensionality and improve model performance by capturing semantic relationships between categories.
        - NLP was preferred over one-hot encoding as it generalizes better for unseen or rare categories.

3. **Model Architecture**
    - **Deep Neural Network (DNN)** with the following layers:
        - Input Layer: Takes in preprocessed data.
        - Hidden Layers: Three hidden layers with 64, 128, and 256 neurons respectively, using ReLU activation functions.
        - Dropout Layers: Added to prevent overfitting by randomly dropping nodes.
        - Output Layer: Provides a probability score between 0 and 1 to classify transactions as fraudulent or legitimate.

4. **Model Evaluation**
    - Achieved a high precision score of **0.9429** in predicting fraudulent transactions.
    - Utilized a confusion matrix to evaluate performance, focusing on reducing both false negatives and false positives.

5. **Explainable AI (XAI) Integration**
    - **LIME (Local Interpretable Model-Agnostic Explanations)** was implemented to help understand why a transaction was marked as fraudulent, addressing the black-box nature of DNNs.
    - The integration of XAI makes the model more explainable and usable in real-life scenarios, particularly in fraud detection.

## Key Features
- **Deep Neural Network (DNN)**: Effective in modeling complex interactions for fraud detection, outperforming traditional machine learning methods.
- **Explainable AI (XAI)**: Enhances the DNN by providing reasons for decisions, which helps stakeholders understand and trust the model.

## Dataset
The dataset used in this project is the **Credit Card Fraud Prediction Dataset** by Kelvin Kelue. It contains transactions, a small fraction of which are fraudulent.

## Installation and Usage

### Requirements
- Python 3.x
- Required libraries (install via `pip`):
  ```bash
  pip install numpy pandas scikit-learn tensorflow lime
  ```

### How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo-url
   ```
2. Run the preprocessing and model training script:
   ```bash
   python fraud_detection.py
   ```
3. Evaluate the model performance using the provided test set.

## Results
The model achieved a **high precision of 0.9429**, with a balanced approach to minimizing false negatives and false positives. The XAI component makes it more interpretable and deployable in real-world applications.

## Future Work
- Experiment with additional feature engineering techniques to improve accuracy.
- Explore advanced methods for handling class imbalance.
- Investigate other model architectures or hybrid models to further improve fraud detection.

---
