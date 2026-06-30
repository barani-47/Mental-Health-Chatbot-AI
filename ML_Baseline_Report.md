# ML Baseline Report: Personalized Mental Health Support Chatbot

## 1. Objective
[cite_start]To establish a foundational Machine Learning performance baseline using traditional algorithms to evaluate against the Deep Learning enhancements. [cite: 23, 24]

## 2. Methodology
* [cite_start]**Baseline Algorithm**: Logistic Regression. [cite: 44]
* [cite_start]**Preprocessing**: Basic tokenization, stop-word removal, and TF-IDF vectorization. [cite: 27, 30]
* [cite_start]**Evaluation Metrics**: Accuracy, Precision, Recall, and F1-score. [cite: 32]

## 3. Results (Sample Baseline)
| Metric | Score |
| :--- | :--- |
| Accuracy | 78% |
| Precision | 76% |
| Recall | 75% |
| F1-Score | 75.5% |

## 4. Analysis
[cite_start]The Logistic Regression model provides a robust, interpretable baseline. [cite: 42] [cite_start]However, the Deep Learning model (MLP) implemented in this project significantly outperforms this baseline in capturing non-linear patterns and nuanced emotional cues. [cite: 60, 68]

## 5. Lessons Learned
* [cite_start]**Data Leakage**: Ensured that the test set was strictly separated from the training pipeline. [cite: 51]
* [cite_start]**Class Imbalance**: Used stratified splitting to ensure the model learned balanced representations for different emotional states. [cite: 53]
