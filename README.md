## Adult Income Prediction (Machine_Learning.py)

This project predicts whether a person's income exceeds \$50K/year based on census data using classification and regression techniques.

### Features:

* **Data Preprocessing**: Missing value handling, one-hot encoding, binary target transformation.
* **Models Used**:

  * Logistic Regression
  * Decision Tree
  * Random Forest
* **Ensemble Techniques**: Bagging, Boosting, and Stacking.
* **Evaluation**: Accuracy, Recall, Cross-Validation, Monte Carlo Simulation.
* **Dimensionality Reduction**: PCA applied to enhance performance.
* **Regression Analysis**: Age prediction using ensemble regressors.

### Final Result:

Logistic Regression performed the best with \~84% accuracy after PCA-based feature extraction.

---

## CAPTCHA Recognition Using CNN (Captcha_CNN.py)

A multi-character CAPTCHA recognition system using CNNs, SVMs, and Random Forests on image-based CAPTCHA data.

### Workflow:

* **Data Preprocessing**: Grayscale conversion, resizing, label encoding.
* **Models Used**:

  * CNN (multi-output for 4-character CAPTCHAs)
  * SVM
  * Random Forest
* **Hyperparameter Tuning**: Performed using `Keras Tuner` and `RandomizedSearchCV`.
* **Evaluation**: Accuracy comparison across models and visualization of training metrics.

### Final Result:

CNN outperformed traditional methods with highest sequence-level accuracy after tuning.


---

## Chatbot with NLP (Chatbot.py)

An interactive Q\&A chatbot for Google Devices using NLP and intent classification.

### Features:

* **Text Preprocessing**: Lemmatization, tokenization, synonym replacement.
* **Intent Detection**: Uses TF-IDF + Random Forest Classifier.
* **Entity Recognition**: Regex-based pattern matching for devices and categories.
* **Similarity Matching**: Retrieves closest known Q\&A from dataset.
* **Deployment**: Implemented via Gradio for interactive web UI.

### Dataset:

Preloaded from an Excel file about Google device questions and answers.

---

## FrozenLake Reinforcement Learning Agent (reinforcement_learning.py)

Trains an agent to solve the FrozenLake-v1 environment using two different RL algorithms.

### Algorithms Implemented:

* **Q-Learning**: Off-policy TD control for value approximation.
* **Monte Carlo**: First-visit Monte Carlo method for policy evaluation and improvement.

### Features:

* Epsilon-greedy policy for exploration
* Q-table update for state-action values
* Average reward and episode length visualization
* Performance comparison between Q-learning and Monte Carlo


