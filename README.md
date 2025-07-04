

E-Waste Generation Classification

Project Description

This project focuses on building an intelligent image classification system to categorize **E-Waste into 10 distinct classes**, such as batteries, cables, chargers, etc. The primary objective is to help streamline the sorting process of electronic waste for proper recycling and disposal.

The model architecture is based on **EfficientNetV2B0**, a state-of-the-art convolutional neural network optimized for both speed and accuracy. We leveraged **transfer learning** to benefit from pre-trained weights, significantly improving model convergence and performance on a relatively small dataset.

To enhance generalization and avoid overfitting, **data augmentation techniques** (such as rotation, flipping, and zoom) and **dropout layers** were added during training. The model was evaluated using **accuracy and loss metrics**, as well as a **confusion matrix** to inspect misclassifications.

For practical deployment, the trained model was integrated into a user-friendly **Gradio interface**, allowing real-time image-based predictions of e-waste categories with a simple drag-and-drop or file upload feature.

---

Dataset Structure

The dataset is hierarchically organized for supervised learning:
`train/` – Contains images used for training the model. Includes 10 subfolders, each representing a class label.
`val/` – Used for validation during training to monitor performance on unseen data and tune hyperparameters.
`test/` – Reserved for final evaluation of model accuracy and robustness.

Each subfolder inside these directories corresponds to a specific e-waste category (e.g., **mobile phones, printers, remote controls**, etc.), enabling class-wise training and testing.

---

Model Performance

* The model achieved a **validation accuracy of over 95%**, showcasing excellent classification capability across all e-waste categories.
* Performance metrics like **loss curves** were monitored throughout the training phase to ensure steady learning and avoid overfitting.
* A **confusion matrix** was plotted to analyze per-class accuracy and identify any confusion between visually similar items.
* **Sample predictions** on test images further validated the model's practical effectiveness and reliability.

---

Tools & Technologies

The following tools and libraries were used in developing and deploying the model:
**Python** – Primary programming language used for scripting and model development.
**TensorFlow / Keras** – For building and training the deep learning model using EfficientNetV2B0.
**EfficientNetV2B0** – A modern, lightweight CNN architecture used with pre-trained weights for transfer learning.
**Gradio** – Used to create a simple and interactive web interface for real-time predictions.
**Matplotlib / Seaborn** – For plotting accuracy/loss curves and the confusion matrix.
**Scikit-learn** – For generating evaluation metrics and preprocessing support.

---

