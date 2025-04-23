

## ArSLSign Language Recognition using Attention-LSTM

This section details the architecture, training process, and performance evaluation of the Attention-LSTM model developed for Arabic Sign Language (ArSL) recognition.

**2. Model Architecture**

The core of the system is an Attention-LSTM model, named `ArSLAttentionLSTM`, implemented using PyTorch. This model combines a pre-trained Convolutional Neural Network (CNN) for feature extraction with a Long Short-Term Memory (LSTM) network enhanced by an attention mechanism for sequence processing.

*   **Feature Extraction (`self.feature_extractor`):**
    *   Utilizes a pre-trained ResNet18 model.
    *   The final two layers (average pooling and fully connected classification layer) of ResNet18 are removed to output spatial feature maps. For a standard 224x224 input, this results in a feature map of shape \[batch, 512, 7, 7].

*   **Reshaping:**
    *   The 3D feature map from the CNN \[batch, channels, height, width] is reshaped into a sequence suitable for the LSTM: \[batch, seq_len, features]. In this case, the shape becomes \[batch, 49, 512], where seq_len = 7 * 7 = 49.

*   **Recurrent Layer (`self.lstm`):**
    *   A 2-layer bidirectional LSTM network with a hidden size of 512 units processes the sequence of features.
    *   Dropout (rate=0.5) is applied between LSTM layers for regularization.

*   **Attention Mechanism (`self.attention`):**
    *   Applied to the LSTM output sequence (`lstm_out`).
    *   Consists of two linear layers with a Tanh activation in between, producing attention weights for each position in the sequence.
    *   A context vector is computed by taking a weighted sum of the LSTM outputs based on these attention weights.

*   **Classification (`self.classifier`):**
    *   A sequential block processes the attention context vector.
    *   Contains two fully connected layers (1024 -> 512, 512 -> 256) with ReLU activations, Batch Normalization, and Dropout (rate=0.5) after each.
    *   A final linear layer maps the 256 features to the number of ArSL classes (29).

**Table 1: ArSLAttentionLSTM Layer Summary**

| Layer (type)      | Output Shape     | Param #      |
| :---------------- | :--------------- | :----------- |
| ResNet18          | \[1, 512, 7, 7]  | 11,176,512   |
| LSTM              | \[1, 49, 1024]   | 10,502,144   |
| Attention         | \[1, 49, 1]    | 262,657      |
| Linear-1          | \[1, 512]        | 524,800      |
| ReLU-1            | \[1, 512]        | 0            |
| BatchNorm1d-1     | \[1, 512]        | 1,024        |
| Dropout-1         | \[1, 512]        | 0            |
| Linear-2          | \[1, 256]        | 131,328      |
| ReLU-2            | \[1, 256]        | 0            |
| BatchNorm1d-2     | \[1, 256]        | 512          |
| Dropout-2         | \[1, 256]        | 0            |
| Linear-3 (Output) | \[1, 29]         | 7,453        |
| **Total params:** |                  | **22,606,430** |
| Trainable params: |                  | 22,606,430   |
| Non-trainable params: |              | 0            |

**3. Training Details**

The model was trained under the following settings:

*   **Device:** CUDA (if available), otherwise CPU.
*   **Loss Function:** CrossEntropyLoss with class weights to handle potential data imbalance.
*   **Optimizer:** AdamW (learning rate = 0.001, weight decay = 1e-4).
*   **Learning Rate Scheduler:** OneCycleLR (max_lr=0.001, epochs=60, pct_start=0.1).
*   **Batch Size:** 32.
*   **Number of Epochs:** 64.
*   **Input Image Size:** 224x224.
*   **Normalization:** Images were normalized using standard ImageNet `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
*   **Augmentation:** RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomAffine were used during training.
*   **Regularization:** Dropout (rate=0.5) in LSTM and classifier, Weight Decay in optimizer, Gradient Clipping (max_norm=1.0).

**4. Results and Evaluation**

The Attention-LSTM model achieved a **Test Accuracy of 98.46%** with a Test Loss of 0.1075. The following metrics provide a detailed evaluation:

**Classification Report**
Table 2 presents the precision, recall, and F1-score for each of the 29 ArSL classes on the test set.

**Table 2: Classification Report (Attention-LSTM)**

| Class     | Precision | Recall   | F1-score | Support   |
| :-------- | :-------- | :------- | :------- | :-------- |
| ain       | 0.989717  | 0.998703 | 0.994190 | 771.00000 |
| aleff     | 0.993515  | 0.993515 | 0.993515 | 771.00000 |
| bb        | 0.990838  | 0.981842 | 0.986319 | 771.00000 |
| dal       | 0.973180  | 0.988327 | 0.980695 | 771.00000 |
| dha       | 0.981865  | 0.983139 | 0.982502 | 771.00000 |
| dhad      | 0.978177  | 0.988327 | 0.983226 | 771.00000 |
| fa        | 0.966581  | 0.975357 | 0.970949 | 771.00000 |
| gaaf      | 0.974026  | 0.972763 | 0.973394 | 771.00000 |
| ghain     | 0.996084  | 0.989624 | 0.992843 | 771.00000 |
| ha        | 0.988266  | 0.983139 | 0.985696 | 771.00000 |
| haa       | 0.981842  | 0.981842 | 0.981842 | 771.00000 |
| jeem      | 0.983139  | 0.983139 | 0.983139 | 771.00000 |
| kaaf      | 0.994695  | 0.972763 | 0.983607 | 771.00000 |
| khaa      | 0.979355  | 0.984436 | 0.981889 | 771.00000 |
| la        | 0.997368  | 0.983139 | 0.990202 | 771.00000 |
| laam      | 0.968434  | 0.994812 | 0.981446 | 771.00000 |
| meem      | 0.980645  | 0.985733 | 0.983182 | 771.00000 |
| nun       | 0.993395  | 0.975357 | 0.984293 | 771.00000 |
| ra        | 0.993307  | 0.962387 | 0.977602 | 771.00000 |
| saad      | 0.980843  | 0.996109 | 0.988417 | 771.00000 |
| seen      | 0.971033  | 1.000000 | 0.985304 | 771.00000 |
| sheen     | 0.998698  | 0.994812 | 0.996751 | 771.00000 |
| ta        | 0.976654  | 0.976654 | 0.976654 | 771.00000 |
| taa       | 0.997361  | 0.980545 | 0.988882 | 771.00000 |
| thaa      | 0.985696  | 0.983139 | 0.984416 | 771.00000 |
| thal      | 0.986979  | 0.983139 | 0.985055 | 771.00000 |
| waw       | 0.993447  | 0.983139 | 0.988266 | 771.00000 |
| ya        | 0.975858  | 0.996109 | 0.985879 | 771.00000 |
| zay       | 0.984375  | 0.980545 | 0.982456 | 771.00000 |
| accuracy  |           |          | 0.984570 | 0.98457   |
| macro avg | 0.984668  | 0.984570 | 0.984573 | 22359.00000 |
| weighted avg| 0.984668  | 0.984570 | 0.984573 | 22359.00000 |

**Confusion Matrix Analysis for Attention-LSTM Performance**
The confusion matrix (Figure 1) provides a granular view of the Attention-LSTM model's classification performance across all 29 ArSL classes on the test data.

*   **Dominant Diagonal:** The heatmap clearly shows strong values along the main diagonal, indicating high numbers of correct predictions for nearly all classes. Many classes show over 740 correct predictions out of 771 samples, visually confirming the high overall test accuracy of 98.46%.

*   **Class-Specific Misclassifications:** Examining the off-diagonal elements reveals specific instances where the model experiences confusion:
    *   The most frequent misclassification is the true class 'kaaf' being predicted as 'seen' (17 instances).
    *   Other notable confusions include 'dha' predicted as 'ta' (11 instances) and 'gaaf' predicted as 'fa' (11 instances).
    *   These minor confusions might stem from visual similarities in handshapes or movements between these specific sign pairs.

*   **Support Balance:** The dataset appears well-balanced, with each class having 771 instances in the test set. This allows for a direct comparison of misclassification counts across classes without significant bias from varying sample sizes.

*   **Potential Impact:** Despite the very high accuracy, the few misclassifications highlighted by the confusion matrix are important. In a real-time sign language translation system, confusing signs like 'kaaf' and 'seen' could lead to misunderstandings. This underscores the value of analyzing these specific errors for potential targeted model improvements or data augmentation strategies.

![confusion_matrix](https://github.com/user-attachments/assets/eddd9513-8873-4c0f-a2f1-5fba8f49eb40)


*Figure 1: Confusion Matrix of Attention-LSTM on the test dataset. Axes represent true and predicted labels, and the heatmap illustrates the frequency of classifications.*

**Loss and Accuracy Curves**
Figure 2 displays the training loss and accuracy curves over the 64 epochs.

*   **Loss Curves:** The training loss decreased steadily throughout training, reaching a very low value (around 0.02), indicating the model learned the training data effectively. The final test loss was 0.1075.
*   **Accuracy Curves:** Training accuracy rapidly increased and plateaued at a very high level (above 99.5%). The final test accuracy reached 98.46%.
*   **Interpretation:** The curves demonstrate successful training convergence. The small gap between the final training accuracy (99.57%) and test accuracy (98.46%) suggests good generalization performance with minimal overfitting. The model learned the underlying patterns effectively and applied them well to unseen data.

![improved_training_history](https://github.com/user-attachments/assets/e7bab395-6a85-4b95-8db3-165fc80181fe)


*Figure 2: Training Loss and Accuracy Curves for the Attention-LSTM model.*

---
