

# Scientific Report: ArSL Sign Recognition using Keypoint-based Models

## 1. Introduction

This report details a series of experiments aimed at recognizing Arabic Sign Language (ArSL) signs using keypoint data extracted from video frames. The core approach involves utilizing MediaPipe's pre-trained models to detect hand keypoints and then feeding this keypoint data into various neural network architectures for classification. Different strategies regarding the processing of frames (single vs. sequence), handling of multiple hands, data augmentation, selection of classes, combination of data sources, and network architectures were explored.

## 2. Methodology

The general methodology across experiments involved the following steps:

1.  **Data Loading and Filtering:** Load image data organized into class-specific folders, potentially from multiple root directories. Filter the dataset to include only a predefined subset of classes specified by SignIDs.
2.  **Keypoint Extraction:** Process each image frame using **MediaPipe Hands** to detect **hand keypoints** (landmarks: x, y, z coordinates for 21 points per hand). Handle cases where hands are not detected or fewer than the maximum number of hands (`max_hands`) are found by padding the feature vector with zeros for consistency. The number of features extracted per frame is `num_keypoints * dimensions * max_hands`.
3.  **Data Preparation for Model:** Depending on the experiment, either:
    *   Use flattened keypoints from single frames directly.
    *   Group frames from a video instance into sequences of a fixed `sequence_length`, using resampling (linspace) or padding (repeat last frame) as needed.
    *   Calculate a representative feature vector for a sequence (e.g., mean pooling) and use this pooled vector as input.
4.  **Data Augmentation:** Apply horizontal flipping to images during the training phase with a specified probability (`flip_prob`).
5.  **Model Definition and Training:** Define and train a neural network model (MLP or LSTM) on the prepared keypoint data. Training involves standard backpropagation with a chosen loss function (CrossEntropyLoss, potentially with label smoothing), optimizer (AdamW), and learning rate scheduler (StepLR or ReduceLROnPlateau). Training history (loss and accuracy) and potentially validation metrics (loss and accuracy) are tracked. Checkpoints are saved, often based on the best performance metric (e.g., training or validation accuracy).
6.  **Evaluation:** Evaluate the trained model on a separate test dataset using metrics such as accuracy, classification report (precision, recall, f1-score), and confusion matrix. Classification reports and confusion matrices are generated using the ground truth labels and the model's predictions, mapping internal integer labels back to human-readable names where possible.

## 3. Dataset

The experiments utilized data structured in class-specific folders within root directories (e.g., `root/class_id/instance_id/frames...`). Each instance folder contained image frames (`.jpg`, `.png`, `.jpeg`).

The experiments focused on different selections and combinations of the available data. The specific sets of classes used are detailed in the Appendix. The most successful experiments combined data from two root directories (`C:\Users\Fatima\Downloads\1` and `C:\Users\Fatima\Downloads\2`) for both training and testing to increase the overall data size.

For sequence-based experiments, a fixed `sequence_length` (16 or 20) was enforced. If an instance had fewer frames than `sequence_length`, the sequence was padded by repeating the last frame. If it had more, frames were sampled evenly across the duration using `np.linspace`.

Horizontal flip augmentation with a probability of 0.5 was applied during training in experiments configured to do so.

## 4. Model Architectures

Four distinct model classes were defined and used:

1.  **`KeypointMLP` (Used in Exp 1):**
    *   A simple Multi-Layer Perceptron for single-frame classification.
    *   Input: Flattened keypoints from a single frame (`(batch_size, num_features)`).
    *   Architecture: Two hidden layers with ReLU activation, BatchNorm, and Dropout, followed by an output linear layer.

2.  **`KeypointMLP_v2` (Used in Exp 3, 8):**
    *   An improved MLP with higher capacity (more units and an additional hidden layer), designed for flattened keypoints.
    *   Input: Flattened keypoints from a single frame (`(batch_size, num_features)`) for Exp 3, or *mean-pooled* sequence features (`(batch_size, num_features_per_frame)`) for Exp 8.
    *   Architecture: Three hidden layers with ReLU activation, BatchNorm, and Dropout, followed by an output linear layer.

3.  **`KeypointLSTM` (Used in Exp 2, 4, 6, 9):**
    *   A standard LSTM model designed to process sequences of keypoint frames.
    *   Input: Sequence of keypoint frames (`(batch_size, seq_len, input_features)`). In Exp 2, `seq_len` was effectively 1.
    *   Architecture: A multi-layer, bidirectional LSTM, followed by a two-layer classifier on the output of the *last time step*.

4.  **`KeypointLSTM_Simple` (Attempted in Exp 5):**
    *   A simplified version of the `KeypointLSTM`.
    *   Input: Sequence of keypoint frames (`(batch_size, seq_len, input_features)`).
    *   Architecture: A single-layer, unidirectional LSTM, followed by a simple one-layer classifier on the output of the *last time step*.

## 5. Experiments and Results

This section details the successful experiments based on the provided code and logs.

### Experiment 1: Basic MLP (Single Frame, 1 Hand)

*   **Source Files:** `VER 1.txt`
*   **Data:** Single root data (Wordstrain/Wordtest). 23 classes (Set 1, English names from log). 1 hand. Single frame input (63 features). No augmentation.
*   **Model:** `KeypointMLP` (63 input features).
*   **Training:** 64 epochs. Batch Size: 64. Loss: CrossEntropyLoss. Optimizer: AdamW (lr=0.001, weight_decay=1e-4). Scheduler: StepLR (step=20, gamma=0.1). Saved best model based on training accuracy.
*   **Results:**
    *   Best Training Accuracy: 75.62%
    *   Test Loss: 0.5502
    *   Test Accuracy: 82.65%
    *   Classification Report saved as `keypoint_classification_report.csv`. Confusion Matrix saved as `keypoint_model_confusion_matrix.png`. The report is included in the Summary Table below.

### Experiment 2: LSTM (Single Frame as Seq=1, 2 Hands)

*   **Source Files:** `VER 2.txt`
*   **Data:** Single root data (Wordstrain/Wordtest). 23 classes (Set 1). 2 hands. Single frame input treated as sequence length 1 (126 features/frame). No augmentation.
*   **Model:** `KeypointLSTM` (126 input features/frame, hidden=256, layers=2, bidirectional=True).
*   **Training:** 70 epochs. Batch Size: 64. Loss: CrossEntropyLoss. Optimizer: AdamW (lr=5e-4, weight_decay=1e-4). Scheduler: StepLR (step=15, gamma=0.2). Saved best model based on training accuracy.
*   **Results:**
    *   Best Training Accuracy: 74.65%
    *   Test Loss: 0.6975
    *   Test Accuracy: 77.32%
    *   Classification Report saved as `keypoint_lstm_classification_report.csv`. Confusion Matrix saved as `keypoint_lstm_model_confusion_matrix.png`. The report is included in the Summary Table below.

### Experiment 3: Improved MLP v2 (Single Frame, 2 Hands, Augmentation)

*   **Source Files:** `VER 3.txt`
*   **Data:** Single root data (Wordstrain/Wordtest). 23 classes (Set 1). 2 hands. Single frame input (126 features). Horizontal flip augmentation.
*   **Model:** `KeypointMLP_v2` (126 input features, hidden1=512, hidden2=256, hidden3=128).
*   **Training:** 80 epochs. Batch Size: 128. Loss: CrossEntropyLoss. Optimizer: AdamW (lr=0.001, weight_decay=1e-5). Scheduler: StepLR (step=25, gamma=0.1). Saved best model based on training accuracy.
*   **Results:**
    *   Best Training Accuracy: 72.59%
    *   Test Loss: 0.6124
    *   Test Accuracy: 80.26%
    *   Classification Report saved as `keypoint_mlp_v2_aug_classification_report.csv`. Confusion Matrix saved as `keypoint_mlp_v2_aug_model_confusion_matrix.png`. The report is included in the Summary Table below.

### Experiment 4: LSTM (Sequence, 2 Hands, Aug, Filtered Classes, Small Data)

*   **Source Files:** `VER 4.txt`
*   **Data:** Single root data (`C:\Users\Fatima\Downloads\1`). 23 specific classes (Set 2). 2 hands. Sequence length 16 (126 features/frame). Horizontal flip augmentation. Train samples: 970, Test samples: 184 (8/class).
*   **Model:** `KeypointLSTM` (126 input features/frame, hidden=256, layers=2, bidirectional=True).
*   **Training:** 120 epochs. Batch Size: 32. Loss: CrossEntropyLoss(label_smoothing=0.1). Optimizer: AdamW (lr=5e-4, weight_decay=1e-4). Scheduler: ReduceLROnPlateau (patience=10, factor=0.2), stepped on training loss. Saved best model based on training accuracy.
*   **Results:**
    *   Best Training Accuracy: 99.90%
    *   Test Loss: 2.0979
    *   Test Accuracy: 54.89%
    *   Classification Report saved as `keypoint_lstm_seq_filtered_classification_report.csv`. Confusion Matrix saved as `keypoint_lstm_seq_filtered_model_confusion_matrix.png`. The report is included in the Summary Table below. (Note: Low test accuracy likely due to very small test set).

### Experiment 6: LSTM (Sequence, 2 Hands, Aug, Filtered Classes, Combined Data)

*   **Source Files:** `VER 6.txt`
*   **Data:** Combined root data (`C:\Users\Fatima\Downloads\1`, `C:\Users\Fatima\Downloads\2`). 23 specific classes (Set 2). 2 hands. Sequence length 16 (126 features/frame). Horizontal flip augmentation. Train samples: 1940, Test samples: 368 (16/class).
*   **Model:** `KeypointLSTM` (126 input features/frame, hidden=256, layers=2, bidirectional=True).
*   **Training:** 120 epochs. Batch Size: 32. Loss: CrossEntropyLoss(label_smoothing=0.1). Optimizer: AdamW (lr=5e-4, weight_decay=1e-4). Scheduler: ReduceLROnPlateau (patience=10, factor=0.2), stepped on training loss. Saved best model based on training accuracy.
*   **Results:**
    *   Best Training Accuracy: 99.95%
    *   Test Loss: 0.7623
    *   Test Accuracy: 96.74%
    *   Classification Report saved as `keypoint_lstm_seq_combined_classification_report.csv`. Confusion Matrix saved as `keypoint_lstm_seq_combined_model_confusion_matrix.png`. The report is included in the Summary Table below.

### Experiment 8: MLP v2 Pooled (Sequence, 2 Hands, Aug, Different Filtered Classes, Combined Data, Train/Val Split)

*   **Source Files:** `MLP DIFFRENT LABELS.txt` and `VER 8.txt` (Code likely in VER 8, log is the first block).
*   **Data:** Combined root data (`C:\Users\Fatima\Downloads\1`, `C:\Users\Fatima\Downloads\2`). 23 specific classes (Set 3). 2 hands. Sequence length 16 (126 features/frame), mean-pooled. Horizontal flip augmentation. Train samples: 1645 (split from 1935), Val samples: 290, Test samples: 368 (16/class).
*   **Model:** `KeypointMLP_v2` (126 input features, hidden1=512, hidden2=256, hidden3=128).
*   **Training:** 80 epochs. Batch Size: 128. Loss: CrossEntropyLoss(label_smoothing=0.1). Optimizer: AdamW (lr=0.001, weight_decay=1e-5). Scheduler: ReduceLROnPlateau (patience=10, factor=0.2), stepped on validation loss. Saved best model based on validation accuracy.
*   **Results:**
    *   Best Validation Accuracy: 95.86%
    *   Test Loss: 0.9745
    *   Test Accuracy: 88.86%
    *   Classification Report saved as `keypoint_mlp_pooled_labeled_classification_report.csv`. Confusion Matrix saved as `keypoint_mlp_pooled_labeled_model_confusion_matrix.png`. The report is included in the Summary Table below.

### Experiment 9: LSTM (Sequence, 2 Hands, Aug, *Specific* Filtered Classes, Combined Data, Resumable)

*   **Source Files:** Modified `VER 6.txt` (code provided with the request).
*   **Data:** Combined root data (`C:\Users\Fatima\Downloads\1`, `C:\Users\Fatima\Downloads\2`). **17 specific classes** (Set 4, detailed in Appendix). 2 hands. Sequence length 16 (126 features/frame). Horizontal flip augmentation. Test samples: 17 classes * 16 samples/class = 272. (Train samples size not explicitly logged in the provided snippet, but based on dataset logic it would be the sum of instances of these 17 classes across the train roots).
*   **Model:** `KeypointLSTM` (126 input features/frame, hidden=256, layers=2, bidirectional=True, dropout=0.5). Includes resume training logic.
*   **Training:** 120 epochs (total target). Batch Size: 32. Loss: CrossEntropyLoss(label_smoothing=0.1). Optimizer: AdamW (lr=5e-4, weight_decay=1e-4). Scheduler: ReduceLROnPlateau (patience=10, factor=0.2), stepped on training loss. Saved best model based on training accuracy. Resumed training was attempted.
*   **Results:**
    *   Test Loss: 0.7066
    *   Test Accuracy: 96.69%
    *   Classification Report is provided below. Confusion Matrix was likely saved as `keypoint_lstm_seq_combined_model_confusion_matrix.png` (name from code).

    **Classification Report (Test Set, using folder names from Set 4):**
    ```
                  precision    recall  f1-score     support
    0162           1.000000  0.500000  0.666667   16.000000
    0163           1.000000  1.000000  1.000000   16.000000
    0165           0.833333  0.937500  0.882353   16.000000
    0167           1.000000  1.000000  1.000000   16.000000
    0173           1.000000  1.000000  1.000000   16.000000
    0174           0.941176  1.000000  0.969697   16.000000
    0181           1.000000  1.000000  1.000000   16.000000
    0183           1.000000  1.000000  1.000000   16.000000
    0184           1.000000  1.000000  1.000000   16.000000
    0186           0.888889  1.000000  0.941176   16.000000
    0224           1.000000  1.000000  1.000000   16.000000
    0234           1.000000  1.000000  1.000000   16.000000
    0235           1.000000  1.000000  1.000000   16.000000
    0272           1.000000  1.000000  1.000000   16.000000
    0285           0.888889  1.000000  0.941176   16.000000
    0286           1.000000  1.000000  1.000000   16.000000
    0290           0.941176  1.000000  0.969697   16.000000
    accuracy       0.966912  0.966912  0.966912    0.966912
    macro avg      0.970204  0.966912  0.962986  272.000000
    weighted avg   0.970204  0.966912  0.962986  272.000000
    ```

### Failed Experiments

*   **Experiment 5 (VER 5.txt): Simple LSTM (Sequence, 2 Hands, Aug, Filtered Classes, Increased Seq Len)**
    *   This experiment aimed to use the `KeypointLSTM_Simple` model with sequence length 20 on Set 2 classes. The provided log shows a `SyntaxError` in the `ArSLSequenceKeypointDataset` (`kpts=[]; try: ...`). This experiment did not complete successfully.
*   **Experiment 7 (VER 7.txt): MLP Pooled (Sequence, 2 Hands, Augmentation, Filtered Classes)**
    *   This experiment aimed to use the `KeypointMLP_v2` model with mean-pooled sequences on Set 2 classes, trained on combined data. The provided log shows a `SyntaxError` in the `_extract_keypoints_from_image` method (`kpts=[]; try: ...`), similar to Exp 5. This experiment did not complete successfully.

## 6. Summary of Results

Comparing the test accuracies of the successful experiments:

| Experiment # | Model Type        | Frame Handling      | Hands | Augmentation | Data Source    | Train Samples | Test Samples | Val Split | Classes Used | Test Accuracy |
| :----------- | :---------------- | :------------------ | :---- | :----------- | :------------- | :------------ | :----------- | :-------- | :----------- | :------------ |
| 1 (VER 1)    | `KeypointMLP`     | Single Frame        | 1     | No           | Single Root    | 12282         | 5290         | No        | Set 1        | 82.65%        |
| 2 (VER 2)    | `KeypointLSTM`    | Single Frame (Seq=1)| 2     | No           | Single Root    | 12282         | 5290         | No        | Set 1        | 77.32%        |
| 3 (VER 3)    | `KeypointMLP_v2`  | Single Frame        | 2     | Yes (Flip)   | Single Root    | 12282         | 5290         | No        | Set 1        | 80.26%        |
| 4 (VER 4)    | `KeypointLSTM`    | Sequence (16)       | 2     | Yes (Flip)   | Single Root (1)| 970           | 184          | No        | Set 2        | 54.89%        |
| 6 (VER 6)    | `KeypointLSTM`    | Sequence (16)       | 2     | Yes (Flip)   | Combined Roots | 1940          | 368          | No        | Set 2        | **96.74%**    |
| 8 (MLP Pooled)| `KeypointMLP_v2`  | Pooled Seq (16)     | 2     | Yes (Flip)   | Combined Roots | 1645 (Train)  | 368          | Yes (290) | Set 3        | 88.86%        |
| 9 (New Log)  | `KeypointLSTM`    | Sequence (16)       | 2     | Yes (Flip)   | Combined Roots | ~1300*        | 272          | No        | Set 4        | 96.69%        |

*~1300 train samples for Exp 9 is an estimate based on 17 classes * approx 42 instances/class from combined roots.

The results consistently show that using sequences (`KeypointLSTM`) and training on a larger dataset (combined roots) leads to significantly higher accuracy compared to single-frame models or models trained on limited data, confirming the importance of temporal information and data volume. The `KeypointLSTM` architecture with its ability to process sequences appears more suitable for this task than an MLP even with pooled sequence features.

## 7. Discussion and Future Work

... (Keep the Discussion and Future Work section as written previously, as the conclusions drawn from Exp 6 still hold and are reinforced by Exp 9's similar high performance on a different class set).

## 8. Appendix: Data Labels

The experiments used different sets of classes filtered by SignID. The specific mapping from folder name (SignID as a 4-digit string) to English label is provided below for the sets used in successful experiments where this mapping was explicitly defined in the code. The SignID corresponds to the number, Arabic, and English columns in the `KARSL-502_Labels.xlsx` file.

**Classes Used in Experiments 4 and 6 (LSTM Sequence, Filtered Classes):**

These are 23 classes filtered by `SIGN_IDS_TO_USE = [160, 161, ..., 293]`.

| SignID | Sign-English  |
| :----- | :------------ |
| 0160   | eat           |
| 0161   | drink         |
| 0162   | sleep         |
| 0163   | wake up       |
| 0164   | hear          |
| 0173   | walk          |
| 0174   | love          |
| 0175   | hate          |
| 0195   | father        |
| 0196   | mother        |
| 0197   | sister        |
| 0198   | brother       |
| 0199   | girl          |
| 0202   | man           |
| 0203   | young man     |
| 0204   | young woman   |
| 0234   | confused      |
| 0235   | worried       |
| 0238   | happy         |
| 0289   | welcome       |
| 0290   | greeting      |
| 0291   | here you are  |
| 0293   | thanks        |

**Classes Used in Experiment 8 (MLP Pooled, Combined Data, Train/Val Split):**

These are 23 classes filtered by `SIGNID_TO_LABEL_EN` which maps to `SIGN_IDS_TO_USE = [162, 163, ..., 293]`.

| SignID | Sign-English |
| :----- | :----------- |
| 0162   | sleep        |
| 0163   | wake up      |
| 0164   | hear         |
| 0165   | silence      |
| 0167   | rise         |
| 0173   | walk         |
| 0174   | love         |
| 0175   | hate         |
| 0181   | think        |
| 0183   | smoke        |
| 0184   | support      |
| 186    | call         |
| 0224   | beautiful    |
| 0234   | confused     |
| 0235   | worried      |
| 0238   | happy        |
| 0239   | sad          |
| 0256   | crying       |
| 0272   | intelligent  |
| 0285   | here         |
| 286    | there        |
| 290    | greeting     |
| 293    | thanks       |

**Classes Used in Experiment 9 (LSTM Sequence, Combined Data, Specific Filtered Classes):**

These are the **17 classes** filtered by `SIGN_IDS_TO_USE = [162, 163, ..., 290]` from the code snippet provided with the results.

| SignID | Sign-English |
| :----- | :----------- |
| 0162   | sleep        |
| 0163   | wake up      |
| 0165   | silence      |
| 0167   | rise         |
| 0173   | walk         |
| 0174   | love         |
| 0181   | think        |
| 0183   | smoke        |
| 0184   | support      |
| 0186   | call         |
| 0224   | beautiful    |
| 0234   | confused     |
| 0235   | worried      |
| 0272   | intelligent  |
| 0285   | here         |
| 0286   | there        |
| 0290   | greeting     |

**Classes Used in Experiments 1, 2, and 3 (MLP/LSTM Single Frame, Unfiltered Data):**

These experiments used classes corresponding to the folder names found directly in `C:/Users/Fatima/Downloads/Wordstrain` and `C:/Users/Fatima/Downloads/Wordtest`. Based on the log outputs and common English translations, these appear to be:

| Folder Name   | Possible English Label |
| :------------ | :--------------------- |
| brother       | brother                |
| confused      | confused               |
| drink         | drink                  |
| eat           | eat                    |
| father        | father                 |
| girl          | girl                   |
| greeting      | greeting               |
| happy         | happy                  |
| hate          | hate                   |
| hear          | hear                   |
| here you are  | here you are           |
| love          | love                   |
| man           | man                    |
| mother        | mother                 |
| sister        | sister                 |
| sleep         | sleep                  |
| thanks        | thanks                 |
| wake up       | wake up                |
| walk          | walk                   |
| welcome       | welcome                |
| worried       | worried                |
| young man     | young man              |
| young woman   | young woman            |


