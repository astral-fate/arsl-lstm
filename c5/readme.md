# Chapter 5: System Implementation

This chapter details the implementation phase of the Arabic Sign Language (ArSL) translator system. It covers the setup of the development environment, the tools and languages utilized, the collection and preparation of the dataset, the development and training of the deep learning models, model integration into a pipeline, and the approach to system testing.

## 1. Tools

The following tools and libraries were essential for the development and implementation of the ArSL translator:

| Tool/Library         | Description                                                                 | Used For                                                                     |
| :------------------- | :-------------------------------------------------------------------------- | :--------------------------------------------------------------------------- |
| **Python**           | A versatile, high-level programming language with extensive ML libraries.     | Core development language for scripting, data processing, model building       |
| **Jupyter Notebook** | An interactive computing environment.                                       | Experimentation, data exploration, model prototyping, documentation          |
| **Flask**            | A lightweight WSGI web application framework in Python.                     | Creating the backend API server to host the model and handle predictions     |
| **PyTorch**          | An open-source machine learning framework based on the Torch library.       | Building, training, and evaluating deep learning models (esp. LSTM variants) |
| **OpenCV (cv2)**     | Library for computer vision tasks.                                          | Image loading, processing, augmentation, potentially keypoint visualization  |
| Conda                | An open-source package management and environment management system.          | Managing project dependencies and creating isolated environments.            |
| Pandas               | A library providing high-performance, easy-to-use data structures and tools. | Data loading, manipulation, and analysis during preprocessing.               |
| NumPy                | Fundamental package for scientific computing with Python.                     | Numerical operations, array manipulation.                                    |
| Matplotlib/Seaborn | Libraries for data visualization.                                             | Plotting dataset distributions, training curves, and results analysis.       |
| Scikit-learn         | Machine learning library for various tasks including data splitting.        | Splitting data into training and testing sets, performance metrics calculation. |
| MediaPipe            | Google's framework for building perception pipelines.                         | Hand keypoint (landmark) extraction for word models.                         |

## 2. Data Collection and Preprocessing

### 2.1 Data Collection

A significant effort was dedicated to constructing a robust dataset for ArSL recognition. Data was gathered and combined from various publicly available sources to increase the diversity and volume of training examples. Key source datasets included:

**1. Mendeley ArSL Dataset:**
*   Source: <https://data.mendeley.com/datasets/y7pckrw6z2/1>
*   Size: ~54,000 images
*   Characteristics: Images without background.
    *   *Image Example:*
        ![ArSL Alphabet Chart](page_3_image.png) *(Placeholder for image on page 3)*

**2. Roboflow AASL Dataset (Dataset 2):**
*   Source: <https://universe.roboflow.com/rgb-arsl/rgb-arabic-alphabet-sign-language-aasl-dataset/dataset/12>
*   Size: 7,286 images
*   Characteristics: YOLO format, images without background.
*   Distribution:
    | Class Label   | Image Count | Class Label    | Image Count |
    | :------------ | :---------- | :------------- | :---------- |
    | Ain           | 244         | Sad            | 261         |
    | Al            | 268         | Seen           | 231         |
    | Alef          | 280         | Sheen          | 265         |
    | Beh           | 299         | Tah            | 216         |
    | Dad           | 259         | Teh            | 300         |
    | Dal           | 229         | Teh Marbuta    | 0           |
    | Feh           | 250         | Thal           | 196         |
    | Ghain         | 225         | Theh           | 291         |
    | Hah           | 238         | Waw            | 232         |
    | Heh           | 247         | Yeh            | 266         |
    | Jeem          | 202         | Zah            | 223         |
    | Kaf           | 250         | Zain           | 190         |
    | Khah          | 242         | ain (lower)    | 244         |
    | Laa           | 259         | alef (lower)   | 280         |
    | Lam           | 233         | al (lower)     | 268         |
    | Meem          | 246         | beh (lower)    | 299         |
    | Noon          | 229         | dad (lower)    | 259         |
    | Qaf           | 197         | dal (lower)    | 229         |
    | Reh           | 218         | feh (lower)    | 250         |
    |               |             | ghain (lower)  | 225         |
    *   *Image Example:*
        ![Roboflow Sample Images](page_4_image_grid.png) *(Placeholder for image grid on page 4)*

**3. Kaggle ArSL No Background v2 (Dataset 3):**
*   Source: <https://www.kaggle.com/datasets/rabieelkharoua/arsl-no-background-v2>
*   Size: 6,985 images in total.
*   Characteristics: No background images.
*   Distribution:
    | Class Label | Image Count | Class Label | Image Count |
    | :---------- | :---------- | :---------- | :---------- |
    | Ain         | 223         | Sad         | 223         |
    | Al          | 259         | Seen        | 256         |
    | Alef        | 275         | Sheen       | 265         |
    | Beh         | 290         | Tah         | 199         |
    | Dad         | 217         | Teh         | 279         |
    | Dal         | 191         | Teh Marbuta | 217         |
    | Feh         | 234         | Thal        | 147         |
    | Ghain       | 206         | Theh        | 273         |
    | Hah         | 217         | Waw         | 215         |
    | Heh         | 239         | Yeh         | 221         |
    | Jeem        | 182         | Zah         | 201         |
    | Kaf         | 258         | Zain        | 163         |
    | Khah        | 212         |             |             |
    | Laa         | 248         |             |             |
    | Lam         | 231         |             |             |
    | Meem        | 233         |             |             |
    | Noon        | 217         |             |             |
    | Qaf         | 200         |             |             |
    | Reh         | 194         |             |             |
    *   *Image Example:*
        ![Kaggle No BG Sample Images](page_5_image_grid.png) *(Placeholder for image grid on page 5)*

**4. Kaggle Arabic Sign Language Unaugmented (Dataset 4):**
*   Source: <https://www.kaggle.com/datasets/sabribelmadoui/arabic-sign-language-unaugmented-dataset>
*   Size: 5,811 images across 28 classes.
*   Characteristics: YOLO format, includes background. Relatively well-balanced.
*   Distribution:
    | Class | Image Count | Class | Image Count |
    | :---- | :---------- | :---- | :---------- |
    | ALIF  | 194         | DHAA  | 205         |
    | BAA   | 193         | AYN   | 203         |
    | TA    | 199         | GHAYN | 206         |
    | THA   | 201         | FAA   | 205         |
    | JEEM  | 200         | QAAF  | 205         |
    | HAA   | 200         | KAAF  | 210         |
    | KHAA  | 200         | LAAM  | 204         |
    | DELL  | 201         | MEEM  | 204         |
    | DHELL | 211         | NOON  | 203         |
    | RAA   | 215         | HA    | 203         |
    | ZAY   | 207         | WAW   | 206         |
    | SEEN  | 218         | YA    | 203         |
    | SHEEN | 226         |       |             |
    | SAD   | 235         |       |             |
    | DAD   | 237         |       |             |
    | TAA   | 217         |       |             |
*   Summary Stats: Avg per class: 207.5, Min: 193 (BAA), Max: 237 (DAD).
    *   *Image Example:*
        ![Kaggle Unaugmented Sample Images](page_7_image_grid.png) *(Placeholder for image grid on page 7)*

**5. Kaggle Arabic Sign Language Dataset 2022 (Dataset 5):**
*   Source: <https://www.kaggle.com/datasets/ammarsayedtaha/arabic-sign-language-dataset-2022>
*   Size: 14,202 images across 32 classes.
*   Characteristics: YOLO format, includes background. Relatively balanced.
*   Distribution:
    | Class | Image Count | Class | Image Count |
    | :---- | :---------- | :---- | :---------- |
    | ain   | 448         | ra    | 427         |
    | al    | 450         | saad  | 450         |
    | aleff | 447         | seen  | 450         |
    | bb    | 447         | sheen | 450         |
    | dal   | 401         | ta    | 450         |
    | dha   | 450         | taa   | 444         |
    | dhad  | 439         | thaa  | 451         |
    | fa    | 450         | thal  | 450         |
    | gaaf  | 444         | toot  | 450         |
    | ghain | 450         | waw   | 412         |
    | ha    | 450         | ya    | 448         |
    | haa   | 449         | yaa   | 450         |
    | jeem  | 448         | zay   | 440         |
    | kaaf  | 446         |       |             |
    | khaa  | 450         |       |             |
    | la    | 450         |       |             |
    | laam  | 440         |       |             |
    | meem  | 450         |       |             |
    | nun   | 421         |       |             |
*   Summary Stats: Avg per class: 444, Min: 401 (dal), Max: 451 (thaa).

**6. KArSL Dataset KFUPM:**
*   Source: Institutional (King Fahad University)
*   Size: 85,167 images.
*   Total classes: 502
*   Alphabet Subset Distribution:
    *   Total Classes: 39 Alphabet
    *   Total Images: 44,419
    | Class      | Count | Percentage | Class | Count | Percentage |
    | :--------- | :---- | :--------- | :---- | :---- | :--------- |
    | aleff      | 1771  | 3.99%      | waw   | 1441  | 3.24%      |
    | bb         | 1701  | 3.83%      | ain   | 1441  | 3.24%      |
    | class_0061 | 1677  | 3.78%      | seen  | 1430  | 3.22%      |
    | class_0060 | 1673  | 3.77%      | laam  | 1429  | 3.22%      |
    | Jiim       | 1555  | 3.50%      | raa   | 1427  | 3.21%      |
    | sheen      | 1517  | 3.42%      | tha   | 1427  | 3.21%      |
    | haa        | 1517  | 3.42%      | qaaf  | 1427  | 3.21%      |
    | ghayn      | 1500  | 3.38%      | saad  | 1421  | 3.20%      |
    | ta         | 1499  | 3.37%      | noon  | 1417  | 3.19%      |
    | haah       | 1498  | 3.37%      | daad  | 1415  | 3.19%      |
    | zay        | 1466  | 3.30%      | dal   | 1413  | 3.18%      |
    | faa        | 1462  | 3.29%      | kaaf  | 1022  | 2.30%      |
    | taa        | 1459  | 3.28%      |       |       |            |
    | meem       | 1458  | 3.28%      |       |       |            |
    | zaa        | 1457  | 3.28%      |       |       |            |
    | kha        | 1449  | 3.26%      |       |       |            |
    | thal       | 1448  | 3.26%      |       |       |            |
    | class_0059 | 1446  | 3.26%      |       |       |            |
    *   *Image Example:*
        ![KFUPM Sample Images](page_9_image_grid.png) *(Placeholder for image sequence on page 9)*

**Initial Combined Dataset Summary (Before Balancing/Preprocessing):**
*   Total Classes: 29
*   Total Images: 111,324
*   Initial Image Distribution per Class:
    | Class | Image Count | Percentage |
    | :---- | :---------- | :--------- |
    | bb    | 4422        | 3.97%      |
    | ghain | 4339        | 3.90%      |
    | fa    | 4306        | 3.87%      |
    | ain   | 4226        | 3.80%      |
    | dhad  | 4215        | 3.79%      |
    | laam  | 4108        | 3.69%      |
    | ya    | 4042        | 3.63%      |
    | la    | 4039        | 3.63%      |
    | saad  | 3989        | 3.58%      |
    | ha    | 3982        | 3.58%      |
    | khaa  | 3968        | 3.56%      |
    | sheen | 3965        | 3.56%      |
    | jeem  | 3937        | 3.54%      |
    | ta    | 3918        | 3.52%      |
    | meem  | 3906        | 3.51%      |
    | thaa  | 3902        | 3.51%      |
    | nun   | 3874        | 3.48%      |
    | gaaf  | 3776        | 3.39%      |
    | seen  | 3774        | 3.39%      |
    | kaaf  | 3710        | 3.33%      |
    | haa   | 3709        | 3.33%      |
    | ra    | 3707        | 3.33%      |
    | waw   | 3645        | 3.27%      |
    | dal   | 3639        | 3.27%      |
    | thal  | 3627        | 3.26%      |
    | zay   | 3573        | 3.21%      |
    | dha   | 3457        | 3.11%      |
    | aleff | 3002        | 2.70%      |
    | taa   | 2567        | 2.31%      |

### 2.2 Data Preprocessing

Observing the class imbalance in the initial combined dataset, a preprocessing step was performed to create a more balanced distribution for training. This involved curating the dataset to ensure a consistent number of samples per class for the training and testing phases, primarily through undersampling the more populated classes.

Standard preprocessing steps were applied to enhance model generalization:
*   **Resizing:** Images were resized to a consistent input size (e.g., 224x224 pixels).
*   **Normalization:** Pixel values were normalized, often using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for models using pre-trained backbones.
*   **Data Augmentation:** Techniques like rotation, flipping (especially horizontal flipping), brightness adjustment, RandomResizedCrop, RandomRotation, ColorJitter, and RandomAffine were applied during training to increase data variability and model robustness.
*   **Tensor Conversion:** Images were converted to PyTorch or TensorFlow tensors.
*   **(KFU Specific):** For the KFUPM dataset (used in word experiments), images were potentially cropped to focus only on the hands.
*   **(YOLO Format Handling):** Datasets provided in YOLO format contain bounding box information, which might be used for cropping or localization depending on the model architecture.
*   **Keypoint Extraction (Word Models):** For word recognition experiments, MediaPipe Hands was used to extract hand keypoints (landmarks) from each frame. Padding with zeros handled cases with missing hands.

**Final Dataset Split and Distribution (Example for Alphabet Model):**
The curated dataset was split into training and testing sets (e.g., 70% train, 30% test).
*   Total Training Images: 52,084
*   Total Test Images: 22,359
*   Total Images in Final Dataset: 74,443

*Final Image Distribution (Post-Balancing and Splitting - Example):*
| Class | Total Images | Train Count | Test Count |
| :---- | :----------- | :---------- | :--------- |
| ain   | 4226         | 1796        | 771        |
| aleff | 3002         | 1796        | 771        |
| bb    | 4422         | 1796        | 771        |
| dal   | 3639         | 1796        | 771        |
| dha   | 3457         | 1796        | 771        |
| dhad  | 4215         | 1796        | 771        |
| fa    | 4306         | 1796        | 771        |
| gaaf  | 3776         | 1796        | 771        |
| ghain | 4339         | 1796        | 771        |
| ha    | 3982         | 1796        | 771        |
| haa   | 3709         | 1796        | 771        |
| jeem  | 3937         | 1796        | 771        |
| kaaf  | 3710         | 1796        | 771        |
| khaa  | 3968         | 1796        | 771        |
| la    | 4039         | 1796        | 771        |
| laam  | 4108         | 1796        | 771        |
| meem  | 3906         | 1796        | 771        |
| nun   | 3874         | 1796        | 771        |
| ra    | 3707         | 1796        | 771        |
| saad  | 3989         | 1796        | 771        |
| seen  | 3774         | 1796        | 771        |
| sheen | 3965         | 1796        | 771        |
| ta    | 3918         | 1796        | 771        |
| taa   | 2567         | 1796        | 771        |
| thaa  | 3902         | 1796        | 771        |
| thal  | 3627         | 1796        | 771        |
| waw   | 3645         | 1796        | 771        |
| ya    | 4042         | 1796        | 771        |
| zay   | 3573         | 1796        | 771        |

## 3. Model Training

Two main types of models were developed and trained: one for recognizing ArSL alphabets from images and another for recognizing ArSL words from sequences of hand keypoints.

### 3.1 Model Training for Alphabets

#### 3.1.1 CNN (Convolutional Neural Network)

**Model Architecture (TensorFlow/Keras Example):**
A Convolutional Neural Network (CNN) was implemented using the TensorFlow/Keras framework.
*   The architecture consists of stacked convolutional layers with Batch Normalization and ReLU activation.
*   Max Pooling layers followed convolutional blocks for spatial downsampling.
*   After the convolutional blocks, features were flattened.
*   Dropout was used for regularization.
*   Features were fed into fully connected (Dense) layers.
*   The final output layer had 29 units (corresponding to the 29 ArSL alphabet classes used in this configuration) with a Softmax activation function for classification.
    *   *Architecture Diagram:*
        ![CNN Architecture Diagram](page_13_image_cnn_arch.png) *(Placeholder for diagram on page 13)*

**Model Architecture (PyTorch Example - ArSLNet):**
An alternative CNN, named ArSLNet, was implemented using PyTorch.
*   **Preprocessing:** Resizing to 224x224, Conversion to PyTorch tensors, Normalization (ImageNet stats).
*   **Architecture Summary:**
    | Layer (type)   | Output Shape        | Param #     |
    | :------------- | :------------------ | :---------- |
    | Conv2d-1       | [1, 32, 224, 224]   | 896         |
    | ReLU-2         | [1, 32, 224, 224]   | 0           |
    | MaxPool2d-3    | [1, 32, 112, 112]   | 0           |
    | Conv2d-4       | [1, 64, 112, 112]   | 18,496      |
    | ReLU-5         | [1, 64, 112, 112]   | 0           |
    | MaxPool2d-6    | [1, 64, 56, 56]     | 0           |
    | Conv2d-7       | [1, 128, 56, 56]    | 73,856      |
    | ReLU-8         | [1, 128, 56, 56]    | 0           |
    | MaxPool2d-9    | [1, 128, 28, 28]    | 0           |
    | Conv2d-10      | [1, 256, 28, 28]    | 295,168     |
    | ReLU-11        | [1, 256, 28, 28]    | 0           |
    | MaxPool2d-12   | [1, 256, 14, 14]    | 0           |
    | Flatten-13     | [1, 50176]          | 0           |
    | Dropout-14     | [1, 50176]          | 0           |
    | Linear-15      | [1, 512]            | 25,690,112  |
    | ReLU-16        | [1, 512]            | 0           |
    | Dropout-17     | [1, 512]            | 0           |
    | Linear-18      | [1, 32]             | 16,416      |
    | **Total Params** |                     | **26,094,944** |


**Training Configuration (CNN):**
The model was trained using the following setup:
*   **Loss Function:** Categorical Crossentropy.
*   **Optimizer:** Adam (Adaptive Moment Estimation), likely with a tuned learning rate (e.g., 0.001).
*   **Learning Rate Scheduler:** A scheduler such as ReduceLROnPlateau might have been used.
*   **Batch Size:** Standard batch size (e.g., 32 or 64).
*   **Number of Epochs:** Approximately 64 epochs.
*   **Input Image Size:** 224x224 pixels (RGB).
*   **Normalization:** Applied as part of the preprocessing pipeline.
*   **Training Process:** Involved feeding the training data (52,084 images) to the model, optimizing parameters using backpropagation and the Adam optimizer, and monitoring performance on a validation subset. Training was accelerated using GPU resources.

#### 3.1.2 LSTM (Long Short-Term Memory) with Attention

**Model Architecture (ArSLAttentionLSTM - PyTorch):**
An Attention-LSTM model was implemented using PyTorch for alphabet recognition.
*   **Feature Extraction (`self.feature_extractor`):**
    *   Utilizes a pre-trained ResNet18 model.
    *   The final two layers (average pooling and fully connected) of ResNet18 were removed to output spatial feature maps (e.g., `[batch, 512, 7, 7]` for 224x224 input).
*   **Reshaping:**
    *   The 3D feature map `[batch, channels, height, width]` was reshaped into a sequence `[batch, seq_len, features]` suitable for LSTM (e.g., `[batch, 49, 512]`, where seq_len = 7 * 7 = 49).
*   **Recurrent Layer (`self.lstm`):**
    *   A 2-layer bidirectional LSTM network with a hidden size of 512 units processed the sequence.
    *   Dropout (rate=0.5) was applied between LSTM layers for regularization.
*   **Attention Mechanism (`self.attention`):**
    *   Applied to the LSTM output sequence (`lstm_out`).
    *   Consisted of two linear layers with a Tanh activation, producing attention weights for each sequence position.
    *   A context vector was computed via a weighted sum of LSTM outputs using these weights.
*   **Classification (`self.classifier`):**
    *   A sequential block processed the attention context vector.
    *   Contained two fully connected layers (1024 -> 512, 512 -> 256) with ReLU, Batch Normalization, and Dropout (rate=0.5) after each.
    *   A final linear layer mapped the 256 features to the number of ArSL classes (29).

*   **Layer Summary (ArSLAttentionLSTM):**
    | Layer (type)        | Output Shape      | Param #      |
    | :------------------ | :---------------- | :----------- |
    | ResNet18 (Feature)  | [1, 512, 7, 7]    | 11,176,512   |
    | LSTM (Bi, 2-layer)  | [1, 49, 1024]     | 10,502,144   |
    | Attention           | [1, 49, 1]        | 262,657      |
    | Linear-1 (Context->)| [1, 512]          | 524,800      |
    | ReLU-1              | [1, 512]          | 0            |
    | BatchNorm1d-1       | [1, 512]          | 1,024        |
    | Dropout-1           | [1, 512]          | 0            |
    | Linear-2            | [1, 256]          | 131,328      |
    | ReLU-2              | [1, 256]          | 0            |
    | BatchNorm1d-2       | [1, 256]          | 512          |
    | Dropout-2           | [1, 256]          | 0            |
    | Linear-3 (Output)   | [1, 29]           | 7,453        |
    | **Total params:**   |                   | **22,606,430** |
    | Trainable params: |                   | 22,606,430   |
    | Non-trainable params:|                   | 0            |

**Training Configuration (Attention-LSTM):**
*   **Device:** CUDA (if available), otherwise CPU.
*   **Loss Function:** CrossEntropyLoss (potentially with class weights).
*   **Optimizer:** AdamW (learning rate = 0.001, weight decay = 1e-4).
*   **Learning Rate Scheduler:** OneCycleLR (max_lr=0.001, epochs=60, pct_start=0.1).
*   **Batch Size:** 32.
*   **Number of Epochs:** 64.
*   **Input Image Size:** 224x224.
*   **Normalization:** Standard ImageNet transforms (`Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`).
*   **Augmentation:** RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomAffine used during training.
*   **Regularization:** Dropout (rate=0.5) in LSTM and classifier, Weight Decay in optimizer, Gradient Clipping (max_norm=1.0).

### 3.2 Model Training for Words

Experiments for recognizing ArSL words focused on processing sequences of hand keypoints extracted using MediaPipe.

#### 3.2.1 Experiments with Architectures

Several architectures were tested across different experiments:

*   **Data Loading & Filtering:** Loaded image data from class-specific folders (potentially multiple roots). Filtered datasets to predefined subsets of classes (Label Sets 1, 2, 3 below).
*   **Keypoint Extraction:** Used MediaPipe Hands to detect landmarks. Padded with zeros if hands weren't fully detected.
*   **Data Preparation:**
    *   Single frames (flattened keypoints) OR sequences of frames.
    *   For sequences, used a fixed `sequence_length` (e.g., 16 or 20). Padded last frame or sampled frames (linspace) to match the length.
    *   For pooled models (Exp 8), keypoints were averaged across the sequence dimension.
*   **Augmentation:** Horizontal flip was applied in some experiments (`flip_prob=0.5`).

**Label Sets Used:**
*   **Label Set 1 (Used in Exp 1, 2, 3):** 23 Classes (brother, confused, drink, eat, father, girl, greeting, happy, hate, hear, here you are, love, man, mother, sister, sleep, thanks, wake up, walk, welcome, worried, young man, young woman)
*   **Label Set 2 (Used in Exp 4, 6):** 23 Classes (eat, drink, sleep, wake up, hear, walk, love, hate, father, mother, sister, brother, girl, man, young man, young woman, confused, worried, happy, welcome, greeting, here you are, thanks)
*   **Label Set 3 (Used in Exp 8):** 23 Classes (sleep, wake up, hear, silence, rise, walk, love, hate, think, smoke, support, call, beautiful, confused, worried, happy, sad, crying, intelligent, here, there, greeting, thanks)

**Model Architectures Explored:**

1.  **KeypointMLP (Used in Exp 1):** Simple MLP.
    *   Input: Flattened keypoints from a single frame (1 hand, 63 features).
    *   Architecture: `Linear(63, 256) -> BN -> ReLU -> Dropout(0.5) -> Linear(256, 128) -> BN -> ReLU -> Dropout(0.5) -> Linear(128, num_classes)`

2.  **KeypointMLP_v2 (Used in Exp 3, 8):** Improved MLP with more capacity.
    *   Input: Flattened keypoints (single frame, Exp 3, 126 features) OR pooled sequence features (Exp 8, 126 features).
    *   Architecture: `Linear(126, 512) -> BN -> ReLU -> Dropout -> Linear(512, 256) -> BN -> ReLU -> Dropout -> Linear(256, 128) -> BN -> ReLU -> Dropout(0.5) -> Linear(128, num_classes)`

3.  **KeypointLSTM (Used in Exp 2, 4, 6):** Standard LSTM for sequences.
    *   Input: Sequence of keypoint frames (`batch_size, seq_len, input_features=126`). `seq_len` was 1 in Exp 2, 16 in Exp 4 & 6.
    *   Architecture:
        *   `LSTM(input_size=126, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)`
        *   Classifier on last time step output: `Dropout -> Linear(512, 256) -> ReLU -> Linear(256, num_classes)` (Note: `num_directions * hidden_size = 2 * 256 = 512`)

4.  **KeypointLSTM_Simple (Attempted in Exp 5):** Simplified LSTM.
    *   Input: Sequence of keypoint frames.
    *   Architecture:
        *   `LSTM(...)` as above.
        *   Classifier on last time step output: `Dropout -> Linear(512, num_classes)`

**Training Details (General for Word Models):**
*   **Loss:** CrossEntropyLoss (sometimes with label smoothing = 0.1).
*   **Optimizer:** AdamW (e.g., lr=0.001 or 5e-4, weight_decay=1e-4 or 1e-5).
*   **Scheduler:** StepLR (e.g., step=20/15, gamma=0.1/0.2) or ReduceLROnPlateau (mode='min', patience=10, factor=0.2).
*   **Epochs:** 64 to 120.
*   **Batch Size:** 32, 64, or 128.
*   **Saving:** Best model saved based on training accuracy (Exp 1-4, 6) or validation accuracy (Exp 8).

### 3.4 Model Integration

*   **Model Saving:** Trained models (both alphabet and word models, particularly the best performing ones like the alphabet Attention-LSTM and the word KeypointLSTM) were saved (e.g., as `.pth` or `.h5` files).
*   **Pipeline Creation:** The model loading, preprocessing steps (resizing, normalization, keypoint extraction for words), prediction, and post-processing logic were encapsulated into a reusable pipeline.
*   **API Development (Flask):** A Flask application was created to serve as the backend.
    *   It loads the saved model(s) and the processing pipeline.
    *   It defines API endpoints (e.g., `/predict/alphabet`, `/predict/word`) to receive input data (images or keypoint sequences).
    *   It processes the input using the pipeline, gets predictions from the model, and returns the results (e.g., predicted letter or word) in a structured format like JSON.
*   **Interface Creation:** A user interface (e.g., a web page using HTML/CSS/JavaScript or a desktop application) was developed.
    *   Allows users to upload images or potentially provide video streams.
    *   Sends the user input to the appropriate Flask API endpoint.
    *   Receives the prediction result from the backend and displays it to the user.

## 4. System Testing

Comprehensive testing was conducted to ensure the reliability and usability of the ArSL translator system.

*   **Unit Testing:** Individual components were tested in isolation.
    *   Tested data loading functions.
    *   Tested specific model layers or blocks.
    *   Tested keypoint extraction algorithm logic.
    *   Tested individual functions within the Flask API (e.g., request parsing, response formatting).
*   **Integration Testing:** Verified interactions between components.
    *   Tested the flow from the frontend image upload to receiving a prediction from the Flask backend.
    *   Ensured the preprocessing pipeline integrated correctly with the model prediction step.
    *   Tested the connection and data transfer between the UI and the API.
*   **System Testing:** Evaluated the end-to-end functionality.
    *   Simulated real-world usage scenarios (uploading various sign images/sequences).
    *   Tested the complete system flow from input to displayed output.
*   **User Acceptance Testing (UAT):** Observed potential users interacting with the system.
    *   Gathered feedback on ease of use, clarity of the interface, and overall user experience.
    *   Identified usability issues or areas for improvement based on user feedback.

## 5. Results

This section presents the quantitative results obtained from evaluating the trained models on their respective held-out test sets.

### 5.1 Results of the CNN (Alphabet Model)

**Overall Performance:** The CNN model (likely the TF/Keras variant described) achieved an overall **accuracy of 97.31%** on the test set (22,359 images).

**Classification Report:**
| Class | Precision | Recall | F1-score | Support |
| :---- | :-------- | :----- | :------- | :------ |
| ain   | 0.991     | 0.984  | 0.988    | 771     |
| aleff | 0.973     | 0.991  | 0.982    | 771     |
| bb    | 0.980     | 0.970  | 0.975    | 771     |
| dal   | 0.966     | 0.982  | 0.974    | 771     |
| dha   | 0.974     | 0.983  | 0.979    | 771     |
| dhad  | 0.980     | 0.971  | 0.976    | 771     |
| fa    | 0.936     | 0.960  | 0.948    | 771     |
| gaaf  | 0.956     | 0.935  | 0.946    | 771     |
| ghain | 0.976     | 0.991  | 0.983    | 771     |
| ha    | 0.965     | 0.970  | 0.968    | 771     |
| haa   | 0.962     | 0.981  | 0.971    | 771     |
| jeem  | 0.954     | 0.965  | 0.959    | 771     |
| kaaf  | 0.980     | 0.965  | 0.973    | 771     |
| khaa  | 0.988     | 0.970  | 0.979    | 771     |
| la    | 0.986     | 0.979  | 0.982    | 771     |
| laam  | 0.973     | 0.981  | 0.977    | 771     |
| meem  | 0.937     | 0.981  | 0.958    | 771     |
| nun   | 0.984     | 0.961  | 0.972    | 771     |
| ra    | 0.975     | 0.953  | 0.964    | 771     |
| saad  | 0.965     | 0.987  | 0.976    | 771     |
| seen  | 0.962     | 0.991  | 0.976    | 771     |
| sheen | 0.995     | 0.978  | 0.986    | 771     |
| ta    | 0.972     | 0.960  | 0.966    | 771     |
| taa   | 0.961     | 0.984  | 0.972    | 771     |
| thaa  | 0.995     | 0.960  | 0.977    | 771     |
| thal  | 0.987     | 0.968  | 0.977    | 771     |
| waw   | 0.993     | 0.971  | 0.982    | 771     |
| ya    | 0.981     | 0.981  | 0.981    | 771     |
| zay   | 0.980     | 0.966  | 0.973    | 771     |
| --- | --- | --- | --- | --- |
| **accuracy** |           |        | **0.973** | **22359** |
| **macro avg** | **0.973** | **0.973** | **0.973** | **22359** |
| **weighted avg** | **0.973** | **0.973** | **0.973** | **22359** |

**Interpretation:**
*   High Precision/Recall: Most classes > 0.95, indicating low false positives and negatives.
*   High F1-Scores: Consistently high, showing good balance between precision/recall.
*   Slight Variations: Classes like 'fa' (F1: 0.948), 'gaaf' (F1: 0.946), 'meem' (F1: 0.958) performed slightly lower, suggesting they might be harder to distinguish.

**Training and Validation Curves:**
[Image: Training/Validation Loss and Accuracy Curves for CNN (Page 16)] *(Placeholder)*
*   **Loss Curves (Left):** Training loss (blue) decreases steadily. Test/validation loss (orange) tracks closely, indicating good generalization and minimal overfitting. Both plateau, suggesting convergence.
*   **Accuracy Curves (Right):** Training accuracy (blue) rises rapidly to ~96-97%. Test/validation accuracy (orange) mirrors this, reaching ~97%, confirming generalization. Close tracking reinforces lack of significant overfitting.

**Confusion Matrix:**
[Image: Confusion Matrix for CNN (Page 17)] *(Placeholder)*
*   **Strong Diagonal:** High values (dark blue) along the diagonal confirm high accuracy. Most classes > 740 correct predictions (out of 771).
*   **Off-Diagonal (Misclassifications):** Generally low (light blue/white). Minor confusions observed:
    *   'fa' confused with 'ha' (13 instances).
    *   'jeem' confused with 'khaa' (13 instances).
    *   'ta' confused with 'dha' (17 instances).
    *   'meem' confused with 'seen' (7) and 'thaa' (7).
    *   Likely due to visual similarities in hand shape/orientation.
*   **Balanced Test Set:** Numbers represent absolute confusion counts.

### 5.2 Result of the LSTM (Alphabet Model - Attention-LSTM)

**Overall Performance:** The Attention-LSTM model achieved a **Test Accuracy of 98.46%** with a Test Loss of 0.1075.

**Classification Report (Attention-LSTM):**
| Class        | Precision  | Recall     | F1-score   | Support       |
| :----------- | :--------- | :--------- | :--------- | :------------ |
| ain          | 0.989717   | 0.998703   | 0.994190   | 771.000000    |
| aleff        | 0.993515   | 0.993515   | 0.993515   | 771.000000    |
| bb           | 0.990838   | 0.981842   | 0.986319   | 771.000000    |
| dal          | 0.973180   | 0.988327   | 0.980695   | 771.000000    |
| dha          | 0.981865   | 0.983139   | 0.982502   | 771.000000    |
| dhad         | 0.978177   | 0.988327   | 0.983226   | 771.000000    |
| fa           | 0.966581   | 0.975357   | 0.970949   | 771.000000    |
| gaaf         | 0.974026   | 0.972763   | 0.973394   | 771.000000    |
| ghain        | 0.996084   | 0.989624   | 0.992843   | 771.000000    |
| ha           | 0.988266   | 0.983139   | 0.985696   | 771.000000    |
| haa          | 0.981842   | 0.981842   | 0.981842   | 771.000000    |
| jeem         | 0.983139   | 0.983139   | 0.983139   | 771.000000    |
| kaaf         | 0.994695   | 0.972763   | 0.983607   | 771.000000    |
| khaa         | 0.979355   | 0.984436   | 0.981889   | 771.000000    |
| la           | 0.997368   | 0.983139   | 0.990202   | 771.000000    |
| laam         | 0.968434   | 0.994812   | 0.981446   | 771.000000    |
| meem         | 0.980645   | 0.985733   | 0.983182   | 771.000000    |
| nun          | 0.993395   | 0.975357   | 0.984293   | 771.000000    |
| ra           | 0.993307   | 0.962387   | 0.977602   | 771.000000    |
| saad         | 0.980843   | 0.996109   | 0.988417   | 771.000000    |
| seen         | 0.971033   | 1.000000   | 0.985304   | 771.000000    |
| sheen        | 0.998698   | 0.994812   | 0.996751   | 771.000000    |
| ta           | 0.976654   | 0.976654   | 0.976654   | 771.000000    |
| taa          | 0.997361   | 0.980545   | 0.988882   | 771.000000    |
| thaa         | 0.985696   | 0.983139   | 0.984416   | 771.000000    |
| thal         | 0.986979   | 0.983139   | 0.985055   | 771.000000    |
| waw          | 0.993447   | 0.983139   | 0.988266   | 771.000000    |
| ya           | 0.975858   | 0.996109   | 0.985879   | 771.000000    |
| zay          | 0.984375   | 0.980545   | 0.982456   | 771.000000    |
| ---          | ---        | ---        | ---        | ---           |
| **accuracy** |            |            | **0.984570** | **22359.00000**|
| **macro avg**| **0.984668** | **0.984570** | **0.984573** | **22359.00000**|
| **weighted avg**| **0.984668** | **0.984570** | **0.984573** | **22359.00000**|

**Confusion Matrix Analysis (Attention-LSTM):**
[Image: Confusion Matrix for Attention-LSTM (Page 24 - Figure 1)] *(Placeholder)*
*   **Dominant Diagonal:** Strong diagonal confirms high accuracy (98.46%). Many classes > 740 correct predictions.
*   **Class-Specific Misclassifications:**
    *   Most frequent: 'kaaf' predicted as 'seen' (17 instances).
    *   Others: 'dha' predicted as 'ta' (11 instances); 'gaaf' predicted as 'fa' (11 instances).
    *   Likely due to visual similarities in handshapes/movements.
*   **Support Balance:** Well-balanced test set (771 instances/class) allows direct comparison of misclassification counts.
*   **Potential Impact:** While highly accurate, confusions like 'kaaf'/'seen' could cause misunderstandings in real-time use.

**Training and Validation Curves (Attention-LSTM):**
[Image: Training Loss and Accuracy Curves for Attention-LSTM (Page 25 - Figure 2)] *(Placeholder)*
*   **Loss Curves:** Training loss decreased steadily to a very low value (~0.02). Final test loss was 0.1075.
*   **Accuracy Curves:** Training accuracy rapidly increased, plateauing > 99.5%. Final test accuracy reached 98.46%.
*   **Interpretation:** Curves show successful convergence. The small gap between final training (99.57%) and test (98.46%) accuracy suggests good generalization with minimal overfitting.

### 5.3 Result of the Seq-LSTM (Word Model - KeypointLSTM)

This section focuses on the results of **Experiment 6**, which used the KeypointLSTM model trained on sequences of keypoints from 2 hands, with augmentation, using Label Set 2, and combined data sources. This configuration yielded the highest accuracy among the reported word model experiments.

**Overall Performance (Exp 6):**
*   Best Training Accuracy: 99.95%
*   Test Loss: 0.7623
*   **Test Accuracy: 96.74%**

**Classification Report (Exp 6 - KeypointLSTM):**
| Class         | Precision  | Recall     | F1-score   | Support    |
| :------------ | :--------- | :--------- | :--------- | :--------- |
| eat           | 1.000000   | 1.000000   | 1.000000   | 16.000000  |
| drink         | 0.941176   | 1.000000   | 0.969697   | 16.000000  |
| sleep         | 1.000000   | 0.500000   | 0.666667   | 16.000000  |
| wake up       | 1.000000   | 1.000000   | 1.000000   | 16.000000  |
| hear          | 0.761905   | 1.000000   | 0.864865   | 16.000000  |
| walk          | 1.000000   | 1.000000   | 1.000000   | 16.000000  |
| love          | 1.000000   | 0.937500   | 0.967742   | 16.000000  |
| hate          | 0.888889   | 1.000000   | 0.941176   | 16.000000  |
| father        | 1.000000   | 0.937500   | 0.967742   | 16.000000  |
| mother        | 1.000000   | 1.000000   | 1.000000   | 16.000000  |
| sister        | 1.000000   | 1.000000   | 1.000000   | 16.000000  |
| brother       | 1.000000   | 1.000000   | 1.000000   | 16.000000  |
| girl          | 1.000000   | 1.000000   | 1.000000   | 16.000000  |
| man           | 1.000000   | 0.937500   | 0.967742   | 16.000000  |
| young man     | 1.000000   | 1.000000   | 1.000000   | 16.000000  |
| young woman   | 1.000000   | 1.000000   | 1.000000   | 16.000000  |
| confused      | 1.000000   | 1.000000   | 1.000000   | 16.000000  |
| worried       | 0.941176   | 1.000000   | 0.969697   | 16.000000  |
| happy         | 1.000000   | 1.000000   | 1.000000   | 16.000000  |
| welcome       | 1.000000   | 0.937500   | 0.967742   | 16.000000  |
| greeting      | 0.941176   | 1.000000   | 0.969697   | 16.000000  |
| here you are  | 1.000000   | 1.000000   | 1.000000   | 16.000000  |
| thanks        | 0.888889   | 1.000000   | 0.941176   | 16.000000  |
| ---           | ---        | ---        | ---        | ---        |
| **accuracy**  |            |            | **0.967391** | **368.000000** |
| **macro avg** | **0.972314** | **0.967391** | **0.964954** | **368.000000** |
| **weighted avg**| **0.972314** | **0.967391** | **0.964954** | **368.000000** |

**Confusion Matrix (Exp 6 - KeypointLSTM):**
[Image: Confusion Matrix for KeypointLSTM Combined (Page 39)] *(Placeholder)*
*(Analysis based on visual inspection of the matrix on page 39):*
*   Very strong diagonal, indicating high accuracy for most word classes.
*   Notable confusions appear minimal. The 'sleep' class seems to have lower recall (8 correct out of 16, confused with 'worried' - class 0235). 'hear' also has perfect recall despite lower precision in the report. Most classes show perfect or near-perfect classification on this test set.

**Comparison with other Word Experiments (Summary from Page 44):**
*   **Sequence vs. Single Frame:** LSTM models on sequences (Exp 4, 6) outperformed MLP models on single frames (Exp 1, 3).
*   **Number of Hands:** Models using 2 hands generally achieved higher accuracies than the 1-hand MLP (Exp 1).
*   **Data Size:** Exp 6 (combined roots, larger test set) significantly outperformed Exp 4 (single root, small test set) using the same architecture (96.74% vs 54.89%).
*   **Model Architecture:** KeypointLSTM (Exp 6, 96.74%) outperformed KeypointMLP_v2 on pooled sequences (Exp 8, 88.86%). LSTM's temporal processing is more effective.
*   **Class Sets:** Performance varied between Exp 6 (Set 2, 96.74%) and Exp 8 (Set 3, 88.86%), indicating varying difficulty among signs.
*   **Saving Strategy:** Exp 8 used validation saving (more robust against overfitting), but its architecture performed worse than Exp 6 (which used training accuracy saving).

**Conclusion from Experiments:** Processing sequences of keypoints using an LSTM model (KeypointLSTM) that incorporates features from two hands and is trained on a larger, combined dataset (like in Exp 6) proved to be the most effective strategy among those explored for word recognition, achieving 96.74% accuracy.

## 6. Conclusion and Future Work (Consolidated)

This chapter detailed the implementation of the ArSL translator, covering tool setup, extensive data aggregation and preprocessing for both alphabets and words, and the development and training of multiple deep learning models (CNNs, LSTMs, MLPs).

**Key Findings:**
*   For **alphabet recognition**, both CNN and Attention-LSTM models achieved high accuracy (97.31% and 98.46% respectively) on a large, balanced dataset. The Attention-LSTM showed a slight edge.
*   For **word recognition** using keypoints, LSTM models processing sequences significantly outperformed MLPs processing single or pooled frames. The KeypointLSTM model trained on combined data sources achieved the highest accuracy (96.74%).
*   Using **features from two hands** and training on **larger, combined datasets** proved crucial for higher performance, especially for word recognition.
*   **Data augmentation** (like horizontal flipping) and appropriate **model saving strategies** (validation-based) are beneficial practices.

**System Components:** Beyond model training, the implementation involved integrating the best models into a pipeline, developing a Flask backend API, and creating a user interface for interaction. Testing included unit, integration, system, and usability evaluations.

**Future Work:**
*   Explore other sequence models (e.g., GRUs, Transformers) for potential improvements, especially for word recognition.
*   Investigate different pooling strategies for MLP models on sequences.
*   Fine-tune hyperparameters for the best-performing models (Attention-LSTM for alphabets, KeypointLSTM for words).
*   Improve the robustness of keypoint extraction, particularly in challenging lighting or occlusion scenarios.
*   Address dataset limitations or errors (e.g., syntax errors mentioned for VER 5/7 if applicable).
*   Expand the vocabulary of recognized signs (both alphabets and words).
*   Conduct more extensive usability testing with target users.

This implementation provides a strong foundation for an ArSL translator, demonstrating high accuracy with current deep learning techniques. Future enhancements can further improve its robustness and practical utility.
