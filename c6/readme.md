tools
juypter notebok
flask
pytorch
openCV

1. data collection
2. data preprosessing
cropped the data to hands only (kfu)
extracted the images yolo
provided the strucutr of the data

3. model traning

3.1 model traning for alphabets
3.1.1 CNN
3.1.2 LSTM

3.2 Model traning for words
3.2.1 experments with archtectures

3.4 Model integeration
saving the model into pipeline, creating interface, creating flask application




system testing

integeration testing of the interface

unit testing of the algorithm

user accipatnce testing



results

1. results of the cnn
2. result of the lstm
3. result of the Seq-LSTM

```
<root_directory_1>               <-- e.g., C:\Users\Fatima\Downloads\1\train
|
├── <class_id_1>                 <-- e.g., 0162 (formatted 4-digit string)
|   |
|   ├── <instance_folder_1>      <-- e.g., sequence_001, video_xyz (name doesn't matter)
|   |   |
|   |   ├── frame_0001.png       <-- Individual image frames of the sign
|   |   ├── frame_0002.png
|   |   ├── ...
|   |   └── frame_NNNN.png
|   |
|   ├── <instance_folder_2>
|   |   |
|   |   ├── 0001.jpg
|   |   ├── 0002.jpg
|   |   └── ...
|   └── ... (more instances of class_id_1)
|
├── <class_id_2>                 <-- e.g., 0165
|   |
|   ├── <instance_folder_A>
|   |   |
|   |   ├── image01.jpeg
|   |   ├── image02.jpeg
|   |   └── ...
|   └── ... (more instances of class_id_2)
|
└── ... (more class folders specified in CLASSES_TO_INCLUDE)


<root_directory_2>               <-- e.g., C:\Users\Fatima\Downloads\2\train
|
├── <class_id_1>                 <-- e.g., 0162 (Can have the same classes as root_1)
|   |
|   ├── <instance_folder_X>
|   |   |
|   |   └── ... (frame images)
|   └── ...
|
├── <class_id_3>                 <-- e.g., 0174
|   |
|   └── ... (instances and frames)
|
└── ... (more class folders specified in CLASSES_TO_INCLUDE)

... and similar structures for your TEST_DATA_ROOTS
    e.g., C:\Users\Fatima\Downloads\1\test
          C:\Users\Fatima\Downloads\2\test

```



```
sign_language_app/
├── model_weights/          # Store your .pth model files here
│   ├── improved_arsl_model.pth   # Alphabet model
│   └── sign_word_t5_classifier_best_3d.pth # Word model (or correct name)
├── static/                   # Static files (CSS, JS - if needed later, favicon)
│   └── favicon.ico           # (Optional)
├── templates/                # HTML template
│   └── index.html
├── app.py                    # Main Flask application runner
├── config.py                 # Configuration settings
├── models/                   # Model definitions
│   ├── __init__.py
│   ├── alphabet_model.py
│   └── word_model.py
├── services/                 # Core prediction logic
│   ├── __init__.py
│   └── prediction_service.py
├── utils/                    # Helper utilities
│   ├── __init__.py
│   ├── image_utils.py
│   └── mediapipe_utils.py
└── requirements.txt          # List of dependencies

```
