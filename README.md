# 🐶🐱 CNN-Based Dog vs. Cat Classifier

## 📌 Project Overview
This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow** and **Keras** to classify images of dogs and cats. The model is trained on a dataset containing images of cats and dogs and can predict whether a given image contains a cat or a dog.

## 📂 Dataset
The dataset should be structured as follows:
```
/path/to/dataset/
    ├── Cat/
    │   ├── cat1.jpg
    │   ├── cat2.jpg
    │   ├── ...
    ├── Dog/
    │   ├── dog1.jpg
    │   ├── dog2.jpg
    │   ├── ...
```

Ensure that the dataset contains enough images for training and validation.

## ⚙️ Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/cnn-dog-cat-classifier.git
   cd cnn-dog-cat-classifier
   ```
2. Install dependencies:
   ```sh
   pip install tensorflow numpy matplotlib opencv-python scikit-learn
   ```

## 🚀 Training the Model
Run the following script to train the CNN model:
```python
python train.py
```

This script:
- Loads the dataset
- Preprocesses the images (resizing and normalization)
- Trains the CNN model
- Saves the trained model as `cats_vs_dogs_cnn.h5`

## 📊 Model Architecture
The CNN model consists of:
- **Three convolutional layers** with `ReLU` activation
- **Max pooling** layers to reduce dimensionality
- **Flatten layer** to convert data into a 1D array
- **Fully connected (Dense) layers** with dropout to prevent overfitting
- **Softmax activation** for classification

## 🖼️ Testing the Model
Use the following script to test the model on new images:
```python
python predict.py --image /path/to/image.jpg
```

## 📌 Example Prediction Output
```
Prediction: Dog 🐶
```

## 💾 Saving and Loading the Model
To save the model:
```python
model.save("cats_vs_dogs_cnn.h5")
```
To load a pre-trained model:
```python
from tensorflow.keras.models import load_model
model = load_model("cats_vs_dogs_cnn.h5")
```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙌 Contributing
Feel free to contribute by submitting issues or pull requests. Suggestions and improvements are always welcome!

---

⭐ **If you found this project useful, don't forget to give it a star!** ⭐
