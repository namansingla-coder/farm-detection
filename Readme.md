# 🌾 Farm Disease Detection

## 📝 Project Description
Farm Disease Detection is a deep learning-based web application built with Streamlit that allows users to detect diseases in farm crops and poultry by uploading images. The application uses trained convolutional neural network (CNN) models to classify images into various disease categories.

## 🚀 Features
- Detects **Poultry Diseases**, **Potato Diseases**, and **Crop Diseases**.
- **User-friendly Interface** with a streamlined image upload and prediction system.
- **Automated Model Downloading** from Google Drive (if missing).
- **Confidence Score Display** with color-coded severity indication.
- **Supports Multiple Disease Categories**:
  - Poultry: Bumblefoot, Fowlpox, Healthy, Coryza, CRD
  - Potato: Early Blight, Late Blight, Healthy
  - Crop: Various diseases from a pre-trained model

## 🛠️ Technologies Used
- **Python**
- **Streamlit** (for UI)
- **TensorFlow/Keras** (for deep learning models)
- **PIL (Pillow)** (for image processing)
- **NumPy** (for data handling)
- **Pickle** (for class indexing)
- **gdown** (for downloading model files from Google Drive)

## 📦 Installation
### 🔧 Prerequisites
Ensure you have **Python 3.7+** installed.

### 📥 Clone the Repository
```sh
git clone https://github.com/namansingla-coder/farm-detection.git
cd farm-detection
```

### 📌 Install Required Dependencies
```sh
pip install -r requirements.txt
```

## 🎯 How to Use
### 🏗️ Run the Streamlit App
```sh
streamlit run app.py
```

### 📤 Upload an Image & Predict
1. Open the app in your browser.
2. Select a disease model (**Poultry Disease, Potato Disease, or Crop Disease**) from the sidebar.
3. Upload an image of the affected plant or poultry.
4. Click **Predict Disease** to analyze and get results.

## 📂 Project Structure
```
📁 farm-disease-detection/
│── 📁 models/                   # Directory for storing trained models
│── 📜 app.py                    # Main Streamlit application
│── 📜 requirements.txt          # Dependencies list
│── 📜 README.md                 # Documentation
```

## 📥 Downloading Models
The Poultry Disease model is automatically downloaded from Google Drive if not found locally.
For Potato and Crop Disease models, place them in the `models/` directory manually.

## 🔬 Model Details
- **Poultry Model**: A CNN trained on multiple poultry disease images.
- **Potato Model**: A deep learning model trained on early and late blight detection.
- **Crop Model**: Multi-class classifier trained on diverse crop disease datasets.

## 🖼️ Image Preprocessing
- Images are resized to `224x224` pixels.
- Pixel values are normalized (`/ 255.0`).
- Batch dimension is added before feeding into the model.

## 🏆 Results Interpretation
- The app displays the **predicted disease name**.
- The **confidence score** is shown with a color-coded severity indicator:
  - **Green** (> 85% confidence) - High confidence
  - **Orange** (60-85% confidence) - Moderate confidence
  - **Red** (< 60% confidence) - Low confidence

## 🎯 Future Enhancements
- Add support for **real-time camera-based disease detection**.
- Train models on **larger and more diverse datasets**.
- Implement **explainable AI (XAI) techniques** to improve model interpretability.

## 🤝 Contribution
Contributions are welcome! If you'd like to improve this project, feel free to submit a pull request.

## 📜 License
This project is licensed under the MIT License.

---
Developed with ❤️ using Streamlit & TensorFlow.

