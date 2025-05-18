# Detection of Football Game Objects Using Deep Learning
This project focuses on detecting key objects in football (soccer) game footage—such as players, ball, goalposts, etc.—using deep learning models. The goal is to build a system capable of identifying and localizing objects of interest to support further analytics or broadcast enhancements.

## 📂 Project Structure
- Detection_of_Football_Game_Objects_Using_Deep_Learning.ipynb - Jupyter Notebook containing the code for training, evaluating, and visualizing object detection on football frames.

- README.md - Project overview and instructions.

---

## 🧠 Models Used
This project leverages the YOLOv7 deep learning model for object detection, benefiting from its speed and accuracy for real-time detection tasks.

---

## 📦 Requirements
Install the required packages using:

```bash
pip install -r requirements.txt
```
Or manually install the main dependencies:
```bash
pip install torch torchvision
pip install opencv-python matplotlib
pip install seaborn pandas numpy
```
Ensure you also clone the YOLOv7 repo if not already done:
```bash
git clone https://github.com/ultralytics/yolov7
cd yolov7
pip install -r requirements.txt
```
---

## 📁 Dataset
- The dataset consists of football match images annotated with bounding boxes for key objects like players, ball, referee, and goalposts.

- Annotation format: YOLO .txt files with class ID and bounding box coordinates.

> Note: Due to licensing, the dataset is not included. You will need to provide your own annotated dataset in the expected format.

---

## 🚀 How to Run
Open the notebook:
1. Detection_of_Football_Game_Objects_Using_Deep_Learning.ipynb
2. Modify the dataset paths as needed.
3. Run the notebook cells sequentially to:
  - Prepare the dataset
  - Train the model
  - Evaluate performance
  - Visualize predictions

---

## 📊 Evaluation Metrics
- Precision, Recall, mAP (mean Average Precision) at IoU thresholds
- Visual comparison of predicted vs actual bounding boxes

---

## 📸 Output Samples
The notebook visualizes the model's predictions on test images, showing bounding boxes and class labels for detected football objects.

---

## ✅ Features
- Custom object detection using YOLOv7
- Model training and inference
- Performance metrics
- Visualization of detection results

---

## 🔧 Future Work
- Support for video stream detection
- Deployment via a Flask app or Streamlit
- Integration with tracking algorithms (e.g., DeepSORT)
