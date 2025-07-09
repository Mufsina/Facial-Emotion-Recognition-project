# 🎭 Facial Emotion Detection Using Deep Learning

This project is a deep learning-based application designed to detect human facial emotions from images. It uses a Convolutional Neural Network (CNN) to classify emotions such as **Happy**, **Sad**, **Angry**, **Surprise**, **Fear**, **Disgust**, and **Neutral**. The trained model is deployed using **Streamlit**, allowing users to interact with it through a simple and responsive web interface.

---

## 📌 Features

- ✅ Real-time emotion prediction from image input
- 🧠 CNN-based model trained on labeled facial images
- 🖤 Grayscale image processing (48x48)
- 🌐 Web interface built with Streamlit
- ⏱️ EarlyStopping to prevent overfitting
- 🚀 Supports both local and cloud deployment (Streamlit Cloud)

---

## 🧾 Project Structure

```
facial_emotion_detection/
│
├── app.py                             # Streamlit web app
├── model_training.ipynb               # Notebook for training the model
├── face_emotion_classificationnn.keras  # Trained CNN model
├── Facial_Images/                     # Dataset (Train and Validation)
│   ├── train/
│   └── validation/
├── presentation_slide.pptx            # Project presentation slides
├── flowchart.png                      # Flowchart / architecture diagram
├── project_proposal.pdf               # Project proposal document
├── demo_video.mp4                     # Short demo video of the system
├── README.md                          # Project documentation (this file)
└── requirements.txt                   # Required Python libraries
```

---

## 🚀 How to Run the Project Locally

### 🧩 Prerequisites:
- Python 3.8+
- pip

### 🔧 Installation & Run:

1. **Clone the repository**
```bash
git clone https://github.com/your-username/facial_emotion_detection.git
cd facial_emotion_detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

4. **Use the web interface**
- Upload a facial image (.jpg/.png)
- View the predicted emotion label on screen

---

## 🌐 Online Deployment (Optional)

You can deploy this app using **Streamlit Cloud** for free.

🔗 Deploy here: https://streamlit.io/cloud

### Deployment Steps:
- Push this project to GitHub
- Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
- Connect your GitHub account
- Select the repository and deploy

---

## 📚 Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

### Key Libraries Used:
- `tensorflow`
- `keras`
- `numpy`
- `opencv-python`
- `matplotlib`
- `streamlit`

---

## 🧠 Model Architecture

- 📐 Input: 48x48 grayscale image
- 🧱 5 Convolutional blocks (Conv2D + MaxPooling + Dropout)
- 🔁 Activation: ReLU
- 🧮 Output: Softmax with 7 emotion classes
- ⚙️ Optimizer: Adam
- 📉 Loss Function: Categorical Crossentropy

### Emotion Classes:
- Angry 😠  
- Disgust 🤢  
- Fear 😨  
- Happy 😄  
- Neutral 😐  
- Sad 😢  
- Surprise 😲  

---

## 🎥 Demo Video

A short video demonstrating the app is included in the repository:  
📁 `demo_video.mp4`

Or upload it to YouTube and link it here.

---

## 📊 Flowchart

Visual workflow of the system is available here:  
📁 `flowchart.jpg`

---

## 📄 Project Proposal

Detailed proposal describing objectives, methodology, and tools used:  
📁 `project_proposal.pdf`

---

## 🖥️ Presentation

Project presentation slide is available at:  
📁 `presentation.pdf`

---

## 🙋‍♀️ Developed By

- **Name**: Sonia Akther Mufsina  
- **University**: North East University Bangladesh  
- **Course**: Deep Learning  
- **Supervisor**: Razorshi Prozzwal Talukder  

---

## 📬 Contact

For queries or suggestions, feel free to:
- 📧 Email the developer
- 📂 Open an issue in the repository

---

> 🎓 This project was completed as part of the academic course requirement for **Deep Learning** under the supervision of **Razorshi Prozzwal Talukder**.

---

**Happy Learning! 🚀**
