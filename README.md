# 🧠 NeuroVoice AI  
### AI-Powered Parkinson’s Disease Detection from Voice Input  

![NeuroVoice AI](https://img.shields.io/badge/AI%20For%20Healthcare-NeuroVoiceAI-blue?style=for-the-badge&logo=ai)

---

## 📘 Project Overview  
*NeuroVoice AI* is a locally running *AI-based diagnostic system* that detects early signs of Parkinson’s disease using a person’s voice sample.  
Developed entirely in *Python (Visual Studio Code), the app extracts unique speech biomarkers such as jitter, shimmer, MFCCs, and pitch variations to predict Parkinson’s risk using a trained **Random Forest machine learning model*.  
The system runs fully offline, ensuring *privacy, reliability, and accessibility* even in low-connectivity areas.

---

## 🎯 Aim  
To design and implement an *AI-powered voice analysis tool* that enables *early detection of Parkinson’s disease* through speech features — making screening fast, non-invasive, and affordable.

---

## ⚙ Tech Stack  
| Component | Technology |
|------------|-------------|
| *Platform* | Visual Studio Code (VS Code) |
| *Programming Language* | Python 3.11 |
| *Framework* | Flask |
| *Audio Processing* | Librosa, SoundFile |
| *ML Model* | Random Forest Classifier |
| *Libraries Used* | NumPy, Pandas, Scikit-learn, FPDF |
| *UI Components* | HTML, CSS, JavaScript |
| *Local Storage* | Temporary Uploads Folder (No Cloud) |

---

## 🧩 Key Features  
✅ Detects Parkinson’s disease from recorded or uploaded voice samples  
✅ Simple, modern web interface (Flask-based)  
✅ Trained AI model on clinical acoustic features  
✅ Generates *PDF report* with patient details & confidence level  
✅ *Privacy Assured:* No data is stored after detection  
✅ Runs *fully offline in VS Code* (no internet required)  


## 🚀 Working Flow  

1. *Run the app* locally using VS Code  
   ```bash
   python app.py
  2. Open your browser and visit
👉 http://127.0.0.1:5000/


3. Record or upload a short .wav or .mp3 voice file (3–5 seconds)


4. The system extracts core voice biomarkers:

Jitter

Shimmer

Harmonics-to-Noise Ratio (HNR)

MFCC Mean & Std

Pitch Mean & Std

Zero Crossing Rate



5. The extracted features are processed by a Random Forest Classifier


6. Result displayed instantly as:

🟢 Healthy

🔴 Parkinson’s Detected



7. Displays confidence percentage and generates a downloadable PDF report




---

🧪 Example Output

Prediction: Parkinson’s Detected  
Confidence: 72.8%  
Status: Moderate Risk

or

Prediction: Healthy  
Confidence: 93.4%  
Status: Normal


---

🧠 Model Information

Parameter	Details

Algorithm	RandomForestClassifier
Dataset	Synthetic + Clinically Inspired Voice Data
Accuracy	85–90% (on test samples)
Core Features	jitter, shimmer, hnr, mfcc, pitch, zcr
Output	Binary Classification (Healthy / Parkinson’s)



---

⚖ Model Performance & Limitations

> The model performs well on clean, consistent voice data but may sometimes misclassify due to microphone noise, accent variation, or recording environment.
This is a known challenge in biomedical AI systems that rely on non-clinical voice datasets.



Future Enhancements:

Include real clinical dataset integration

Apply CNN + LSTM architectures for higher precision

Add live noise filtering and voice normalization

Expand to other neurological conditions (ALS, Dysarthria, etc.)



---

🏁 How to Run (in VS Code)

1. Clone this repository

git clone https://github.com/sadigesharath/NeuroVoiceAI.git
cd NeuroVoiceAI


2. Install dependencies

pip install -r requirements.txt


3. Run the Flask server

python app.py


4. Open browser
Go to: http://127.0.0.1:5000/


5. Record or upload a voice sample → Wait for analysis → Download report.




---

📄 Files in Project

File	Description

app.py	Main Flask application file
train_model.py	ML model training script
parkinsons_model.pkl	Trained AI model
requirements.txt	All project dependencies
/templates/index.html	Frontend layout file
/static/style.css	CSS file for styling
/static/script.js	JavaScript for recording & interactivity
/uploads/keep.txt	Placeholder to keep uploads folder active



---

🧭 Project Impact

🧬 Enables early screening for neurological disorders using AI
💡 Promotes accessible and affordable diagnostics
🌍 Works offline — suitable for rural or low-network areas
🔒 Maintains privacy and local data processing
🎓 Supports AI-driven healthcare innovation


---

👨‍💻 Developed By

Project Title: NeuroVoice AI
Developer: Sadige Sharath and Team
Department: ECE, CMR Engineering College
Hackathon: CSA Hackathon 2025–26
Motto: “Saving lives through smart detection.”
