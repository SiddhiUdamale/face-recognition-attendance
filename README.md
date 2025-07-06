# 👁️ Face Recognition Attendance System with Eye Blink Detection

This is a real-time face recognition-based attendance system built using **OpenCV**, **face_recognition**, **dlib**, and **Flask**. It also uses **eye blink detection** to ensure a real person is attending, not a photo.

---

## ✅ Features

- 🔍 Detects known faces using webcam
- 👁️ Detects eye blinks (prevents cheating with photos)
- 🕒 Records Name, Date, and Time into CSV
- 📸 Live webcam preview with bounding boxes
- 📁 Modular & easy-to-edit Python code

---

## 🧪 How It Works

1. Upload known faces to `ImagesAttendance/`
2. Start webcam via `attendanceProject.py`
3. System checks for eye blink using facial landmarks
4. Once confirmed, matches the face and logs attendance in `Attendance.csv`

---

## 📁 Folder Structure
Face-Recognition-Attendance/
├── attendanceProject.py
├── Attendance.csv
├── ImagesAttendance/
│ ├── Alice.jpg
│ └── Bob.jpg
├── shape_predictor_68_face_landmarks.dat
├── requirements.txt
└── README.md


---

## 🚀 Run It Locally

```bash
git clone https://github.com/your-username/face-recognition-attendance.git
cd face-recognition-attendance
pip install -r requirements.txt
python attendanceProject.py


