# E-ATM for Card-Less Transactions with Three-Layer Protection 

## Overview
The Cardless E-ATM system enables users to securely perform ATM transactions—such as withdrawals and balance checks—without a physical ATM card. Users authenticate with facial recognition and eye-tracking for a seamless and secure banking experience.

---

## Features
- **Face Detection & Recognition:** Robust authentication via your face, enhancing security and eliminating card-related risks.
- **Eye-Tracking Virtual Keyboard:** Enter your PIN and navigate on-screen options using eye movement and blinking detection.
- **Secure, Cardless Transactions:** Withdraw cash or check your balance safely, without needing a card or touching physical buttons[web:1][web:4].
- **Add Users Easily:** Expand your system by simply adding new users’ images.

---

## Project Structure

E-ATM/
├── Cardless_E-ATM/
│ ├── templates/
│ ├── static/
│ ├── pingenerate.py
│ ├── trainer.py
│ ├── app.py
│ ├── requirements.txt
│ └── models/
├── .gitignore
├── README.md
└── Dataset/


---

## Installation

1. **Clone the repository:**
    ```
    git clone https://github.com/anjalidecode/Cardless_E-ATM.git
    ```

2. **Create and activate Conda environment:**
    ```
    conda create -n E_atm python=3.9
    conda activate E_atm
    ```

3. **Install dependencies:**
    ```
    pip install -r Cardless_E-ATM/requirements.txt
    ```

4. **Download Dlib shape predictor:**  
   Place `shape_predictor_68_face_landmarks.dat` inside the project directory.

---

## Usage

1. **Train the face recognition model:**
    ```
    python Cardless_E-ATM/trainer.py
    ```
2. **Run the application:**
    ```
    python Cardless_E-ATM/app.py
    ```
3. **Follow the on-screen instructions:**  
   Authenticate with your face and use the eye-controlled PIN entry.

---

## Tools & Libraries

- Python 3.9+
- OpenCV
- Dlib
- face_recognition
- Numpy
- Pyglet
- Scikit-learn
- Flask

---

## Notes

- Ensure your camera is functional before running the app.
- For best results, keep your face visible and well-lit.
- To register new users, add corresponding images to the `Dataset/` folder and re-train the model.

---

## License

This project is open-source and intended for educational purposes.

