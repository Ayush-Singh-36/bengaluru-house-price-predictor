# 🏡 Bengaluru House Price Predictor

A streamlined, end-to-end machine learning web application that estimates residential real estate prices in Bengaluru, India based on key property characteristics. 

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bengaluru-house-price-predictor-pq8vmj45uf8pbty4mj9gw5.streamlit.app/)

👉 **Try the Live App:** [Bengaluru House Price Predictor](https://bengaluru-house-price-predictor-pq8vmj45uf8pbty4mj9gw5.streamlit.app/)

---

## 🛠️ System Architecture

The application is deployed directly on **Streamlit Cloud** using an in-memory prediction pipeline for fast, low-latency inferencing:

* **Frontend UI & Processing (Streamlit):** Interactive user interface built with Streamlit to accept property parameters (location, square footage, BHK size, bathrooms, balconies, and area type).
* **Inference Pipeline (Scikit-Learn):** Loads pre-trained model artifacts (`bengaluru_house_production_bundle.pkl`) directly into memory. Preprocesses input features with categorical One-Hot Encoding and numerical scaling, then performs real-time valuation inference.
* **Smart Formatting:** Dynamically calculates estimated valuations and formats results intuitively in **Lakhs** or **Crores**.

---

## 📦 Repository Structure

```text
bengaluru-house-price-predictor/
├── app.py                             # Main Streamlit web application & prediction logic
├── bengaluru_house_production_bundle.pkl  # Trained ML model, OneHotEncoder, and scaler artifacts
├── requirements.txt                   # Python dependencies for deployment
└── README.md                          # Project documentation
💻 Running Locally

To run this app on your local machine:
Clone the repository:
git clone [https://github.com/Ayush-Singh-36/bengaluru-house-price-predictor.git](https://github.com/Ayush-Singh-36/bengaluru-house-price-predictor.git)
cd bengaluru-house-price-predictor

Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt
Launch the Streamlit app:
streamlit run app.py

🛠️ Tech Stack
Language: Python
Machine Learning: Scikit-Learn, Pandas, NumPy
Web Framework: Streamlit
Deployment & CI/CD: Streamlit Cloud, GitHub


---

### Step-by-Step Instructions:

1. On your current GitHub tab ([Editing README.md](https://github.com/Ayush-Singh-36/bengaluru-house-price-predictor/edit/main/README.md)), select all existing text and delete it.
2. Paste the markdown block above into the editor.
3. Click the green **Commit changes...** button at the top right.
