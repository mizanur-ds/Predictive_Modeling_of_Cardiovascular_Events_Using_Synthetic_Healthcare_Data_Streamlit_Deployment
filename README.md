## 📄 Abstract
This project predicts the risk of heart attack or stroke using a synthetic UK primary care dataset. After data analysis, preprocessing, and testing multiple machine learning models, **Logistic Regression** was chosen for its accuracy and interpretability. The model is deployed as a **Streamlit web app**, enabling users to enter health data and instantly receive a personalized risk probability — all while maintaining privacy through synthetic data.

---

## 🌐 Live App
You can try the interactive prediction tool here:  
**[Heart Attack and Stroke Risk Prediction App](https://heartattackandstrokeriskprediction.streamlit.app/)**

Enter your health details step-by-step and instantly see your estimated risk probability.

---

## 📊 Dataset
- **Source**: NIHR ARC Wessex – Synthetic dataset for cardiovascular event prediction  
  🔗 [ARC Wessex Datasets](https://www.arc-wx.nihr.ac.uk/data-sets?utm_source=chatgpt.com)
- **Description**: Mimics real UK primary care patient records, incorporating realistic complexities such as missingness, noise, and feature interactions.
- **Files Used**:  
  - `cvd_synthetic_dataset_v0.2.csv` → Raw dataset used in `1_data_analysis_preprocessing.ipynb`  
  - `final_df.csv` → Processed dataset used in modeling and Streamlit app

---

## 📂 Repository Structure
```plaintext
📦 Heart-Stroke-Prediction
│
├── 1_data_analysis_preprocessing.ipynb   # Data cleaning, EDA, feature engineering
├── 2_data_modeling_prediction.ipynb      # Model training, evaluation
├── Attack_or_Stroke_prediction.py        # Streamlit app
├── cvd_synthetic_dataset_v0.2.csv        # Raw dataset
├── final_df.csv                          # Processed dataset ready for modeling
├── requirements.txt                      # Dependencies
└── README.md                             # Project documentation
