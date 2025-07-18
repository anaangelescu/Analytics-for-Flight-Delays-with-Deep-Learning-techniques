# Predictive Analytics for Flight Delays

**Module 7: Prescriptive Analytics - Automated Decision Systems, Expert Systems, Knowledge Management, and Collaborative Systems**

A case study applying deep learning (LSTM) and traditional machine learning techniques to forecast flight delays using historical flight data, weather conditions, and airport congestion factors.

## 🚀 Tech Stack

* **Language:** Python 3.9+
* **Frameworks & Libraries:** TensorFlow/Keras, scikit-learn, Pandas, NumPy
* **Environment:** Google Colab
* **Reporting:** Microsoft PowerPoint (slides in PDF)

## 📂 Repository Contents

* `Case_study_Module7.pdf` – Project report and slides
* `Module7_LSTM.ipynb` – Google Colab notebook implementing LSTM models
* `requirements.txt` – List of Python dependencies

## 🔧 Getting Started

```bash
git clone https://github.com/anaangelescu/predictive-flight-delays.git
cd predictive-flight-delays
pip install -r requirements.txt
```


## 📊 Key Results

| Model                 | Dataset  | Test Accuracy | Test Loss |
| --------------------- | -------- | ------------- | --------- |
| LSTM (unbalanced)     | Full set | 81.41%        | 0.4477    |
| LSTM (balanced 50/50) | Balanced | 64.71%        | 0.6246    |

## 📝 Conclusions & Future Work

* **Performance:** The LSTM model achieved higher overall accuracy on the unbalanced dataset, but balancing improved recall for delayed flights at the cost of overall accuracy.
* **Next Steps:** Hyperparameter tuning, advanced feature engineering, and incorporating additional external factors (e.g., real-time weather APIs).

---

*Prepared as part of Module 7 coursework on Prescriptive Analytics.*
