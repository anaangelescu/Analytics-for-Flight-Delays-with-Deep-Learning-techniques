# Flight Delay Prediction

**Module Overview**

Applied machine learning and deep learning techniques (LSTM and traditional classifiers) to forecast flight delays using historical flight data, weather conditions, and airport congestion factors. The goal was to improve on-time performance predictions and enable proactive operational adjustments.

## 🚀 Tech Stack

* **Language:** Python 3.9+
* **Libraries:** TensorFlow/Keras, scikit-learn, Pandas, NumPy
* **Environment:** Google Colab (`Module7_LSTM.ipynb`)
* **Reporting:** Microsoft PowerPoint (`Case_study_Module7.pdf`)


## 🔧 Getting Started

1. **Clone this repository**

   ```bash
   git clone https://github.com/anaangelescu/airline-analytics-portfolio.git
   cd airline-analytics-portfolio/Module7_Flight_Delays
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Open and run the notebook**

   * Open `Module7_LSTM.ipynb` in Google Colab:
     [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anaangelescu/airline-analytics-portfolio/blob/main/Module7_Flight_Delays/Module7_LSTM.ipynb)

## 🔍 Methodology Highlights

1. **Data Preparation:** Merged flight performance records with weather and airport congestion data, cleaned missing values, and created features such as departure delay, distance, temperature, and visibility.
2. **Modeling:**

   * **Traditional ML:** Trained classifiers (Logistic Regression, Random Forest) with cross-validation.
   * **Deep Learning:** Built LSTM networks to capture temporal patterns in delay history.
3. **Evaluation:** Compared models using accuracy, precision, recall, and loss metrics on balanced and unbalanced datasets.

## 📈 Key Results

| Model                 | Dataset  | Test Accuracy | Test Loss |
| --------------------- | -------- | ------------- | --------- |
| LSTM (unbalanced)     | Full set | 81.41%        | 0.4477    |
| LSTM (balanced 50/50) | Balanced | 64.71%        | 0.6246    |
| Random Forest         | Full set | 78.33%        | N/A       |
| Logistic Regression   | Full set | 75.12%        | N/A       |

* **Best Performance:** LSTM on unbalanced data achieved the highest accuracy.
* **Balanced Trade-off:** Balancing improved recall for delayed flights at the cost of overall accuracy.

---

*For detailed code, diagrams, and results, see `Case_study_Module7.pdf`.*

