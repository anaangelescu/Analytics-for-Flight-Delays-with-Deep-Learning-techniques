# Predictive Maintenance for Aircraft Components

**Module Overview**

Developed a machine learning pipeline to predict failures of critical aircraft components using historical maintenance logs, sensor readings, and operational metrics. The goal was to schedule proactive maintenance, reduce unscheduled downtime, and optimize fleet availability.

## 🚀 Tech Stack

* **Language:** Python (3.8+)
* **Libraries:** Pandas, NumPy, scikit-learn, XGBoost
* **Data Storage:** CSV files (historical logs), SQLite for prototyping
* **Reporting:** Microsoft PowerPoint (`Case_study_Module6.pdf`)

## 📂 Repository Contents

* `Case_study_Module6.pdf` – Full case study report, methodology, and results slides
* `data/` – Sample CSV datasets (`maintenance_logs.csv`, `sensor_readings.csv`)
* `notebooks/` – Jupyter notebooks with data exploration and modeling (`Predictive_Maintenance.ipynb`)
* `scripts/` – Python scripts for training models and generating evaluation reports
* `requirements.txt` – List of Python dependencies

## 🔧 Getting Started

1. **Clone this repository**

   ```bash
   git clone https://github.com/anaangelescu/airline-analytics-portfolio.git
   cd airline-analytics-portfolio/Module6_Predictive_Maintenance
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Explore data and run notebooks**

   * Launch Jupyter Lab or Notebook:

     ```bash
     jupyter lab notebooks/Predictive_Maintenance.ipynb
     ```

4. **Train models via script**

   ```bash
   python scripts/train_model.py --data-dir data/ --output models/
   ```

5. **Generate evaluation report**

   ```bash
   python scripts/evaluate_model.py --models models/ --report reports/evaluation_results.csv
   ```

## 🔍 Methodology & Highlights

1. **Data Preprocessing**: Cleaned and merged maintenance logs with time-series sensor data, handled missing values, and created rolling-window features for sensor trends.
2. **Feature Engineering**: Generated features such as time since last maintenance, moving averages of vibration and temperature readings, and usage metrics.
3. **Modeling**: Compared classification algorithms (Random Forest, XGBoost, Logistic Regression) using cross-validation and grid search for hyperparameter tuning.
4. **Evaluation**: Assessed models using precision, recall, F1-score, and ROC-AUC; selected XGBoost for best trade-off between performance and interpretability.

## 📈 Key Results

* **Best Model (XGBoost)**: ROC-AUC = 0.89, Precision = 0.85, Recall = 0.78
* **Lead Time**: Predicted failures up to 7 days in advance, enabling proactive maintenance scheduling.
* **Operational Impact**: Estimated 25% reduction in unscheduled downtime and savings of \$250K/year in maintenance costs.

---

*For full methodology, code snippets, and detailed results, see `Case_study_Module6.pdf`.*

