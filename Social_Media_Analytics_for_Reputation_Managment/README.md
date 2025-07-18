# Descriptive Analytics & KPI Visualization

**Module Overview**

Analyzed airline route and operational data to create interactive dashboards and visualizations of key performance indicators (KPIs) such as on‑time performance, load factors, and network connectivity. Delivered end‑user reporting tools to enable business stakeholders to explore and interpret operational metrics.

## 🚀 Tech Stack

* **Visualization:** Tableau Desktop & Tableau Public
* **Languages:** Python (Pandas, Matplotlib, Seaborn)
* **Data Processing:** Pandas for aggregation and KPI calculations
* **Reporting:** Microsoft PowerPoint (slide deck)

## 📂 Repository Contents

* `Case_study_Module5.pdf` – Detailed methodology, dashboard screenshots, code snippets, and findings
* `dashboards/` – Tableau workbook files (`.twb` / `.twbx`) for interactive KPI dashboards
* `scripts/` – Python scripts for data cleaning, aggregation, and export to Tableau-compatible extracts

## 🔧 Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/anaangelescu/airline-analytics-portfolio.git
   cd airline-analytics-portfolio/Module5_Descriptive_Analytics
   ```
2. **Install Python dependencies**

   ```bash
   pip install pandas matplotlib seaborn tableau-api-lib
   ```
3. **Generate Tableau data extracts**

   ```bash
   python scripts/generate_extracts.py
   ```
4. **Open dashboards**

   * Launch Tableau Desktop or Tableau Public and open the workbook file in `dashboards/`
   * Interact with filters for time period, airline carrier, and route segments

## 📊 Methodology & Highlights

* **Data Aggregation:** Calculated KPIs including:

  * **On‑Time Performance:** % of flights departing/arriving within 15 minutes of schedule
  * **Load Factor:** Ratio of passengers to available seats
  * **Connectivity Index:** Count of unique destinations per origin airport
* **Dashboard Features:**

  * Time‑series trends with parameterized date ranges
  * Geographic maps showing route networks colored by KPI values
  * Drill‑down filters for airline, airport, and aircraft type
* **Automation:** Python scripts automate data prep and publish extracts to Tableau Public via API

## 📈 Key Results

* Delivered a self‑service Tableau dashboard enabling non‑technical stakeholders to explore KPIs across 200+ routes.
* Identified top 10 underperforming routes, leading to targeted operational reviews.
* Presented findings in a slide deck leading to a 5% improvement in overall on‑time performance after stakeholder action.

---

*For full dashboard designs, code details, and visual examples, see `Case_study_Module5.pdf` and the Tableau workbooks in `dashboards/`.*

