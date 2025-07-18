# Module 3: Data Warehousing & ETL

**Project Overview**

Built a scalable data warehouse to aggregate and analyze airline booking and operations data from 2009–2012. Integrated on‑time performance records with weather and airport metadata to enable fast, reliable querying and reporting.

## 🚀 Tech Stack

* **Database:** PostgreSQL (or your preferred SQL engine)
* **ETL:** Python (Pandas), Shell scripts
* **Modeling:** Star schema design with fact and dimension tables
* **Tools:** Microsoft PowerPoint (design diagrams and slide deck)

## 📂 Repository Contents

* `Case_study_Module3.pptx` – Detailed report, data model diagrams, ETL workflow, and results

## 🔧 Getting Started

1. **Clone this repo**

   ```bash
   git clone https://github.com/anaangelescu/airline-analytics-portfolio.git
   cd airline-analytics-portfolio/Module3_Data_Warehousing
   ```
2. **Open the slide deck**

   * Launch `Case_study_Module3.pptx` in PowerPoint to view the full methodology and results.

## 🗄️ Data Warehouse Design Highlights

* **Fact table:** `fact_flights` capturing flight-level metrics (delays, cancellations, distances)
* **Dimensions:**

  * `dim_time` (date, month, quarter, year)
  * `dim_aircraft` (tail number, model, manufacturer)
  * `dim_airport` (airport code, city, state)
  * `dim_weather` (temperature, precipitation, visibility)
* **ETL Workflow:**

  1. Extract raw CSV datasets (`ONTIME_REPORTING_*.csv`)
  2. Clean and standardize fields (dates, numeric conversions)
  3. Load into staging tables
  4. Transform and insert into final star schema tables

## 📈 Key Results

* Processed **over 10 million** flight records across four years
* Achieved **60% reduction** in average query time through indexing and schema optimization
* Demonstrated a modular ETL pipeline supporting incremental loads and easy schema evolution

---

*For full diagrams, code snippets, and detailed analysis, please refer to the slide deck (`Case_study_Module3.pptx`).*
