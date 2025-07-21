# Big Data Analytics for Flight Optimization

**Module Overview**

Explored and analyzed large-scale airline operations datasets to optimize flight planning, reduce delays, and improve overall operational efficiency. Leveraged Big Data tools and frameworks to build scalable pipelines, perform feature engineering, and deliver actionable insights.

## 🚀 Tech Stack

* **Languages & Environments:** R (RStudio), Python
* **Big Data Frameworks:** Hadoop (HDFS), Apache Spark
* **Data Processing:** Pandas, data.table
* **Visualization:** ggplot2, Matplotlib
* **Reporting:** Microsoft PowerPoint (15‑minute slide deck)

## 📂 Repository Contents

* `Module_2_Big_data_analytics.R` – R script executing the core data processing and analysis pipeline *(if available)*

## 🔧 Getting Started

1. **Clone this repository**

   ```bash
   git clone https://github.com/anaangelescu/airline-analytics-portfolio.git
   cd airline-analytics-portfolio/Module2_Big_Data_Analytics
   ```
2. **Install dependencies**

   * **R packages:**

     ```r
     install.packages(c("data.table", "dplyr", "ggplot2", "sparklyr"))
     sparklyr::spark_install(version = "3.0.0")
     ```
   * **Python (optional):**

     ```bash
     pip install pandas matplotlib pyspark
     ```
3. **Run the analysis script**

   ```bash
   Rscript Module_2_Big_data_analytics.R
   ```
4. **Review the slide deck**

   * Open `Case_study_Module2.pdf` in PowerPoint or any PDF viewer to see detailed findings and recommendations.

## 🔍 Methodology Highlights

* **Data Ingestion:** Loaded raw CSVs (flight performance, passenger activity, aircraft inventory, airport coordinates, carrier codes, employee stats, weather) into HDFS and Spark DataFrames.
* **Data Cleaning:** Handled missing values, standardized date formats, deduplicated records.
* **Dataset Merging:** Joined multiple sources to enrich flight records with carrier, capacity, employee, and weather features.
* **Feature Engineering:** Created metrics for airport busyness, aircraft utilization, weather impact, and resource allocation ratios.
* **Scalable Processing:** Demonstrated distributed analytics on multi‑million‑row datasets using Spark.

## 📈 Key Outcomes

* **Route Optimization Insights:** Identified underperforming routes and recommended schedule adjustments to reduce average delays by an estimated 8–12%.
* **Scalable Architecture:** Developed a reusable pipeline supporting ad hoc and batch analytics on >5 million records.
* **Presentation:** Delivered a 15‑minute pitch highlighting efficiency gains, cost‑saving opportunities, and a path to production deployment.


