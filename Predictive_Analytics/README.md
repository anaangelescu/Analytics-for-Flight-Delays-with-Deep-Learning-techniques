# Expert Systems for Flight Operations

**Module Overview**

Developed a rule‑based expert system to support flight operations decision‑making, automating routine checks and providing recommendations for delay mitigation and resource allocation. The system integrates domain rules, flight schedules, and real‑time status data.

## 🚀 Tech Stack

* **Rule Engine:** Prolog (SWI‑Prolog)
* **Integration:** Python wrappers for data preprocessing and result visualization
* **Knowledge Authoring:** Custom rule files (`.pl`) defining flight, crew, and airport operational logic
* **Reporting:** Microsoft PowerPoint (slide deck)

## 📂 Repository Contents

* `Case_study_Module4.pdf` – Detailed system architecture, rule definitions, inference examples, and evaluation results
* `rules/` – Prolog source files defining the expert system knowledge base
* `scripts/` – Python scripts for data loading, invoking the Prolog engine, and visualizing outputs

## 🔧 Getting Started

1. **Clone this repo**

   ```bash
   git clone https://github.com/anaangelescu/airline-analytics-portfolio.git
   cd airline-analytics-portfolio/Module4_Flight_Operations
   ```
2. **Install dependencies**

   * **Prolog:** Install SWI‑Prolog (version 8.x+)
   * **Python:**

     ```bash
     pip install pandas matplotlib pyswip
     ```
3. **Run the expert system**

   ```bash
   # Launch Prolog and load rules
   swipl -s rules/flight_rules.pl
   # Within the Prolog prompt, query recommendations, for example:
   ?- find_delay_mitigation(FlightID, Recommendation).
   ```
4. **Visualize results**

   ```bash
   python scripts/visualize_recommendations.py
   ```

## 🛠️ System Architecture & Methodology

* **Knowledge Base:** Defined facts and rules for flight delays, crew availability, and airport capacity
* **Inference Engine:** Used backward‑chaining in Prolog to derive actionable insights (e.g., reassign crew, adjust gate assignments)
* **Data Flow:** Python preprocesses CSV flight logs into Prolog facts, then post‑processes engine output into charts
* **Rule Examples:**

  * `delay_threshold(Flight, Delay) :- flight_status(Flight, Scheduled, Actual), Delay is Actual - Scheduled, Delay > 15.`
  * `recommend_reassignment(Flight, Crew) :- delay_threshold(Flight, D), D > 30, available_crew(Crew, Flight).`

## 📈 Key Results

* **Decision Accuracy:** Expert system recommendations matched 92% of manual dispatch decisions in historical validation.
* **Runtime Performance:** Average inference time < 150 ms per query, enabling near‑real‑time support.
* **Operational Impact:** Projected to reduce average delay resolution time by 20% when integrated into dispatch workflows.

---

*For full rule listings, evaluation details, and system diagrams, see `Case_study_Module4.pdf`.*

