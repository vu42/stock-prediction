## 1) System context diagram (Section 2 – Overall system description)

[Inference]

* **Purpose**: Show the system as a single box and how it interacts with external actors and services.
* **Content**:

  * Actors: End User, Data Scientist.
  * External systems: Browser, AWS infrastructure, Airflow, VN30 market data provider, Object storage (S3), Postgres.
  * Your “Stock Prediction System” in the center, with arrows for data and requests.
* **Why useful**: Matches the “big picture” that lecturers like in SRS documents; helps show where ML and pipelines live inside the larger environment.

---

## 2) High level architecture / component diagram (Section 4 – Data model and system integration)

[Inference]

* **Purpose**: Show the internal blocks of your system.
* **Content**:

  * Frontend web app.
  * Backend API (with sub components like Auth, Home service, Stock service, Model service, Pipeline proxy).
  * Training worker.
  * Airflow.
  * Postgres and object storage.
* **Links**:

  * Browser → Frontend → Backend.
  * Backend ↔ Postgres, Backend ↔ Airflow, Backend ↔ Object storage.
  * Backend → training queue → Training worker → Postgres + object storage.
* **Why useful**: Directly supports Section 4 where you describe components and data flow from raw market data to predictions.

---

## 3) Logical data model / ER diagram (Section 4 – Data model)

[Inference]

* **Purpose**: Visualize tables and their relationships.
* **Key entities**:

  * `users`, `stocks`, `stock_prices`.
  * `stock_prediction_summaries`, `stock_prediction_points`.
  * `model_statuses`, `model_horizon_metrics`.
  * `training_configs`, `experiment_runs`, `experiment_logs`, `experiment_ticker_artifacts`.
  * `pipeline_dags`, `pipeline_runs`, `pipeline_run_tasks`, `pipeline_run_logs`.
* **Relationships**:

  * `stocks` 1–n `stock_prices`; `stocks` 1–n `stock_prediction_*`; `stocks` 1–1 or 1–n `model_statuses`.
  * `training_configs` 1–n `experiment_runs`.
  * `experiment_runs` 1–n `experiment_logs`, 1–n `experiment_ticker_artifacts`.
  * `pipeline_dags` 1–n `pipeline_runs` 1–n `pipeline_run_tasks` 1–n `pipeline_run_logs`.
* **Why useful**: Fits perfectly under “Data model” subsection and helps the reader see how ML, predictions, and pipelines are stored.

---

## 4) Use case diagram (Section 2.2 and Section 3 – Use cases)

[Inference]
Even though you already have textual use case tables, a small use case diagram is a nice summary.

* **Actors**: End User, Data Scientist.
* **Use cases** (bubbles):

  * Login.
  * View Home.
  * View Stock Detail.
  * View Models.
  * Configure Training.
  * Run Experiment.
  * Monitor Pipelines.
* **Why useful**: Shows quickly which actor can do what, and complements your detailed tables in Section 3.

---

## 5) Sequence diagram for a key flow (Section 3 – Use case scenarios)

[Inference]
Choose one or two important flows and show a UML sequence diagram.

Good candidates:

1. **“View Stock Detail” sequence**

   * Lifelines: User, Browser, Frontend, Backend API, Postgres.
   * Messages: click row on Home → frontend routing → API call to `/stock/{ticker}` → DB query → response → render chart and predictions.

2. **“Run training experiment” sequence**

   * Lifelines: Data Scientist, Browser, Frontend, Backend API, Queue, Training worker, Postgres, Object storage.
   * Messages: submit config → API saves `training_configs` and creates `experiment_runs` → enqueue job → worker pulls job → trains models → writes metrics to DB and artifacts to S3 → status polled by frontend.

* **Why useful**: Shows how components interact step by step, nicely tying together APIs, DB, and ML parts.

---

## 6) Data flow diagram for ML pipeline (Section 6 – Machine learning models)

[Inference]
This diagram focuses on the data transformation path.

* **Boxes / steps**:

  1. VN30 raw prices (for your 8 tickers).
  2. Feature engineering (lookback window, indicators).
  3. Train / validation split by time.
  4. Model training per horizon (ridge, SVR, random forest, gradient boosting).
  5. Ensemble combination.
  6. Evaluation and MAPE computation.
  7. Write predictions and metrics to DB; write plots to object storage.
  8. UI reads summaries for Home, Stock Detail, Models.

* **Why useful**: Makes Section 6 more concrete and clearly connects math to system implementation.

---

## 7) Small “horizon and target” diagram (Section 6.1 – Problem formulation)

[Inference]
This is a simple time axis illustration, but lecturers usually like it.

* **Content**:

  * Time axis t, t+1, …, t+7, t+15, t+30.
  * At day t: feature vector (\mathbf{x}_t) from lookback window.
  * Show arrows from t to t+7, t+15, t+30 labelled (y_{t+7}), (y_{t+15}), (y_{t+30}) (percentage changes).
* **Why useful**: Visually clarifies how you define the target and horizons.

