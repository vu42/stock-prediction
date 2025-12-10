# 1. Introduction

## 1.1 Purpose

The purpose of this document is to specify the functional and non functional requirements of a stock prediction and analytics system that predicts future percentage price changes for VN30 stocks, supports model training and evaluation, and exposes results through a web based user interface and REST API. 

The document is intended for:

* The development team implementing the frontend, backend, data pipelines, and machine learning models. 
* Teaching staff evaluating the Intelligent Systems assignment, including correctness of requirements coverage and alignment between UI, backend, and ML components. 
* Future maintainers who need a concise but complete description of system behavior, data model, and integration points.

## 1.2 Scope

The system is a web based stock prediction application focused on VN30 index constituents. Its main goals are to:

* Provide end users with a Home page that surfaces top opportunities to buy or sell, along with a market table that combines actual and predicted horizon based percentage changes. 
* Allow end users to open a Stock Detail page for a specific ticker, view recent price history, predicted future percentage changes for multiple horizons, and current model status. 
* Provide data scientists with screens for configuring training features, running experiments, monitoring data and training pipelines in Airflow, and inspecting model level metrics and artifacts.
* Provide a Models page that summarizes per ticker metrics (MAPE for 7D, 15D, 30D), latest predictions, and evaluation plots, using a minimal table oriented design.

The system includes:

* A frontend single page application implementing all screens described in this document. 
* A backend API service that implements authentication, authorization, and all business logic and integration with Postgres, Airflow, cache, and object storage. 
* A training worker that executes model training experiments and writes metrics and artifacts. 
* Airflow DAGs and pipeline metadata tables for VN30 data crawling and model training.
* A PostgreSQL database that stores users, roles, stocks, experiments, predictions, model status, and pipeline metadata.

Out of scope are advanced features such as historical model comparison, retrain buttons from the UI, advanced filtering on the Models page, and full production grade risk management. 

This section reflects your project scope decision to work on a subset of VN30 tickers due to resource limits on AWS. I cannot verify these constraints independently.

Here is a revised version of **1.3 Definitions, acronyms, and abbreviations** that you can drop into the report.

---

### 1.3 Definitions, acronyms, and abbreviations

* **VN30**: The VN30 stock index consisting of 30 large cap Vietnamese stocks. In this project, due to compute and cost limitations for model training on AWS, the system only supports a fixed subset of VN30 tickers defined by `VN30_STOCKS`.

* **VN30_STOCKS**: The subset of VN30 tickers that the system actually processes and predicts for:
  `["FPT", "VCB", "VNM", "HPG", "VIC", "VHM", "MSN", "SAB", "TCB", "GAS"]`.

* **Ticker**: Stock symbol of a company, for example FPT, VCB, HPG. In this system, tickers are restricted to the `VN30_STOCKS` subset described above.

* **End User**: A user who views predictions and market information for investment oriented interpretation.

* **Data Scientist**: A user who configures features, runs training experiments, monitors pipelines, and inspects model performance.

* **Horizon**: Prediction offset in days, for example 7, 15, or 30 days from a reference time $t$.

* **Feature vector $\mathbf{x}_t$:** A vector containing technical indicators and historical price information at time $t$.

* **Target $y_{t+h}$:** Percentage change in closing price from day $t$ to day $t+h$ for horizon $h$.

* **Experiment**: A training run that uses a specific configuration (index, indicators, horizons, models, ensemble strategy) and produces metrics and artifacts.

* **Pipeline DAG**: A directed acyclic graph in Airflow that orchestrates data crawling or model training jobs.

* **MAPE**: Mean Absolute Percentage Error, the main evaluation metric used for each prediction horizon.

* **Model status**: Aggregated status per ticker and horizon summarizing freshness and accuracy of the latest trained model.

---

# 2. Overall system description

## 2.1 User roles and permissions

The system distinguishes users by role to support different navigation and permissions.

* End User

  * Default landing page: Home.
  * Available navigation: Home, Stock Detail. 
  * Can view market wide summaries, top picks, their personal list, and detailed predictions for specific stocks.

* Data Scientist

  * Default landing page: Training.
  * Available navigation: Training, Pipelines, Models, Home, Stock Detail. 
  * Can configure feature sets and training options, start experiments, monitor experiment runs, monitor VN30 data and training pipelines, and inspect per ticker model metrics and artifacts.

Authentication and role based navigation are enforced by the backend using the `users` table and role fields, and the frontend adjusts the visible navigation items according to the authenticated role.

## 2.2 High level use case description

The main use cases of the system can be grouped into three logical clusters, following the style of the example requirements document but adapted to stock prediction:

1. End user insights and navigation

   * Log in as end user and be redirected to the Home page. 
   * View top picks grouped into Should Buy, Should Sell, and My List, and open Stock Detail for any row.
   * On Stock Detail, inspect current and historical prices, actual and predicted percentage changes for multiple horizons, and a summary of model status.

2. Training, experiments, and pipelines for data scientists

   * Log in as data scientist and be redirected to the Training screen. 
   * Configure input features, prediction horizons, models, and ensemble options for a training configuration. 
   * Start a new experiment run and monitor its status, logs, and artifacts.
   * Monitor VN30 data crawling and model training pipelines, including DAG list, run history, status, and basic controls such as trigger and stop.

3. Model monitoring and evaluation

   * View a Models overview table summarizing per ticker metrics (MAPE for 7D, 15D, 30D), current predictions, and last trained timestamp.
   * Open a plot modal for a specific ticker to view evaluation plots stored in object storage, including actual versus predicted curves and overlays of metrics.

## 2.3 Functional requirements

This subsection summarizes functional requirements grouped by logical clusters that correspond to major screens and flows. Detailed use case scenarios are presented in Section 3.

### 2.3.1 Authentication and session management

* The system shall provide a login form with username and password fields, a show or hide password option, and a Sign In button. 
* The system shall display example test accounts for each role on the login page to support demonstration. 
* On successful login, the system shall redirect end users to Home and data scientists to Training, and render the role appropriate navigation menu. 
* On invalid credentials, the system shall display an error state and keep the user on the login screen. 

### 2.3.2 End User Home

* The system shall display a Top Picks area with three tabs: Should Buy, Should Sell, and My List. 
* For each tab, the system shall show a list of up to five VN30 stocks including ticker, name, current price, and predicted percentage change for the primary horizon, highlighting buy or sell recommendations.
* The system shall display a market table that, for each VN30 stock, shows name, symbol, current price, actual percentage change for 7D, 15D, and 30D, predicted percentage change for 7D, and a short historical sparkline. 
* The system shall allow the user to click any row in the Top Picks or market table to navigate to the Stock Detail page for that ticker.
* The Home page shall load its data via backend endpoints that return top picks, market table, and user watchlist entries as JSON.

### 2.3.3 Stock Detail

* The system shall provide a Stock Detail page per ticker that can be accessed from Home or directly via URL.
* The page shall display basic information including ticker, company name, current price, and recent percentage change. 
* The page shall display a price chart over a configurable window (for example 60 or 90 trading days), using daily prices from the database. 
* The page shall display predicted percentage changes for horizons 7D, 15D, and 30D, including values and visual indication such as up or down arrows and color coding consistent with the Models page.
* The page shall display a summary of model status for the stock, including last trained time, MAPE metrics, and qualitative labels such as excellent, acceptable, or needs improvement based on MAPE thresholds.

### 2.3.4 Training and Experiments

* The system shall provide a Training screen for data scientists that allows them to configure features and training options. 
* The configuration shall include index selection (VN30), ticker filters, lookback window size, technical indicators (such as SMA, EMA, RSI, MACD, Bollinger Bands, ATR, volume moving averages), prediction horizons, models to train, and ensemble strategies.
* The system shall validate feature configurations for consistency and save them in `training_configs`.
* The system shall allow data scientists to start experiments that create `experiment_runs` records, enqueue background training jobs, and show a detail view per run including status, metrics, logs, and links to artifacts.

### 2.3.5 Pipelines and Monitoring

* The system shall provide a Pipelines screen that lists all registered Airflow DAGs related to VN30 data and training, such as `vn30_data_crawler` and `vn30_model_training`.
* For each DAG, the system shall allow the user to view recent runs, their states, start and end times, and durations, based on mirrored metadata in `pipeline_dags` and `pipeline_runs`.
* The system shall allow data scientists to trigger DAG runs with optional configuration payloads, pause or unpause DAGs, and request stopping running DAG instances.
* The system shall provide basic introspection into tasks of a specific run based on `pipeline_run_tasks` and shall display pipeline logs based on `pipeline_run_logs`.

### 2.3.6 Models and Evaluation

* The system shall provide a Models page that displays a table with ticker, last trained time, MAPE for 7D, 15D, and 30D, predicted percentage changes for the same horizons, and an action column for viewing evaluation plots.
* The system shall color code MAPE values based on thresholds: green when below 5 percent, yellow when between 5 and 10 percent, and red when above 10 percent. 
* The system shall color code predictions based on sign, using green for positive predicted change and red for negative predicted change, and shall display direction indicators.
* When the user selects the view action for a ticker, the system shall open a modal that displays the evaluation plot image referenced by a URL stored in the database and pointing to object storage.

## 2.4 Non functional requirements

The non functional requirements adapt the structure of the example document to the stock prediction context.

 The following requirements extend the given specs to make the system more robust in a realistic deployment.

### 2.4.1 Performance



* The system should support at least a moderate number of concurrent users (for example tens of users) viewing Home and Stock Detail without noticeable degradation.
* For typical Home and Stock Detail requests served from cache or Postgres, the backend should respond within a few seconds in normal conditions, including computation of derived fields such as percentage changes and color coding.
* Training experiments and pipeline runs are allowed to take minutes to hours, but the API must respond quickly with job identifiers and later serve status updates without blocking.

### 2.4.2 Scalability



* The system should be deployable in a way that allows scaling the API and training workers horizontally as load increases, using containerization and queue based background processing.
* Cache and object storage should be used to reduce load on Postgres for read heavy endpoints such as Home, Stock Detail, and Models.

### 2.4.3 Reliability and availability



* The system should be designed to minimize downtime, targeting a high availability percentage in practice, by separating frontend, backend, worker, and Airflow components and using health checks.
* All critical operations (login, viewing predictions, model status, pipeline monitoring) should log successes and failures for later diagnosis.

### 2.4.4 Security



* All HTTP communication between frontend and backend should use TLS.
* Passwords should be stored using secure hashing in the `users` table and authenticated sessions should use secure tokens.
* Role based access control should ensure that only data scientists can access Training, Pipelines, and Models configuration endpoints, while end users can only access view oriented endpoints.

### 2.4.5 Usability



* The UI should keep the design simple and clean, using consistent colors for MAPE and predictions and an easy to read table layout on the Models page.
* The Home and Models pages should remain usable on smaller screens by supporting horizontal scrolling of tables and responsive modals. 

### 2.4.6 Maintainability



* The backend codebase should follow the recommended monorepo structure, separating API, worker, integrations, models, and migrations to ease maintenance and evolution.
* Domain logic should be encapsulated in service layers and shared models to keep Airflow DAGs and other integration points thin and maintainable.

### 2.4.7 Backup and recovery



* The PostgreSQL database should be backed up regularly to protect user accounts, experiments, predictions, and pipeline metadata.
* Object storage for artifacts should retain evaluation plots and model files for at least the duration of the assignment, with a documented procedure for restoring artifacts if needed.

---

# 3. Use case scenarios

Use cases are presented in table form following the style of the example document, but adapted to the stock prediction system.

## 3.1 UC1 – Login

| Field          | Description                                                                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| ID             | UC1                                                                                                                                                   |
| Name           | User login                                                                                                                                            |
| Description    | User authenticates and is redirected to the role appropriate landing page.                                                                            |
| Actors         | End User, Data Scientist                                                                                                                              |
| Preconditions  | The user account exists in the `users` table with a valid role and password hash.                                                                     |
| Postconditions | A valid session or token is created and stored; the user is redirected to Home (end user) or Training (data scientist) with correct navigation items. |

**Main flow**

1. User opens the application and navigates to the login page. 
2. User enters username and password. 
3. User clicks the Sign In button. 
4. The backend validates credentials and role, creates a session or token, and returns success.
5. The frontend redirects the user to Home if the role is end user or to Training if the role is data scientist and renders the corresponding navigation. 

**Alternative flows**

* A1: Test account selection

  1. User clicks a provided test account credential pair; the frontend auto fills the login form and proceeds with steps 3 to 5. 

**Exception flows**

* E1: Invalid credentials

  1. Backend returns an error for invalid username or password.
  2. Frontend displays an error message and leaves the user on the login page.

## 3.2 UC2 – View Home page

| Field          | Description                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| ID             | UC2                                                                                                          |
| Name           | View Home with top picks and market table                                                                    |
| Description    | End User views Should Buy, Should Sell, and My List, and the market table, and may navigate to Stock Detail. |
| Actors         | End User                                                                                                     |
| Preconditions  | User is authenticated as end user and has access to the Home page.                                           |
| Postconditions | Top picks and market table are displayed; any selected ticker is ready to navigate to Stock Detail.          |

**Main flow**

1. User logs in as end user and is redirected to Home. 
2. Frontend calls the backend endpoints to fetch top picks, market table, and user list data. 
3. System displays three tabs: Should Buy, Should Sell, and My List. 
4. For the active tab, system displays up to five rows showing ticker, name, current price, and main predicted horizon percentage change.
5. System renders the market table with columns including actual and predicted percentage changes for multiple horizons, using appropriate formatting. 
6. User clicks a stock row to open Stock Detail (UC3). 

**Alternative flows**

* A1: Empty My List

  1. If My List has no entries, the system displays an empty state message and hints on how to add stocks to the list. 

**Exception flows**

* E1: Backend error

  1. If the backend returns an error for top picks or market table, the system displays an error message and does not crash the page; user can retry loading.

## 3.3 UC3 – View Stock Detail page

| Field          | Description                                                                       |
| -------------- | --------------------------------------------------------------------------------- |
| ID             | UC3                                                                               |
| Name           | View Stock Detail                                                                 |
| Description    | User inspects detailed information and predictions for a specific ticker.         |
| Actors         | End User, Data Scientist                                                          |
| Preconditions  | User is authenticated; ticker exists in `stocks`.                                 |
| Postconditions | Price chart, predictions, and model status for the selected ticker are displayed. |

**Main flow**

1. User opens Stock Detail by clicking a stock row from Home (UC2) or using a direct URL.
2. Frontend calls backend endpoints to fetch stock metadata, recent price history, predictions per horizon, and model status.
3. System displays header information: ticker, name, current price, and recent percentage change. 
4. System renders a price chart using historical price series. 
5. System displays a predictions panel with 7D, 15D, and 30D predicted percentage changes, including up or down indicators and colors.
6. System displays model status including last trained time and MAPE metrics for each horizon.

**Alternative flows**

* A1: Missing predictions

  1. If no predictions are available for one or more horizons, system displays an explicit "Not available" or similar label for that horizon.

**Exception flows**

* E1: Ticker not found

  1. If the ticker does not exist in `stocks`, system returns a not found error and the frontend shows an informative message.

## 3.4 UC4 – Configure and start a training experiment

| Field          | Description                                                                                                             |
| -------------- | ----------------------------------------------------------------------------------------------------------------------- |
| ID             | UC4                                                                                                                     |
| Name           | Configure and run experiment                                                                                            |
| Description    | Data Scientist configures features, horizons, models, and ensemble strategy, then starts an experiment.                 |
| Actors         | Data Scientist                                                                                                          |
| Preconditions  | User is authenticated as data scientist; feature configuration screen is accessible.                                    |
| Postconditions | A new `experiment_runs` record is created; training job is enqueued; run status becomes visible on the Training screen. |

**Main flow**

1. Data Scientist opens the Training screen.
2. System loads existing feature configuration from `training_configs` or provides defaults.
3. Data Scientist selects or confirms index (VN30), tickers, lookback window, and list of technical indicators. 
4. Data Scientist selects horizon set (7D, 15D, 30D) and models (ridge regression, SVR with RBF kernel, random forest, gradient boosting).
5. Data Scientist selects ensemble strategy (for example mean, median, or weighted).
6. Data Scientist saves the configuration.
7. Data Scientist triggers an experiment; the backend creates an `experiment_runs` record and enqueues a training job.
8. Training worker consumes the job, executes data preprocessing, model training, and evaluation, and writes metrics and artifacts to Postgres and object storage.
9. Training screen updates to show run status, metrics, and links to artifacts.

**Alternative flows**

* A1: Edit existing configuration

  1. Data Scientist loads an existing configuration from `training_configs`, edits indicators or models, and saves; subsequent experiments use the updated configuration. 

**Exception flows**

* E1: Configuration validation error

  1. If configuration is invalid (for example unsupported combination of indicators), backend returns validation errors and frontend highlights invalid fields.

## 3.5 UC5 – Monitor pipelines in Airflow

| Field          | Description                                                                        |
| -------------- | ---------------------------------------------------------------------------------- |
| ID             | UC5                                                                                |
| Name           | Monitor pipelines                                                                  |
| Description    | Data Scientist monitors VN30 data and model training DAGs and runs basic controls. |
| Actors         | Data Scientist                                                                     |
| Preconditions  | User is authenticated as data scientist; pipeline metadata is synchronized.        |
| Postconditions | DAG list, run history, and statuses are visible; user may trigger or stop runs.    |

**Main flow**

1. Data Scientist opens the Pipelines screen. 
2. Frontend calls backend endpoints to retrieve DAG list and basic metadata. 
3. System displays each relevant DAG with id, description, schedule, and state.
4. Data Scientist selects a DAG to view recent runs; system loads `pipeline_runs` records and shows run ids, states, and timestamps. 
5. Data Scientist may trigger a new run; backend proxy calls Airflow and persists metadata.
6. Data Scientist may inspect tasks for a run and view pipeline logs.

**Alternative flows**

* A1: Pause DAG

  1. Data Scientist chooses to pause a DAG; backend calls Airflow to change pause state and updates metadata accordingly.

**Exception flows**

* E1: Airflow unavailable

  1. If Airflow is temporarily unavailable, backend returns an error; frontend displays an error state for pipeline operations while leaving other parts of the system functional.

## 3.6 UC6 – View Models page and evaluation plots

| Field          | Description                                                                                                           |
| -------------- | --------------------------------------------------------------------------------------------------------------------- |
| ID             | UC6                                                                                                                   |
| Name           | View models and evaluation plots                                                                                      |
| Description    | User views per ticker metrics, predictions, and evaluation plots on the Models page.                                  |
| Actors         | Data Scientist (primary), End User (optional read only)                                                               |
| Preconditions  | At least one experiment run has produced metrics and artifacts and `model_statuses` and related tables are populated. |
| Postconditions | Models table is rendered; user can open and close evaluation plot modal per ticker.                                   |

**Main flow**

1. User navigates to the Models page.
2. Frontend calls `GET /api/models` to obtain a list of tickers, last trained timestamps, MAPE values for 7D, 15D, 30D, predictions for those horizons, and plot URLs.
3. System renders the table with columns Ticker, Last Trained, MAPE 7D, MAPE 15D, MAPE 30D, Pred 7D, Pred 15D, Pred 30D, and View.
4. System applies color coding for MAPE and predictions according to configured thresholds and sign.
5. User clicks the view action for a row; frontend opens a modal and loads the evaluation plot image from the given URL.
6. User closes the modal using the Close control or by clicking outside or pressing ESC. 

**Alternative flows**

* A1: No trained models

  1. If `GET /api/models` returns an empty list, system displays an empty state message instructing the data scientist to run training experiments.

**Exception flows**

* E1: Image load failure

  1. If an evaluation plot image cannot be loaded, the modal shows a placeholder and an explanatory message instead of crashing.

---

# 4. Data model and system integration

## 4.1 Main entities and relationships

The core tables supporting the system are defined in the data model specification and can be summarized as follows. 

* `users`: Stores user accounts, including username, password hash, role, and metadata.
* `stocks`: Stores ticker level metadata for VN30 stocks such as symbol and name. 
* `stock_prices`: Stores historical daily prices per stock; used to compute features and targets and to render price charts. 
* `stock_prediction_summaries`: Stores aggregated prediction data per ticker and horizon, used for Home and Stock Detail surfaces. 
* `stock_prediction_points`: Stores time series of prediction points per ticker and horizon, used for detailed charts and evaluation. 
* `model_statuses`: Stores per ticker model level information including latest experiment run id, status, and summary metrics. 
* `model_horizon_metrics`: Stores per horizon metrics such as MAPE values for 7D, 15D, and 30D per ticker and model status. 
* `training_configs`: Stores reusable configuration for feature engineering, horizons, and models.
* `experiment_runs`: Stores individual training runs, including configuration reference, state, metrics, and artifacts, and links to logs and ticker level artifacts.
* `experiment_logs` and `experiment_ticker_artifacts`: Store run level logs and object storage URLs for artifacts such as evaluation plots and future prediction CSVs.
* `pipeline_dags`, `pipeline_runs`, `pipeline_run_tasks`, `pipeline_run_logs`: Store mirrored Airflow metadata and logs to support pipeline monitoring.

The ER summary indicates that users are linked to training configurations, experiments, and pipeline runs; stocks are linked to prices, predictions, model statuses, and artifacts; model statuses are linked to horizon metrics; training configurations are linked to experiment runs and further to logs and artifacts; experiment runs optionally link back into prediction tables; and pipelines tables are linked hierarchically from DAGs to runs to tasks and logs. 

## 4.2 System architecture and data flow

The system architecture includes the following components.

* Frontend web app

  * Single page application that implements all UI screens (Login, Home, Stock Detail, Training, Pipelines, Models).
  * Talks only to the backend API and handles navigation, loading states, and role based menus.

* Backend API service

  * Python service exposing `/api/v1/**` endpoints, implementing authentication, authorization, validation, business logic, and integration with Postgres, Airflow, cache, and object storage.

* Training worker

  * Background worker that consumes training jobs, executes ML pipelines, and writes metrics and artifacts to the database and object storage.

* Airflow

  * Orchestrates VN30 data crawler and model training DAGs, exposes status and history that the backend wraps as pipeline endpoints.

* PostgreSQL database and object storage

  * Serve as system of record for all relational data and as storage for evaluation plots, model pickles, and future predictions.

![High level architecture of the stock prediction system](/docs/diagrams/system-architecture.png)
Figure 4.1 High level architecture of the stock prediction system

This diagram shows the main components of the stock prediction system and how they interact. End users and data scientists access the system through a browser that loads the single page frontend application. The frontend communicates with a backend API service over HTTP and JSON, which centralizes all business logic, authentication, model and prediction queries, and pipeline control. The backend reads and writes relational data in PostgreSQL, stores plots and model artifacts in object storage, and proxies requests to Airflow for VN30 data crawling and training pipelines. Training jobs are enqueued by the backend and executed by a separate training worker, which builds features, trains per horizon models, computes metrics, and writes predictions and artifacts back to the database and storage. The frontend then uses the API to render the Login, Home, Stock Detail, Training, Pipelines, and Models pages based on this shared infrastructure.


High level data flow from raw market data to predictions in the UI is as follows.

1. VN30 data crawling

   * Airflow periodically runs `vn30_data_crawler` DAG to fetch raw market data and write it into `stock_prices` and related tables.

2. Feature engineering and training

   * When a data scientist starts an experiment, training worker reads historical prices and computes technical indicators and features according to the configuration.
   * Worker constructs time ordered feature windows and targets for each horizon, trains models, and computes metrics such as MAPE.

3. Predictions and aggregation

   * Worker writes prediction points and summary metrics into `stock_prediction_points`, `stock_prediction_summaries`, `model_statuses`, and `model_horizon_metrics`.

4. Artifacts and model metadata

   * Worker generates evaluation plots and future prediction CSVs per ticker and uploads them to object storage; their URLs are stored in `experiment_ticker_artifacts` and surfaced via `/api/models`.

5. Frontend consumption

   * Home and Stock Detail pages call backend endpoints that read from prediction and model status tables (often via cache) to prepare responses.
   * Models page calls `/api/models` to retrieve model metadata and plot URLs and renders them in a summary table.

---

# 5. APIs and component interactions

This section summarizes key API groups and how they support the user flows and requirements. Detailed endpoint names follow the API mapping in the system architecture specification.

## 5.1 Authentication and session management

* Authentication endpoints handle login and token issuance based on credentials stored in `users`.
* Upon successful authentication, the frontend stores session information and adjusts navigation based on the role returned.
* Error responses are structured with codes and messages to support user friendly error states on the login page.

## 5.2 Home top picks, market table, and watchlist

* A set of endpoints exposes top picks per tab (Should Buy, Should Sell, My List), using aggregated prediction data and user specific watchlist associations.
* Market table endpoints return combined actual and predicted percentage changes, plus metadata needed to render columns and sparklines.
* Watchlist endpoints allow users to add or remove tickers from My List; Home uses these endpoints to compute the My List tab content.

## 5.3 Stock detail and time series data

* Stock detail endpoints provide ticker metadata, recent price history, and predictions per horizon.
* Time series endpoints fetch either raw or derived series used for price charts and evaluation overlays; they read from `stock_prices` and `stock_prediction_points`.
* Model status endpoints provide summary metrics and labels displayed on Stock Detail.

## 5.4 Training experiments

* Configuration endpoints (`/features/config`, `/features/validate`) allow retrieval and update of `training_configs` and validation of feature choices before running experiments.
* Experiment endpoints (`/experiments/run`, `/experiments/{runId}`, `/experiments/{runId}/logs/tail`, `/experiments/{runId}/artifacts`) allow enqueuing training jobs, monitoring their status, tailing logs, and listing artifacts.
* These endpoints directly support UC4 by allowing data scientists to configure and run experiments and observe metrics produced by the ML models.

## 5.5 Pipeline control and monitoring

* Pipeline endpoints provide listing, detail, run history, triggering, pausing, stopping, and configuration operations for Airflow DAGs:
  * `GET /pipeline/dags` – list all DAGs with status and last run info
  * `GET /pipeline/dags/{dagId}` – get DAG details including schedule, owner, tags
  * `GET /pipeline/dags/{dagId}/runs` – list runs for a DAG with filters (state, date range, pagination)
  * `GET /pipeline/runs/{runId}` – get run details
  * `POST /pipeline/dags/{dagId}/trigger` – trigger a new DAG run with optional config payload
  * `POST /pipeline/dags/{dagId}/pause` – pause or resume a DAG
  * `POST /pipeline/dags/{dagId}/stopRun` – stop an active DAG run
  * `PATCH /pipeline/dags/{dagId}/settings` – update DAG settings (schedule, retries, timezone)
  * `POST /pipeline/dags/sync` – sync DAGs from Airflow to local database
  * `GET /pipeline/runs/{runId}/graph` – get task graph for visualization
  * `GET /pipeline/runs/{runId}/gantt` – get Gantt chart data for task timeline
  * `GET /pipeline/runs/{runId}/logs` – get logs for a run or specific task
* These endpoints are proxies around Airflow REST or DB level operations, sometimes updating mirrored metadata in `pipeline_*` tables.
* They support UC5 by enabling monitoring and control of data and training pipelines from within the web app.

## 5.6 Models overview and model details

* `GET /api/models` returns a list of model entries per ticker, including last trained time, MAPE values for 7D, 15D, 30D, predictions for those horizons, and plot URLs.
* Internally, this endpoint reads from model status and metrics tables and joins with artifact metadata pointing to evaluation plot images in object storage.
* These endpoints directly support UC6 and connect ML training artifacts with the Models UI.

---

# 6. Machine learning models and mathematical description

## 6.1 Prediction problem formulation

For each stock ticker and prediction horizon $h \in \{7, 15, 30\}$, the system models the percentage change in closing price $h$ days into the future.

Let $P_t$ denote the closing price of a given stock at trading day $t$. For a chosen horizon $h$, the target is defined as

$$
y_{t+h} = \frac{P_{t+h} - P_t}{P_t} \times 100
$$

expressed as a percentage change between day $t$ and day $t+h$.

The feature vector $\mathbf{x}_t$ at time $t$ includes, for a fixed lookback window:

* Raw prices such as open, high, low, close, and volume over recent days.
* Derived technical indicators computed from the historical window, including but not limited to: simple and exponential moving averages (SMA, EMA), Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Bollinger Bands, Average True Range (ATR), and volume moving averages.

For each horizon $h$, the prediction problem is to learn a function $f_h$ such that

$$
\hat{y}_{t+h} = f_h(\mathbf{x}_t)
$$

approximates the true target $y_{t+h}$ for all valid time indices $t$. Given a dataset of pairs $\{(\mathbf{x}_t, y_{t+h})\}_{t=1}^{N_h}$ constructed from historical data, the models described below are trained to minimize suitable loss functions over this dataset.

## 6.2 Selected models and their mathematics

The system uses several regression models per horizon and ticker, as defined in the training configuration.

### 6.2.1 Ridge regression

Ridge regression is a linear regression model with L2 regularization that assumes a linear relationship between features and target.

The prediction function has the form

$$
\hat{y} = \mathbf{w}^\top \mathbf{x} + b
$$

where $\mathbf{w}$ is the weight vector and $b$ is the bias term.

Given training examples $(\mathbf{x}_i, y_i)$ for $i = 1,\dots,N$, ridge regression solves

$$
\min_{\mathbf{w}, b} \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{w}^\top \mathbf{x}_i - b)^2 + \lambda \lVert \mathbf{w} \rVert_2^2
$$

where $\lambda \ge 0$ is the regularization parameter.

 The regularization term $\lambda \lVert \mathbf{w} \rVert_2^2$ discourages large weights, which helps control variance and reduces overfitting, especially when features are correlated or the feature dimension is large relative to the number of training examples.

### 6.2.2 Support Vector Regression (SVR with RBF kernel)

Support Vector Regression aims to find a function that deviates from the targets by at most $\epsilon$ for as many training points as possible, while maintaining a flat function in feature space.

Conceptually, SVR solves an optimization problem of the form

$$
\min_{\mathbf{w}, b, \xi_i, \xi_i^*}
\frac{1}{2} \lVert \mathbf{w} \rVert^2
+ C \sum_{i=1}^N (\xi_i + \xi_i^*)
$$

subject to

$$
\begin{aligned}
y_i - (\mathbf{w}^\top \phi(\mathbf{x}_i) + b) &\le \epsilon + \xi_i \\
(\mathbf{w}^\top \phi(\mathbf{x}_i) + b) - y_i &\le \epsilon + \xi_i^* \\
\xi_i, \xi_i^* &\ge 0
\end{aligned}
$$

where $\phi(\cdot)$ is a mapping to a feature space, $C > 0$ is a regularization parameter controlling the trade off between flatness and allowed deviations, and $\epsilon \ge 0$ defines the width of the $\epsilon$ insensitive band around the regression function.

Using the kernel trick, SVR with RBF kernel uses

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \lVert \mathbf{x}_i - \mathbf{x}_j \rVert^2)
$$

to define similarity between data points.

 Parameter $\gamma > 0$ controls the effective width of the kernel; smaller values lead to smoother functions that vary slowly with $\mathbf{x}$, while larger values allow more complex, localized fits around individual training points.

### 6.2.3 Random Forest Regression

Random forest regression is an ensemble of decision trees trained on bootstrap samples of the training data with feature subsampling at each split.

Let $h_t(\mathbf{x})$ denote the prediction of tree $t$ for $t = 1,\dots,T$. The random forest prediction is

$$
\hat{y}(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^T h_t(\mathbf{x})
$$

 Each tree is trained on a bootstrap resample of the training data and, at each split, considers a random subset of features, which encourages diversity among trees; averaging across trees reduces variance and often improves generalization compared to a single tree.

### 6.2.4 Gradient Boosting Regression

Gradient boosting builds an additive model by sequentially adding decision trees that correct errors of the current model.

Let $F_m(\mathbf{x})$ denote the model after $m$ iterations. The general form is

$$
F_M(\mathbf{x}) = \sum_{m=1}^M \nu \, h_m(\mathbf{x})
$$

where $h_m$ is the tree added at iteration $m$, $M$ is the number of boosting stages, and $0 < \nu \le 1$ is the learning rate that scales each tree's contribution.

 At each iteration, a new tree is fitted to approximate the negative gradient of the loss function with respect to the current model predictions; for squared error loss, this gradient reduces to residuals $y_i - F_{m-1}(\mathbf{x}_i)$.

## 6.3 Data preprocessing and splitting

* Lookback window construction

  * For each ticker, time ordered daily prices and volumes are collected from `stock_prices`.
  * For each time index $t$ where sufficient history exists, a feature vector $\mathbf{x}_t$ is constructed using raw prices and technical indicators computed over the lookback window.

* Target alignment

  * For each horizon $h$, the target $y_{t+h}$ is computed using prices at times $t$ and $t+h$, as defined in Section 6.1.
  * Feature vectors without valid future prices are excluded from training for that horizon.

* Train and test split strategy

  * The data is split into training and test segments along the time axis to respect temporal ordering, typically by selecting an earlier period for training and a more recent contiguous block for testing; shuffling across time is avoided to mimic realistic trading conditions.

* Standardization and scaling

  * For models sensitive to feature scales, such as ridge regression and SVR, features are standardized using statistics computed on the training set (for example mean and standard deviation per feature) and the same transformation is applied to test data.

![Machine learning pipeline data flow](/docs/diagrams/ml-pineline.png)

Figure 6.1 Machine learning pipeline data flow

This diagram summarizes the end to end data flow of the machine learning pipeline. Historical price data for the selected VN30 subset is first transformed into time ordered feature vectors using a fixed lookback window and a set of technical indicators. The data is then split into training and validation segments that respect time order and used to build per horizon datasets for 7, 15, and 30 day targets. For each horizon, multiple models such as ridge regression, SVR with RBF kernel, random forest, and gradient boosting are trained and produce per model predictions. These predictions are combined by an ensemble module, evaluated using MAPE on the validation set, and the resulting predictions, metrics, and plots are written to the database and object storage. Finally, the backend API exposes this information to the frontend, which renders the Home, Stock Detail, and Models pages.

## 6.4 Ensemble strategies


For each horizon $h$, the system can combine predictions from multiple models (for example ridge regression, SVR, random forest, gradient boosting) into an ensemble prediction.

Let $\hat{y}_{h,k}$ denote the prediction of model $k$ for horizon $h$. A simple weighted ensemble prediction is

$$
\hat{y}_h = \sum_k w_{h,k} \hat{y}_{h,k}
$$

with weights satisfying

$$
\sum_k w_{h,k} = 1, \quad w_{h,k} \ge 0.
$$

 Weights may be chosen based on validation performance such as inverse MAPE on a validation set, so that models with better historical accuracy receive higher weight in the ensemble. Mean ensemble corresponds to equal weights; median ensemble corresponds to taking the median rather than a weighted sum.

## 6.5 Evaluation metrics (MAPE for 7D, 15D, 30D)

For a given horizon and a set of $N$ test examples with targets $y_i$ and predictions $\hat{y}_i$, the Mean Absolute Percentage Error (MAPE) is defined as

$$
\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right|.
$$

MAPE is computed separately for each horizon (7D, 15D, 30D) and stored per ticker and model in `model_horizon_metrics`.

The Models page displays these values in columns labeled MAPE 7D, MAPE 15D, and MAPE 30D, using color coding based on thresholds: green for values below 5 percent, yellow for values between 5 and 10 percent, and red for values above 10 percent.

These MAPE values can be interpreted as average relative errors for the given horizon; lower values indicate more reliable forecasts. They can also be mapped to status labels such as excellent, acceptable, and needs improvement as reflected by color codes on the Models and Stock Detail pages.

**Note:** Although MAPE is widely used as a percentage based error metric, it has a known limitation when the true target value $y_i$ is equal to zero or extremely close to zero, because the term $|y_i - \hat{y}_i| / |y_i|$ becomes undefined or numerically unstable. In our setting, this situation can theoretically occur, since the target represents percentage price changes and some values may be near zero. 
In this report, we acknowledge this limitation but use the standard MAPE implementation without additional corrections, in order to keep the evaluation pipeline simple and comparable across models. A more robust treatment, for example clipping the denominator away from zero or switching to alternative metrics such as SMAPE, is left for future work and is outside the scope of this assignment.

## 6.6 Link between mathematics and UI

The mathematical quantities defined above are directly reflected in the UI in several ways.

* Predicted percentage change $\hat{y}_h$

  * For each horizon $h \in \{7, 15, 30\}$, the ensemble prediction $\hat{y}_h$ is displayed as Pred 7D, Pred 15D, and Pred 30D in the Models table, and as horizon specific predicted changes on Stock Detail and Home.
  * The sign of $\hat{y}_h$ determines arrow direction and color: positive values produce upward arrows and green color, while negative values produce downward arrows and red color.

* Actual versus predicted comparison

  * Evaluation plots per ticker display time series of actual prices alongside model predictions, which are derived from $\hat{y}_{t+h}$ values applied cumulatively over time.
  * These plots provide a visual representation of model fit consistent with the numerical metrics in the Models table.

* MAPE driven quality indicators

  * MAPE values computed as in Section 6.5 are presented in the Models page and inform color coding and qualitative labels.
  * Stock Detail can re use these metrics to show a concise assessment of current model quality for the selected ticker.

* Model freshness and last trained time

  * The `last_trained` timestamp per ticker, derived from the most recent successful experiment run linked to `model_statuses`, is displayed in the Models table and can be rendered as relative time (for example “2 days ago”).
  * Users can interpret fresher models with strong MAPE performance as more trustworthy for current market conditions.

Through these mechanisms, the mathematical formulation of the prediction problem and models is tightly connected to what users observe: numeric predictions and errors, visual evaluation plots, and status indicators used to guide investment oriented interpretation.



# 7. Implementation artifacts

This section lists the main source code artifacts that implement the stock prediction system described in this document.

## 7.1 Backend repository

* **Name**: Stock prediction backend
* **URL**: [https://github.com/vu42/stock-prediction](https://github.com/vu42/stock-prediction)

## 7.2 Frontend repository

* **Name**: Stock prediction web UI
* **URL**: [https://github.com/vu42/stock-prediction-ui](https://github.com/vu42/stock-prediction-ui)   


# 8. References

[1] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd ed. New York: Springer, 2009.
(General background on supervised learning, regularization, and ensemble methods such as random forests and gradient boosting.)

[2] C. M. Bishop, *Pattern Recognition and Machine Learning*. New York: Springer, 2006.
(Reference for probabilistic modeling, regression, loss functions, and evaluation metrics.)

[3] A. J. Smola and B. Schölkopf, “A tutorial on support vector regression,” *Statistics and Computing*, vol. 14, no. 3, pp. 199–222, 2004.
(Primary reference for Support Vector Regression and ε–insensitive loss.)

[4] L. Breiman, “Random forests,” *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
(Foundational paper introducing random forest models.)

[5] J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.
(Original paper on gradient boosting and additive tree models.)

[6] R. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021.
(Accessible reference for time series forecasting, evaluation, and error measures such as MAPE.)

[7] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
(Reference for the practical implementation of ridge regression, SVR, random forest, and gradient boosting used in many Python based ML pipelines.)

[8] Apache Airflow, “Apache Airflow Documentation,” The Apache Software Foundation.
(Reference for DAG based workflow orchestration used for VN30 data crawling and model training pipelines.)

[9] PostgreSQL Global Development Group, “PostgreSQL 16 Documentation,” 2023.
(Reference for the relational database used to store users, prices, predictions, model metrics, and pipeline metadata.)

[10] Ho Chi Minh Stock Exchange (HOSE), “VN30 Index Methodology,” Ho Chi Minh Stock Exchange.
(Background on the VN30 index composition and methodology relevant to the choice of VN30 tickers in the system.)
