# Stock Prediction App Specs

## Table of Contents

- [1. Overview](#1-overview)
   * [1.1 Members](#11-members)
   * [1.2 Purpose](#12-purpose)
   * [1.3 Roles and Navigation](#13-roles-and-navigation)
- [2. Functional Requirements (UI & Behavior)](#2-functional-requirements-ui-behavior)
   * [2.1 Screens](#21-screens)
   * [2.2 Login](#22-login)
   * [2.3 End User – Home](#23-end-user-home)
   * [2.4 End User – Stock Detail](#24-end-user-stock-detail)
   * [2.5 Data Scientist – Training](#25-data-scientist-training)
   * [2.6 Data Scientist – Pipelines (Airflow Control)](#26-data-scientist-pipelines-airflow-control)
   * [2.7 Data Scientist – Models](#27-data-scientist-models)
- [3. Role-Based Navigation](#3-role-based-navigation)
- [4. Non-Functional UI Requirements](#4-non-functional-ui-requirements)
   * [4.1 Visual System (consistent with DS pages)](#41-visual-system-consistent-with-ds-pages)
   * [4.2 States and Accessibility](#42-states-and-accessibility)
   * [4.3 Test Accounts (for prototypes)](#43-test-accounts-for-prototypes)
- [5. API Specification](#5-api-specification)
   * [5.1 API Mapping (per UI page)](#51-api-mapping-per-ui-page)
- [6. Data Model (Database Schema & ER)](#6-data-model-database-schema-er)
   * [6.1 Database Schema (overview)](#61-database-schema-overview)
   * [6.2 PostgreSQL schema (reference)](#62-postgresql-schema-reference)
   * [6.3 ER relationships (summary)](#63-er-relationships-summary)
- [7. System & Backend Architecture](#7-system-backend-architecture)
   * [7.1 System Architecture Overview](#71-system-architecture-overview)
   * [Major components](#major-components)
   * [Supporting services / infrastructure](#supporting-services-infrastructure)
   * [Component interactions & data flow (summary)](#component-interactions-data-flow-summary)
   * [7.2 Python Backend Monorepo Architecture](#72-python-backend-monorepo-architecture)
   * [Components inside the monorepo](#components-inside-the-monorepo)
   * [Recommended directory structure (backend repo)](#recommended-directory-structure-backend-repo)
   * [Best practices for code organization & configuration](#best-practices-for-code-organization-configuration)
   * [Local development & workflows (backend)](#local-development-workflows-backend)
   * [Connecting the ReactJS frontend (separate repo)](#connecting-the-reactjs-frontend-separate-repo)
   * [7.3 Design choices and options](#73-design-choices-and-options)
- [8. Open Questions](#8-open-questions)
- [9. Deliverables](#9-deliverables)
- [10. Change Log](#10-change-log)

## 1. Overview

### 1.1 Members

Group 2 - Intelligent Systems:

Nguyễn Viết Vũ

Phan Hoàng Tú

Trần Hoàng Duy

### 1.2 Purpose

- Explain the intent of each page, user roles, navigation, and core interactions shown in the mockups.
- Align UI with backend/API and clarify acceptance criteria.

### 1.3 Roles and Navigation

- Roles: End User, Data Scientist
- Top nav (role-based):
  - End User: Home, Stock Detail (default landing: Home)
  - Data Scientist: Training, Pipelines, Models, Home, Stock Detail (default landing: Training)

## 2. Functional Requirements (UI & Behavior)

### 2.1 Screens

- Login
- End User – Home
- End User – Stock Detail
- Data Scientist – Training
- Data Scientist – Pipelines
- Data Scientist – Models

### 2.2 Login

- Goal: Role-based access and default landing.
- Contents:
  - Username, Password, Show/Hide password
  - Sign In button
  - Test accounts (rendered on card):
    - End User: enduser1/pass1234, enduser2/pass1234
    - Data Scientist: ds1/pass1234, ds2/pass1234
  - Hint: “End Users → Home, Data Scientists → Training”
- States: idle, invalid credentials (error), loading (spinner on button)
- Acceptance:
  - enduser1 → Home with Home/Stock Detail in nav
  - ds1 → Training with Training/Pipelines/Models/Home/Stock Detail in nav

### 2.3 End User – Home

- Goal: Surface top opportunities, browse market table.
- Top Picks (tabs):
  - Should Buy: 5 stocks with highest predicted % growth (green)
  - Should Sell: 5 stocks with highest predicted % drop (red), i.e., predicted to fall 
  - My List: 5 stocks that the user has added to their list
- Market Table:
  - Name, Symbol, Current Price, % Change 7D IRL, % Change 15D IRL, % Change 30D IRL, % Change 7D Predicted, 14D sparkline
  - Each horizon cell shows `percentage / price`, e.g. `+3.0% / 32000.125` (actual close vs actual close price)
  - Search (by symbol, name), Filter (category/sector/My list), Sort ( % change IRL: increase, decrease, % change predicted: increase, decrease, price currently)
  - Row click → Stock Detail
- States: loading (skeleton), empty (no matches), error (retry)
- Acceptance:
  - Tab switches update lists
  - Horizon columns clearly show `percentage / price` pairs
  - Table sort/search/filter works, row navigates to Stock Detail

### 2.4 End User – Stock Detail

- Goal: Explain current and predicted performance for a ticker.
- Layout: 25% Predictions & Model Status, 50% Price & Forecast chart, 25% Stock Overview
- Predictions: 7D/15D/30D % change with green/red cues
- Chart: historical vs predicted, range tabs (7D/15D/30D), tooltip date/actual/predicted
- Overview: company logo/name, short description, key stats (market cap, volume), links
- Model status:
  - Status pill (e.g. Fresh/Stable/Stale) indicating recency of latest training run
  - Last updated timestamp (e.g. `2025-11-03 17:05 ICT`) with hint “From latest training run”
  - Headline accuracy metric for key horizon (e.g. `MAPE (7D): 3.2%`), colored by quality
  - Horizon chips (7D/15D/30D) each showing error %, e.g. `7D 2.9%`, `15D 3.6%`, `30D 4.8%`
- Acceptance:
  - Tabs update chart range
  - Tooltip shows actual and predicted values
  - Model status shows last updated time and accuracy metrics per horizon for the selected ticker

### 2.5 Data Scientist – Training

- Goal: Configure stock universe, data window, indicators, targets, models, ensemble, and reproducibility, then validate and save the training configuration.
- Layout: Single scrollable column of configuration cards, each with a small `Reset` link in the top-right; primary actions pinned in a footer bar.
- Stock Universe:
  - VN30 tickers shown as selectable chips (e.g., FPT, VCB, VNM, HPG, VIC, VHM, MSN, SAB, TCB, GAS), multiple selection supported.
  - `Use all VN30` toggle to quickly select the full VN30 universe.
- Data Window:
  - Choice between `Last N days` (numeric input, e.g., 240) and `Date range`.
  - Checkbox `Skip re-fetch (use DB)` to reuse existing data.
  - Helper text about VN market calendar and holiday handling.
- Indicators:
  - Price: `SMA` (multi-window input, e.g., 5 / 20 / 60), `EMA` (fast/slow, e.g., 12 / 26), optional `ROC`.
  - Momentum: `RSI` (window, e.g., 14), `MACD` (fast/slow/signal, e.g., 12 / 26 / 9).
  - Volatility: `Bollinger Bands` (window + std, e.g., 20 / 2), `ATR` (window, e.g., 14).
  - Volume: `Volume MA` (window, e.g., 20).
  - Leakage guard: toggle with helper text "enforce t-1 or earlier features".
- Target & Splits:
  - Horizon chips: 7d, 15d, 30d (multi-select).
  - `Lookback window` numeric input (e.g., 60).
  - `Train %` and `Test %` inputs (e.g., 80 / 20).
- Models & Parameters:
  - Model checkboxes: `RandomForest`, `GradientBoosting`, `SVR (RBF)`, `Ridge`.
  - Each model shows its key hyperparameters inline:
    - RandomForest: `n_estimators`, `max_depth`.
    - GradientBoosting: `n_estimators`, `learning_rate`.
    - SVR (RBF): `C`, `epsilon`, `gamma` (with `scale` as default).
    - Ridge: `alpha`.
- Scaling:
  - Radio options: `StandardScaler` or `None`.
  - Helper text: "SVR works best with scaling."
- Ensemble:
  - Options list: `Mean`, `Median`, `Weighted`.
  - Toggle `Learn weights from CV` with helper text "Optimizes MAE via time-series CV; weights may differ per horizon."
- Reproducibility:
  - `Random seed` numeric input (e.g., 42).
- Actions:
  - Footer buttons: `Validate Config`, `Save Changes`, `Discard Changes`.
- Acceptance:
  - Validate Config highlights missing/invalid fields and prevents save until resolved.
  - Save Changes persists the current configuration; Discard Changes reverts to the last saved config.
  - Reset on each card restores that section to its default values.

### 2.6 Data Scientist – Pipelines (Airflow Control)

- Goal: Trigger and monitor DAG runs, inspect history and run details, and edit DAG-level settings.
- Layout: Left column shows DAG catalog cards; right column shows details for the selected DAG with tabs.
- DAG Catalog (left):
  - Cards for `vn30_data_crawler` and `vn30_model_training`, each with name, short description, status pill (Active/Paused), schedule line (`cron` + human label), next run, and last run with state.
  - Primary actions on each card: `Run now` and `Pause` (toggles to `Resume` when paused).
  - Selected DAG card highlighted with a blue border.
- Selected DAG header (right):
  - DAG name and status pill.
  - Primary controls: `Trigger Run`, `Pause` (or `Resume`), `Stop Active Run` (only enabled when there is a running instance).
  - Metadata row: Owner, Tags, Timezone.
- Tabs (right): `Overview`, `Run History`, `Run Details`, `Edit DAG`.
- Overview tab:
  - `Status & Schedule` card: state, schedule (cron + human label), catchup on/off, max active runs.
  - `Current/Last Run` card: run_id, duration, state pill, and actions `View logs` and `Rerun with same conf`.
- Run History tab:
  - Filter row with state chips (Running, Success, Failed, Queued), date range pickers (From/To), and `Search run_id` field plus `Reset filters` link.
  - Table with columns: run_id, Start, End, Duration, Triggered by, State, Actions (e.g., `Run`/`View`).
  - Pagination controls (Previous/Next) and total count (e.g., “Showing 1–10 of 26 runs”).
- Run Details tab:
  - Run summary (same fields as Current/Last Run) followed by sub-tabs: `Graph`, `Gantt`, `Logs`.
  - Graph view shows DAG boxes (e.g., start → fetch_api → push_to_db → end) with success state styling and a caption when all tasks complete.
  - Logs view shows a scrollable text area with run logs.
- Edit DAG tab:
  - `DAG Settings` card: CRON schedule input, timezone select, `Catchup` toggle, `Max active runs` input.
  - `Default Arguments` card: `Retries`, `Retry delay (min)`, `Owner`, `Tags` chips with add/remove, and helper text “Changes affect future runs only.”
  - Footer actions: `Save changes` and `Discard`.
- Acceptance:
  - Trigger Run starts a new run that appears in Run History and becomes the Current/Last Run.
  - Stop Active Run is only enabled while a run is in Running state and moves it to a terminal state when used.
  - Edits to schedule/timezone/catchup/max active runs/retries/owner/tags persist after Save changes and affect future runs only.

### 2.7 Data Scientist – Models

- Goal: Monitor trained model performance and predictions across all VN30 stocks.
- Layout: Simple table view with essential metrics and predictions for each trained model.
- Table columns:
  - Ticker: Stock symbol (e.g., FPT, VCB, HPG)
  - Last Trained: Relative time since last training (e.g., "2 days ago")
  - MAPE 7D: Model accuracy for 7-day predictions (%)
  - MAPE 15D: Model accuracy for 15-day predictions (%)
  - MAPE 30D: Model accuracy for 30-day predictions (%)
  - Pred 7D: Predicted % change for 7 days ahead
  - Pred 15D: Predicted % change for 15 days ahead
  - Pred 30D: Predicted % change for 30 days ahead
  - View: Button to open evaluation plot modal
- Color coding:
  - MAPE values: Green (< 5%, excellent), Yellow (5-10%, acceptable), Red (> 10%, needs improvement)
  - Predictions: Green with ↑ arrow for positive changes, Red with ↓ arrow for negative changes
- Plot modal:
  - Opens when user clicks View button
  - Displays evaluation plot image from S3 showing actual vs predicted prices with metrics overlay
  - Close button (X) and clicking outside modal closes it
  - ESC key also closes modal
- States: loading (table skeleton), empty (no models found), error (retry button)
- Acceptance:
  - Table displays all VN30 stocks with trained models
  - MAPE values are color-coded according to performance thresholds
  - Predictions show directional arrows and color coding
  - Clicking View button opens modal with evaluation plot from S3
  - Modal closes properly via X button, outside click, or ESC key
  - Images load correctly from S3 URLs

## 3. Role-Based Navigation

- After successful login:
  - End User: show Home (active), Stock Detail, hide DS pages
  - Data Scientist: show Training (active), Pipelines, Models, Home, Stock Detail


## 4. Non-Functional UI Requirements

### 4.1 Visual System (consistent with DS pages)

- Frame 1440×1024, light theme
- 16px spacing, 8px card radius, subtle shadows
- Same tabs, chips, tables, buttons, inputs, pills, status pills
- Green for positive, red for negative, gray neutrals, monospace for logs

### 4.2 States and Accessibility

- States: loading (skeleton), empty, error (inline messages), disabled controls
- A11y: sufficient contrast, focus rings on inputs/buttons, ARIA labels for tabs, keyboard navigation

### 4.3 Test Accounts (for prototypes)

- End User: enduser1/pass1234, enduser2/pass1234 → default Home → access to Home, Stock Detail
- Data Scientist: ds1/pass1234, ds2/pass1234 → default Training → access to Training, Pipelines, Models, Home, Stock Detail

## 5. API Specification

### 5.1 API Mapping (per UI page)

- Login:
  - POST /api/v1/auth/login
    - Body: { username, password }.
    - Response: { refreshToken, user: { id, username, role (end_user|data_scientist), displayName } }.
    - Errors: 401 with { code, message } for invalid credentials.
  - GET /api/v1/auth/me
    - Purpose: restore session on refresh and drive role-based nav.
    - Response: { user: { id, username, role, displayName } }.

- End User – Home:
  - GET /api/v1/stocks/top-picks
    - Purpose: power "Should Buy" and "Should Sell" tabs.
    - Query params: role=end_user, bucket=should_buy|should_sell, limit (default 5), horizonDays (default 7).
    - Response: [ { ticker, name, sector, horizonDays, predictedChangePct, currentPrice } ].
  - GET /api/v1/stocks/my-list
    - Purpose: power "My List" tab with the current user's saved stocks.
    - Query params: limit (default 5), horizonDays (default 7).
    - Response: [ { ticker, name, sector, horizonDays, predictedChangePct, currentPrice, addedAt } ].
    - Errors: 401 for unauthenticated, 4xx/5xx with { code, message }.
  - GET /api/v1/stocks/market-table
    - Purpose: populate market table with search/filter/sort and actual vs predicted %.
    - Query params: search (string), sector (string, optional), sortBy (change_7d|change_15d|change_30d|price|predicted_change_7d), sortDir (asc|desc), page, pageSize.
    - Response:
      - data: [ { symbol, name, sector, currentPrice, pctChange: { "7d": { actualPct, actualPrice }, "15d": { actualPct, actualPrice }, "30d": { actualPct, actualPrice } }, predictedPctChange: {"7d": {predictedPct, predicedtPrice}}, sparkline14d: [ { date, price, isPredicted } ] } ].
      - meta: { total, page, pageSize, sectors: ["Technology", ...] }.
    - Errors: 4xx/5xx with { code, message } to support error state.
- End User – Stock Detail:
  - GET /api/v1/stocks/{ticker}
    - Purpose: populate Stock Overview panel.
    - Path params: ticker (e.g., "FPT").
    - Response: { ticker, name, logoUrl, description, sector, exchange, marketCap, tradingVolume, links: { financialReportUrl, companyWebsiteUrl } }.
  - GET /api/v1/stocks/{ticker}/predictions
    - Purpose: populate Predicted Change card (7D/15D/30D).
    - Query params: horizons=7,15,30 (comma-separated days).
    - Response: { ticker, horizons: { "7": { predictedChangePct }, "15": { predictedChangePct }, "30": { predictedChangePct } } }.
  - GET /api/v1/stocks/{ticker}/chart
    - Purpose: drive Price & Forecast chart with range tabs.
    - Query params: historicalRange=15d|30d|60d|90d, predictionRange=7d|15d|30d
    - Response: { points: [ { date, actualPrice, predictedPrice? } ], range }.
  - GET /api/v1/models/{ticker}/status
    - Purpose: fill Model status card (state, last updated, MAPE per horizon).
    - Response: { state: "fresh"|"stable"|"stale", lastUpdatedAt, metrics: { "7d": { mapePct }, "15d": { mapePct }, "30d": { mapePct } } }.

- Data Scientist – Training:
  - GET /api/v1/features/config
    - Purpose: load latest saved training configuration for the current user/team.
    - Response: full config object matching UI sections (universe, dataWindow, indicators, targets, models, ensemble, reproducibility).
  - POST /api/v1/features/config
    - Purpose: create or update a saved configuration.
    - Body: config object from UI.
    - Response: { configId, savedAt }.
  - POST /api/v1/features/validate
    - Purpose: validate config and compute a rough run preview / cost estimate.
    - Body: config object or { configId }.
    - Response: { isValid, blockers: [ { fieldPath, message } ], warnings: [ ... ], runPreview: { estRuntimeMinutes, estCost? } }.
  - POST /api/v1/experiments/run
    - Purpose: start a new training run using a validated configuration.
    - Body: { configId, scope: "all_vn30"|"selected", seeds: { globalSeed }, notes }.
    - Response: { runId }.
  - GET /api/v1/experiments/{runId}
    - Purpose: power Active Run panel (progress, ETA, state).
    - Response: { runId, state, progressPct, eta, startedAt, finishedAt?, scope, notes }.
  - GET /api/v1/experiments/{runId}/logs/tail
    - Purpose: fill Log Tail area.
    - Query params: cursor (optional) for incremental streaming.
    - Response: { entries: [ { timestamp, level, message } ], nextCursor? }.
  - GET /api/v1/experiments/runs
    - Purpose: list Past Runs.
    - Query params: limit, cursor or page/pageSize, state (optional).
    - Response: { data: [ { runId, createdAt, state, configSummary } ], meta: { nextCursor? } }.
  - GET /api/v1/experiments/{runId}/artifacts
    - Purpose: list artifacts per ticker (metrics, plots, models, predictions).
    - Query params: ticker (optional).
    - Response: { tickerArtifacts: [ { ticker, metrics: { mape7dPct, ... }, files: [ { type: "evaluation_png"|"future_png"|"model_pkl"|"scaler_pkl"|"future_predictions_csv", url } ] } ] }.

- Data Scientist – Pipelines (Airflow Control):
  - GET /api/v1/pipeline/dags
    - Purpose: populate left-hand DAG catalog.
    - Response: [ { dagId, name, description, status, scheduleCron, scheduleLabel, nextRunAt, lastRunAt, lastRunState } ].
  - GET /api/v1/pipeline/dags/{dagId}
    - Purpose: populate selected DAG header and Overview tab basics.
    - Response: { dagId, name, status, owner, tags, timezone, scheduleCron, scheduleLabel, catchup, maxActiveRuns }.
  - POST /api/v1/pipeline/dags/{dagId}/trigger
    - Purpose: Trigger Run button.
    - Body (optional): { conf: object }.
    - Response: { runId }.
  - POST /api/v1/pipeline/dags/{dagId}/pause
    - Purpose: Pause/Resume toggle.
    - Body: { paused: true|false }.
  - POST /api/v1/pipeline/dags/{dagId}/stopRun
    - Purpose: Stop Active Run button.
    - Body: { runId }.
  - GET /api/v1/pipeline/dags/{dagId}/runs
    - Purpose: Run History table and filters.
    - Query params: state (Running|Success|Failed|Queued, optional), from, to, searchRunId, page, pageSize.
    - Response: { data: [ { runId, start, end, durationSeconds, triggeredBy, state } ], meta: { page, pageSize, total } }.
  - GET /api/v1/pipeline/runs/{runId}
    - Purpose: Current/Last Run cards and Run Details summary.
    - Response: { runId, dagId, conf, state, start, end, durationSeconds }.
  - GET /api/v1/pipeline/runs/{runId}/graph
    - Purpose: Graph sub-tab.
    - Response: { nodes: [ { id, label, state } ], edges: [ { from, to } ] }.
  - GET /api/v1/pipeline/runs/{runId}/gantt
    - Purpose: Gantt sub-tab.
    - Response: { tasks: [ { taskId, label, start, end, state } ] }.
  - GET /api/v1/pipeline/runs/{runId}/logs
    - Purpose: Logs sub-tab.
    - Query params: cursor (optional).
    - Response: { entries: [ { timestamp, level, message } ], nextCursor? }.
  - PATCH /api/v1/pipeline/dags/{dagId}/settings
    - Purpose: Edit DAG tab (schedule, timezone, catchup, max active runs, default args).
    - Body: { scheduleCron, timezone, catchup, maxActiveRuns, defaultArgs: { retries, retryDelayMinutes, owner, tags: [string] } }.
    - Response: updated DAG object.

- Data Scientist – Models:
  - GET /api/v1/models
    - Purpose: populate models overview table with performance metrics and predictions.
    - Response: [ { ticker, lastTrained, mape: { "7d": mapePct, "15d": mapePct, "30d": mapePct }, predictions: { "7d": pctChange, "15d": pctChange, "30d": pctChange }, plotUrl } ].
    - Notes: 
      - `lastTrained`: ISO timestamp of last training run
      - `mape`: MAPE percentages for each horizon (color-coded: green < 5%, yellow 5-10%, red > 10%)
      - `predictions`: predicted percentage changes for each horizon (positive/negative with arrows)
      - `plotUrl`: S3 URL to evaluation plot image (e.g., `https://s3.amazonaws.com/bucket/FPT/FPT_evaluation.png`)

## 6. Data Model (Database Schema & ER)

### 6.1 Database Schema (overview)

- Users & auth:
  - `users`: application users with role-based access (`end_user`, `data_scientist`, `admin`), username, password hash, display name, timestamps.
- Market data & predictions:
  - `stocks`: master data for VN30 tickers (symbol, name, sector, exchange, description, links, is_active).
  - `stock_prices`: daily OHLCV data per stock with unique `(stock_id, price_date)` and indexes for fast range queries and % change calculations.
  - `stock_prediction_summaries`: per `(stock, as_of_date, horizon_days)` predicted % change used for Top Picks / Should Buy / Should Sell and Home table; optionally linked to the training run that produced it.
  - `stock_prediction_points`: per-day forecast prices per `(stock, horizon_days)` for the Price & Forecast chart overlay.
  - `model_statuses`: latest model state per stock (fresh/stable/stale, last updated, reference to training run).
  - `model_horizon_metrics`: per-model, per-horizon metrics (e.g., MAPE) backing the Model Status card.
- Training configs & experiments:
  - `training_configs`: versioned configuration blobs (JSON) for stock universe, data window, indicators, targets, models, ensemble, and reproducibility, owned by a data scientist.
  - `experiment_runs`: individual training runs referencing a config and triggered user, with state (`pending`, `running`, `success`, `failed`, `cancelled`), progress, ETA, timestamps, notes.
  - `experiment_logs`: ordered log entries per run (timestamp, level, message) used for the Log Tail and detailed debugging; for very large volumes, logs can alternatively be streamed to an external log store (e.g., ELK) instead of PostgreSQL.
  - `experiment_ticker_artifacts`: per-run, per-ticker artifact records with metric JSON and URLs to evaluation/future plots, model/scaler binaries, and future_predictions CSVs.
- Pipelines (Airflow control):
  - `pipeline_dags`: metadata and editable settings for DAGs (`dag_id`, name, description, schedule CRON, timezone, catchup, max_active_runs, default retries/owner/tags, status).
  - `pipeline_runs`: individual DAG run records (string `run_id`, `dag_id`, state, start/end/duration, triggered_by, conf JSON) powering Overview, Run History, and Run Details cards.
  - `pipeline_run_tasks`: per-run task nodes (task_id, label, start/end, state) backing Graph and Gantt views.
  - `pipeline_run_logs`: optional detailed log entries per run for the Logs sub-tab; for high-volume deployments this table can be replaced or supplemented by a document/log store.

> See the database design section in the technical documentation for full PostgreSQL `CREATE TABLE` statements and index definitions corresponding to these logical entities.

### 6.2 PostgreSQL schema (reference)

```sql
-- Users & auth -------------------------------------------------------------

CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username        TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    display_name    TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('end_user', 'data_scientist', 'admin')),
    email           TEXT UNIQUE,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- Market data & stocks -----------------------------------------------------

CREATE TABLE stocks (
    id                      BIGSERIAL PRIMARY KEY,
    ticker                  TEXT NOT NULL UNIQUE,
    name                    TEXT NOT NULL,
    sector                  TEXT,
    exchange                TEXT,
    description             TEXT,
    logo_url                TEXT,
    financial_report_url    TEXT,
    company_website_url     TEXT,
    is_active               BOOLEAN NOT NULL DEFAULT TRUE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE stock_prices (
    id              BIGSERIAL PRIMARY KEY,
    stock_id        BIGINT NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    price_date      DATE NOT NULL,
    open_price      NUMERIC(18,4),
    high_price      NUMERIC(18,4),
    low_price       NUMERIC(18,4),
    close_price     NUMERIC(18,4) NOT NULL,
    volume          BIGINT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (stock_id, price_date)
);

CREATE INDEX idx_stock_prices_stock_date
    ON stock_prices (stock_id, price_date DESC);

CREATE TABLE stock_prediction_summaries (
    id                      BIGSERIAL PRIMARY KEY,
    stock_id                BIGINT NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    as_of_date              DATE NOT NULL,
    horizon_days            INTEGER NOT NULL,
    predicted_change_pct    NUMERIC(7,4) NOT NULL,
    experiment_run_id       UUID REFERENCES experiment_runs(id),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (stock_id, as_of_date, horizon_days)
);

CREATE INDEX idx_pred_summaries_horizon_date
    ON stock_prediction_summaries (horizon_days, as_of_date DESC);

CREATE TABLE stock_prediction_points (
    id                  BIGSERIAL PRIMARY KEY,
    stock_id            BIGINT NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    experiment_run_id   UUID REFERENCES experiment_runs(id),
    horizon_days        INTEGER NOT NULL,
    prediction_date     DATE NOT NULL,
    predicted_price     NUMERIC(18,4) NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (stock_id, experiment_run_id, horizon_days, prediction_date)
);

CREATE TABLE model_statuses (
    id                  BIGSERIAL PRIMARY KEY,
    stock_id            BIGINT NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    experiment_run_id   UUID REFERENCES experiment_runs(id),
    freshness_state     TEXT NOT NULL CHECK (freshness_state IN ('fresh', 'stable', 'stale')),
    last_updated_at     TIMESTAMPTZ NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX ux_model_status_latest
    ON model_statuses (stock_id, last_updated_at DESC);

CREATE TABLE model_horizon_metrics (
    id              BIGSERIAL PRIMARY KEY,
    model_status_id BIGINT NOT NULL REFERENCES model_statuses(id) ON DELETE CASCADE,
    horizon_days    INTEGER NOT NULL,
    mape_pct        NUMERIC(6,3) NOT NULL,
    UNIQUE (model_status_id, horizon_days)
);

-- Training configs & experiments ------------------------------------------

CREATE TABLE training_configs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_user_id   UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    description     TEXT,
    config          JSONB NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (owner_user_id, name, version)
);

CREATE INDEX idx_training_configs_owner_active
    ON training_configs (owner_user_id, is_active);

CREATE TABLE experiment_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_id       UUID NOT NULL REFERENCES training_configs(id) ON DELETE RESTRICT,
    owner_user_id   UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    scope           TEXT,
    state           TEXT NOT NULL CHECK (state IN ('pending', 'running', 'success', 'failed', 'cancelled')),
    progress_pct    NUMERIC(5,2),
    eta             TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    notes           TEXT,
    summary_metrics JSONB,
    CONSTRAINT chk_experiment_times
        CHECK (finished_at IS NULL OR finished_at >= created_at)
);

CREATE INDEX idx_experiment_runs_state_created
    ON experiment_runs (state, created_at DESC);

CREATE TABLE experiment_logs (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    level           TEXT NOT NULL,
    message         TEXT NOT NULL
);

CREATE INDEX idx_experiment_logs_run_ts
    ON experiment_logs (run_id, ts);

CREATE TABLE experiment_ticker_artifacts (
    id                      BIGSERIAL PRIMARY KEY,
    run_id                  UUID NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
    stock_id                BIGINT NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    metrics                 JSONB,
    evaluation_png_url      TEXT,
    future_png_url          TEXT,
    model_pkl_url           TEXT,
    scaler_pkl_url          TEXT,
    future_predictions_csv  TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, stock_id)
);

-- Pipelines (Airflow control) ---------------------------------------------

CREATE TABLE pipeline_dags (
    dag_id              TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    description         TEXT,
    status              TEXT NOT NULL CHECK (status IN ('active', 'paused')),
    schedule_cron       TEXT NOT NULL,
    schedule_label      TEXT,
    timezone            TEXT NOT NULL DEFAULT 'Asia/Ho_Chi_Minh',
    catchup             BOOLEAN NOT NULL DEFAULT FALSE,
    max_active_runs     INTEGER NOT NULL DEFAULT 1,
    default_retries     INTEGER NOT NULL DEFAULT 0,
    default_retry_delay_minutes INTEGER NOT NULL DEFAULT 5,
    default_owner       TEXT,
    default_tags        TEXT[] DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE pipeline_runs (
    run_id          TEXT PRIMARY KEY,
    dag_id          TEXT NOT NULL REFERENCES pipeline_dags(dag_id) ON DELETE CASCADE,
    conf            JSONB,
    state           TEXT NOT NULL CHECK (state IN ('running', 'success', 'failed', 'queued', 'up_for_retry', 'cancelled')),
    start_time      TIMESTAMPTZ,
    end_time        TIMESTAMPTZ,
    duration_seconds INTEGER,
    triggered_by_label TEXT NOT NULL,
    triggered_by_user_id UUID REFERENCES users(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_pipeline_runs_dag_state_start
    ON pipeline_runs (dag_id, state, start_time DESC);

CREATE TABLE pipeline_run_tasks (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    task_id         TEXT NOT NULL,
    label           TEXT NOT NULL,
    state           TEXT NOT NULL,
    start_time      TIMESTAMPTZ,
    end_time        TIMESTAMPTZ,
    UNIQUE (run_id, task_id)
);

CREATE TABLE pipeline_run_logs (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    level       TEXT NOT NULL,
    message     TEXT NOT NULL
);

CREATE INDEX idx_pipeline_run_logs_run_ts
    ON pipeline_run_logs (run_id, ts);
```

### 6.3 ER relationships (summary)

- `users` → `training_configs`, `experiment_runs`, `pipeline_runs`.
- `stocks` → `stock_prices`, `stock_prediction_summaries`, `stock_prediction_points`, `model_statuses`, `experiment_ticker_artifacts`.
- `model_statuses` → `model_horizon_metrics`.
- `training_configs` → `experiment_runs` → `experiment_logs` and `experiment_ticker_artifacts`.
- `experiment_runs` optionally link back into `stock_prediction_summaries`, `stock_prediction_points`, and `model_statuses` via their `experiment_run_id` fields.
- `pipeline_dags` → `pipeline_runs` → `pipeline_run_tasks` and `pipeline_run_logs`.

## 7. System & Backend Architecture

### 7.1 System Architecture Overview

### Major components

- **Frontend Web App**
  - SPA that implements all UI screens in this spec (Login, Home, Stock Detail, Training, Pipelines).
  - Talks only to the backend API, handles client-side routing, role-based nav, state management, and loading/empty/error states.
- **Backend API Service**
  - Python web service (e.g., FastAPI/Flask) exposing all `/api/v1/**` endpoints.
  - Implements authentication/authorization, validation, business logic, and interaction with Postgres, Airflow, cache, and object storage.
- **Training Worker**
  - Background worker consuming training jobs created by `/experiments/run`.
  - Runs the ML pipelines (feature generation, model training, prediction), writes artifacts to object storage, and metadata/predictions to Postgres.
- **Airflow (Pipelines Orchestrator)**
  - Owns execution of `vn30_data_crawler` and `vn30_model_training` DAGs.
  - Exposes status and history via Airflow’s REST API or metadata DB; the backend wraps this as `/api/v1/pipeline/**`.
- **PostgreSQL Database**
  - System of record for users, roles, configs, experiments, predictions metadata, and mirrored pipeline metadata as defined above.

### Supporting services / infrastructure

- **Authentication & RBAC**
  - Implemented inside the backend using `users`. 
  - Middleware enforces roles (`end_user`, `data_scientist`, `admin`) on protected endpoints (Training, Pipelines, experiments).
- **Object Storage for Artifacts**
  - S3-compatible bucket (e.g., AWS S3 or Cloudflare R2) for model artifacts and images (`evaluation.png`, `future.png`, `model.pkl`, `scaler.pkl`, `future_predictions.csv`).
  - URLs stored in `experiment_ticker_artifacts`.
- **Message Queue**
  - Redis/RabbitMQ/Kafka queue for training jobs; prevents long-running work from blocking HTTP requests.
- **Cache** (Optional)
  - Redis cache for read-heavy endpoints: top picks, market tables, stock detail charts, DAG catalogs.
- **API Gateway / Reverse Proxy** (Optional)
  - Nginx/Envoy/Traefik in front of backend and static frontend for TLS termination, routing, and basic rate limiting.
- **Logging & Monitoring** (Optional)
  - Centralized logging (ELK/Loki) aggregating backend/worker/Airflow logs plus DB-backed logs (`experiment_logs`, `pipeline_run_logs`).
  - Metrics/alerting via Prometheus/Grafana or cloud equivalents (request latency, error rates, DAG failures, experiment failures).
- **CI/CD & Containerization** (Optional)
  - CI pipeline that runs tests, builds Docker images for frontend/backend/worker, and deploys to staging/production.
  - Runtime via Docker Compose (small) or Kubernetes (production), using managed Postgres and object storage where possible.

### Component interactions & data flow (summary)

- **End User – Home / Stock Detail**
  - Frontend → Backend (`/stocks/top-picks`, `/stocks/market-table`, `/stocks/{ticker}`, `/stocks/{ticker}/predictions`, `/stocks/{ticker}/chart`, `/models/{ticker}/status`).
  - Backend → Cache → Postgres (and occasionally object storage) to serve lists, tables, and charts.
- **Data Scientist – Training**
  - Frontend → Backend (`/features/config`, `/features/validate`, `/features/config` save, `/experiments/run`, `/experiments/{runId}`, `/experiments/{runId}/logs/tail`, `/experiments/{runId}/artifacts`).
  - Backend writes configs and runs to Postgres and pushes jobs to the queue.
  - Training worker consumes jobs, reads data from Postgres, writes predictions/metrics/artifacts back, and streams logs.
- **Data Scientist – Pipelines**
  - Frontend → Backend (`/pipeline/dags`, `/pipeline/dags/{dagId}`, `/pipeline/dags/{dagId}/runs`, `/pipeline/runs/{runId}`, `/pipeline/dags/{dagId}/trigger`, `/pause`, `/stopRun`, `/settings`).
  - Backend proxies to Airflow (REST or DB) and/or keeps a synchronized mirror of metadata in `pipeline_*` tables.

### 7.2 Python Backend Monorepo Architecture

### Components inside the monorepo

- **REST API service (`/api/v1/**`)**
  - FastAPI (or Flask) app exposing all endpoints defined in *API Mapping (per UI page)*.
  - Uses SQLAlchemy models that mirror the PostgreSQL schema defined in this file.
  - Implements authentication (JWT/opaque tokens), role checks, request/response validation, error handling, and OpenAPI docs.
- **Training worker(s)**
  - Python processes (e.g., Celery/RQ workers) that consume training jobs from a queue.
  - Reuse the same models/config code as the API (imported from shared packages in the monorepo).
  - Responsible for executing experiments, writing metrics/predictions/artifacts, and streaming logs into `experiment_*` tables and object storage.
- **Airflow integration library**
  - Thin client that wraps Airflow’s REST API or metadata DB and exposes high-level functions used by the API service:
    - `list_dags()`, `get_dag(dag_id)`, `trigger_dag_run(dag_id, conf)`, `pause_dag(dag_id, paused)`, `stop_run(dag_id, run_id)`, `list_runs(dag_id, filters)`, `get_run_graph(run_id)`, `get_run_logs(run_id)`, `update_dag_settings(...)`.
  - Keeps Airflow-specific details (URLs, auth, schemas) localized to one module.
- **Database access layer**
  - SQLAlchemy `models.py` / `schemas.py` that directly correspond to the `CREATE TABLE` statements in this spec.
  - Session management module (e.g., `db/session.py`) that provides `get_db()` dependency for FastAPI routes and workers.
  - Alembic (or equivalent) migration environment kept in the repo and generated from the SQLAlchemy models.
- **Infrastructure utilities**
  - `config`: typed settings loaded from environment variables / `.env` (database DSN, Redis URL, S3 bucket, Airflow base URL, JWT secret, etc.).
  - `logging`: centralized logging configuration (JSON logs, correlation IDs, request IDs).
  - `security`: password hashing, token generation/verification, role decorators, CORS settings.
  - `storage`: S3 client wrapper to generate signed URLs and manage artifact uploads.
  - `tasks`: shared job definitions for training runs and (optionally) maintenance jobs (cleanup, re-indexing, etc.).

### Recommended directory structure (backend repo)

At the repository root (this Python monorepo):

```text
backend/
  pyproject.toml / setup.cfg / requirements.txt
  .env.example
  alembic.ini
  src/
    app/                    # REST API service
      main.py               # FastAPI/Flask entrypoint
      api/
        v1/
          auth.py
          stocks.py
          predictions.py
          training.py
          experiments.py
          pipelines.py
      core/
        config.py
        logging.py
        security.py
        errors.py
      db/
        base.py             # SQLAlchemy Base
        session.py
        models/             # One module per area (users, stocks, training, pipelines)
      schemas/              # Pydantic DTOs for requests/responses
      services/             # Business logic (prediction queries, validation, etc.)
      integrations/
        airflow_client.py
        storage_s3.py
        queue.py
    worker/
      main.py               # Background worker entrypoint (Celery/RQ)
      tasks/
        experiments.py      # train_model_run(run_id: UUID)
    migrations/
      versions/             # Alembic migration scripts
  tests/
    api/
    services/
    integrations/
  docker/
    docker-compose.dev.yml
    Dockerfile.api
    Dockerfile.worker
```

> The existing `dags/` directory for Airflow DAGs and any legacy `modules/` can either be moved under `backend/` (e.g., `backend/airflow/dags/`) or gradually refactored into `app/services` and `worker/tasks` while keeping DAG files thin.

### Best practices for code organization & configuration

- **Single source of truth for domain models**
  - Keep SQLAlchemy models and Pydantic schemas in shared modules used by both API and workers.
  - Ensure the Alembic migration environment imports the same models so the DB and code stay in sync.
- **Environment management**
  - Use `pyenv` + `venv` or Poetry; document standard commands in `README.md`.
  - Load configuration via `pydantic-settings` or similar so `config.py` reads from environment variables with sane defaults and `.env` for local development.
- **Configuration separation**
  - Have explicit profiles (dev/stage/prod) controlled by `APP_ENV`, affecting debug flags, CORS, logging verbosity, and external URLs (Postgres, Redis, S3, Airflow).
- **Error handling**
  - Central exception handlers that map domain errors (validation, not-found, permission denied) into structured JSON responses with error codes used by the frontend for user-friendly messages.
  - Log unexpected exceptions with correlation IDs to simplify debugging across services.
- **Testing**
  - Unit tests for services and integrations, plus API tests using a test database and fixtures.
  - Optional snapshot tests to ensure API responses remain compatible with the React frontend.

### Local development & workflows (backend)

- **Prerequisites**
  - Python 3.11 (via pyenv), Docker (for Postgres, Redis, MinIO/S3, Airflow), and a virtual environment.
- **Setup steps**
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt` (or `poetry install` if using Poetry).
  - Copy `.env.example` to `.env` and fill in local DSNs (Postgres, Redis, S3, Airflow URL, JWT secret).
  - Start local infra via `docker-compose -f docker/docker-compose.dev.yml up -d` (Postgres, Redis, MinIO, Airflow).
  - Apply DB migrations: `alembic upgrade head`.
- **Run services**
  - API: `uvicorn app.main:app --reload` (or `python -m app.main`).
  - Worker: `celery -A worker.main worker -l info` (or equivalent for chosen task queue).
  - Airflow: managed by `docker-compose` or a separate deployment; ensure Airflow base URL and credentials match `config.py`.
- **Pipelines**
  - Define VN30 DAGs in `dags/` (inside Airflow deployment) that call the training worker or backend as needed.
  - Use the `/api/v1/pipeline/**` endpoints to control runs from the UI; backend maps them to Airflow.

### Connecting the ReactJS frontend (separate repo)

- Frontend should:
  - Use a single configurable `API_BASE_URL` (e.g., `http://localhost:8000/api/v1`) and call the endpoints defined in this spec.
  - Store the access token in HTTP-only cookies or memory and send it via `Authorization: Bearer <token>`.
  - Handle CORS; backend should allow the frontend origin(s) in CORS settings for dev and production.
- For local development:
  - Run backend at `http://localhost:8000` and frontend dev server at `http://localhost:3000`.
  - Configure CORS and proxy settings (e.g., CRA/Vite dev proxy) to avoid mixed-origin headaches.

### 7.3 Design choices and options

- **Architecture style**
  - Recommended initial approach: single backend monolith + separate training worker + Airflow deployment.
  - Microservices (separate services for auth, predictions, training, pipelines) are possible later if team or load grows.
- **Logs and metrics**
  - Current schema stores logs in Postgres for experiments and pipelines; at higher scale, detailed logs can move to an external log store with only pointers kept in DB.
- **Deployment targets**
  - Cloud (managed Postgres, S3, Kubernetes, managed Airflow) is preferred for reliability and scaling; self-hosted is possible but increases ops burden.

## 8. Open Questions

- **Predictions & ensembles**
  - Do we need confidence intervals and/or prediction intervals surfaced in the UI?
  - How should ensemble weights be determined (manual vs learned via CV per horizon) and versioned?
- **Model lifecycle**
  - Exact rules for moving models between `fresh` / `stable` / `stale` states.
  - Retention policy for old models, experiments, and predictions (per ticker/per horizon).
- **Training & validation**
  - Detailed criteria for `features/validate` blockers vs warnings and how to estimate `estRuntimeMinutes` and optional `estCost`.
- **Pipelines & permissions**
  - Granularity of access control for DAG triggers/edits (per-user, per-role, or per-team).
- **Multi-user semantics**
  - Are training configs and runs private to a user, shared within a group, or global across all data scientists?

## 9. Deliverables

- Complete, working implementation of:
  - Backend API and training worker consistent with this spec and database schema.
  - Airflow DAGs for `vn30_data_crawler` and `vn30_model_training` wired to the backend/worker.
  - React frontend implementing all screens and behaviors described here (Login, Home, Stock Detail, Training, Pipelines, Models).
- Tests and tooling:
  - Automated tests (unit + integration) for critical backend paths and data pipelines.
  - CI pipeline that builds, tests, and deploys backend + worker containers.

## 10. Change Log

- **v1.4** - Nov 30, 2025
  - Added Models page specification (section 2.7) for Data Scientists.
  - Added GET /api/v1/models endpoint to API specification.
  - Updated navigation to include Models page for Data Scientists.
  - Models page displays simple table with MAPE metrics, predictions, and evaluation plots from S3.
- **v1.3**
  - Reorganized `SPECS.md` with table of contents, clearer sectioning (overview, functional, API, data model, architecture, non-functional, open questions, deliverables, change log).
  - Added explicit backend monorepo architecture and PostgreSQL schema reference.
- **v1.2**
  - Documented detailed API mapping per UI page and database schema/ER model.
- **v1.1**
  - Refined UI specs for End User Home, Stock Detail, Training, and Pipelines screens.
- **v1.0**
  - Initial draft of app purpose, roles, navigation, and core screens.