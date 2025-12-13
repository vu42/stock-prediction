# How to Generate Evaluation Results for Report Section 6.7

This guide will walk you through generating the evaluation results tables for your report using the `generate_evaluation_results.py` script.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step-by-Step Instructions](#step-by-step-instructions)
3. [Understanding the Output](#understanding-the-output)
4. [Troubleshooting](#troubleshooting)
5. [Contact Support](#contact-support)

---

## Quick Visual Guide

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Setup Application                                 │
│  👉 Follow README.md Quick Start section                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Seed Model Metrics                                │
│  $ docker exec stock-prediction-api python -m \             │
│    scripts.seed_mock_model_metrics                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Generate Evaluation Results                       │
│  $ docker exec stock-prediction-api python -m \             │
│    scripts.generate_evaluation_results                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Access Output Files                               │
│  📄 output/section_6_7_evaluation.md                        │
│  📊 output/mape_metrics.csv                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before running the evaluation results script, you need to have the application running.

👉 **Follow the [Quick Start](../README.md#quick-start) section in the main README.md to set up the application.**

### Verify Services are Running

After following the Quick Start guide, verify all Docker containers are running:

```bash
docker ps --filter "name=stock-prediction"
```

You should see these containers with status "Up":
- `stock-prediction-api`
- `stock-prediction-worker`
- `stock-prediction-postgres`
- `stock-prediction-redis`
- `stock-prediction-minio`
- `stock-prediction-airflow`

---

## Step-by-Step Instructions

### Step 1: Seed Model Metrics (Required)

The evaluation results script needs model performance metrics in the database. If you haven't trained real models yet, seed mock metrics:

```bash
docker exec stock-prediction-api python -m scripts.seed_mock_model_metrics
```

**Expected output:**
```
📊 Seeding mock model metrics for 5 stocks...
   Symbols: FPT, VCB, VNM, HPG, VIC
✓ FPT: Created new model status
   7D MAPE: 2.44%
   15D MAPE: 6.84%
   30D MAPE: 6.71%
...
✅ Seeding complete!
```

### Step 2: Run the Evaluation Results Script

From the **project root directory**, run:

```bash
docker exec stock-prediction-api python -m scripts.generate_evaluation_results
```

**Expected output:**
```
Using SQLAlchemy ORM mode
Connecting to database...
Querying MAPE metrics...
======================================================================
EVALUATION RESULTS SUMMARY
======================================================================

Tickers with data: 5
Tickers: FPT, HPG, VHM, VIC, VNM

Aggregate Statistics by Horizon:
--------------------------------------------------
   7D horizon: Mean=4.12%, Min=2.44%, Max=4.95%
  15D horizon: Mean=5.21%, Min=3.62%, Max=6.84%
  30D horizon: Mean=7.63%, Min=5.66%, Max=11.24%

Per-Ticker MAPE Values:
--------------------------------------------------
Ticker           7D        15D        30D
--------------------------------------------------
FPT            2.44       6.84       6.71
HPG            4.95       6.54      11.24
VHM            4.36       3.64       5.66
VIC            4.02       3.62       6.39
VNM            4.80       5.39       8.14

======================================================================
REPORT SECTION 6.7 (copy below this line)
======================================================================

### 6.7 Evaluation results
...
✅ Report section saved to: /app/output/section_6_7_evaluation.md
✅ CSV export saved to: /app/output/mape_metrics.csv
```

### Step 3: Access the Generated Files

The script generates two files that are accessible via the mounted Docker volume:

```bash
# From the project root directory
ls -lh output/

# You should see:
# section_6_7_evaluation.md  - Complete markdown section for the report
# mape_metrics.csv            - CSV export of all MAPE values
```

### Step 4: Copy to Your Report

**Option A: View the markdown file**
```bash
cat output/section_6_7_evaluation.md
```

**Option B: Copy to your report**
```bash
# The content is already in docs/REPORT.md, but if you need to update it:
# Copy the content from output/section_6_7_evaluation.md
# and paste it into section 6.7 of your report
```

**Option C: View the CSV data**
```bash
cat output/mape_metrics.csv
```

---

## Understanding the Output

### Output Files

#### 1. `section_6_7_evaluation.md`

This file contains the complete markdown section ready to be inserted into your report. It includes:

- **Table 6.1**: Aggregate MAPE statistics across all tickers
  - Mean MAPE for each horizon (7D, 15D, 30D)
  - Minimum and Maximum MAPE values
  
- **Table 6.2**: Example MAPE values for FPT ticker
  - Shows per-ticker breakdown

- **Commentary**: Explanatory text about the results

#### 2. `mape_metrics.csv`

A CSV file containing all MAPE values for all tickers:

```csv
Ticker,MAPE_7D,MAPE_15D,MAPE_30D
FPT,2.44,6.84,6.71
HPG,4.95,6.54,11.24
VHM,4.36,3.64,5.66
VIC,4.02,3.62,6.39
VNM,4.80,5.39,8.14
```

This can be shared with teaching staff for detailed inspection.

### What is MAPE?

**MAPE (Mean Absolute Percentage Error)** is the main evaluation metric used in this system:

- **Lower values = Better predictions** (more accurate)
- Expressed as a percentage (%)
- Calculated separately for each prediction horizon (7, 15, 30 days)

**Interpretation:**
- **< 5%**: Excellent prediction accuracy (green in UI)
- **5-10%**: Acceptable accuracy (yellow in UI)
- **> 10%**: Needs improvement (red in UI)

---

## Troubleshooting

### Problem 1: "No MAPE data found in the database"

**Solution:** You need to seed model metrics first:

```bash
docker exec stock-prediction-api python -m scripts.seed_mock_model_metrics
```

### Problem 2: Docker container not found

**Error message:** `Error: No such container: stock-prediction-api`

**Solution:** Start the Docker services:

```bash
cd backend
docker-compose -f docker/docker-compose.dev.yml up -d
```

### Problem 3: Permission denied

**Error message:** `Permission denied`

**Solution:** Make sure you're running from the project root directory and Docker has proper permissions.

### Problem 4: Database connection error

**Error message:** `could not connect to server`

**Solution:** 
1. Check if PostgreSQL container is running:
   ```bash
   docker ps --filter "name=postgres"
   ```

2. Restart the database:
   ```bash
   docker restart stock-prediction-postgres
   ```

3. Wait a few seconds and try again

### Problem 5: Import errors or module not found

**Error message:** `ModuleNotFoundError: No module named 'app'`

**Solution:** This should not happen when running via Docker. If it does:

```bash
# Restart the API container
docker restart stock-prediction-api

# Try again
docker exec stock-prediction-api python -m scripts.generate_evaluation_results
```

---

## Alternative: Running Locally (Advanced)

If you prefer to run the script locally instead of via Docker:

### Prerequisites
- Python 3.11+
- PostgreSQL accessible on localhost:5432
- Required Python packages installed

### Steps

```bash
# From project root
python generate_evaluation_results.py
```

Output files will be saved to the `docs/` folder instead of `output/`.

---

## Contact Support

If you encounter any issues that are not covered in this guide, please contact:

**📱 Zalo Support:**
- **Phone:** `0123-456-789` *(Replace with your actual phone number)*
- **Name:** Support Team

**When contacting support, please provide:**
1. The exact error message you're seeing
2. The command you ran
3. Output from `docker ps --filter "name=stock-prediction"`
4. Screenshot of the error (if applicable)

**Response time:** Usually within 24 hours

---

## Quick Reference

### Common Commands

```bash
# Check if services are running
docker ps --filter "name=stock-prediction"

# Seed model metrics (mock data for testing)
docker exec stock-prediction-api python -m scripts.seed_mock_model_metrics

# Generate evaluation results
docker exec stock-prediction-api python -m scripts.generate_evaluation_results

# View generated files
ls -lh output/
cat output/section_6_7_evaluation.md
cat output/mape_metrics.csv

# View API logs (if troubleshooting)
docker logs stock-prediction-api -f
docker logs stock-prediction-airflow -f

# Trigger training DAG manually
docker exec stock-prediction-airflow airflow dags trigger vn30_model_training

# Check DAG status
docker exec stock-prediction-airflow airflow dags list
docker exec stock-prediction-airflow airflow dags list-import-errors

# Restart services after code changes
docker restart stock-prediction-api stock-prediction-airflow
```

### File Locations

| File | Location | Description |
|------|----------|-------------|
| Script | `generate_evaluation_results.py` | Main script in project root |
| Backend copy | `backend/scripts/generate_evaluation_results.py` | Copy for Docker execution |
| Output (Docker) | `output/section_6_7_evaluation.md` | Generated markdown section |
| Output (Docker) | `output/mape_metrics.csv` | CSV export of metrics |
| Documentation | `docs/HOW_TO_GENERATE_EVALUATION_RESULTS.md` | This guide |

---

## Summary

1. ✅ Setup the application following [README.md Quick Start](../README.md#quick-start)
2. ✅ Seed model metrics: `docker exec stock-prediction-api python -m scripts.seed_mock_model_metrics`
3. ✅ Generate results: `docker exec stock-prediction-api python -m scripts.generate_evaluation_results`
4. ✅ Access files in `output/` folder
5. ✅ Copy content to your report

**That's it! You now have the evaluation results for your report section 6.7.**
