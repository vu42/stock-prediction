# Models Page Specification (MVP - 1 Week)

## Overview

A **simple, minimal models page** to display trained model performance for the assignment. Focus on **essential information only** with no complex features.

---

## Scope

### ✅ In Scope (Must Have)
- Simple table showing all trained models
- Basic metrics (MAPE for 7D/15D/30D)
- Last trained date
- Current predictions (% change)
- Click to view evaluation plot image

### ❌ Out of Scope (Too Complex)
- ~~Search and filtering~~
- ~~Sorting~~
- ~~Model configuration details~~
- ~~Artifact downloads~~
- ~~Training history~~
- ~~Retrain button~~
- ~~Role-based permissions~~

---

## Page Layout

**One simple table - that's it!**

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Models Overview                              │
│                 Performance metrics for trained models                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Ticker │ Last Trained │    MAPE (%)       │  Predictions (%)        │
│         │              │  7D │ 15D │ 30D   │  7D │ 15D │ 30D  │View│
│  ───────┼──────────────┼─────┼─────┼───────┼─────┼─────┼──────┼────│
│  FPT    │ 2 days ago   │ 3.2 │ 4.1 │ 5.8   │+4.2 │+6.8 │+9.1  │[📊]│
│  VCB    │ 2 days ago   │ 2.9 │ 3.8 │ 4.9   │+2.5 │+3.8 │+5.2  │[📊]│
│  HPG    │ 5 days ago   │ 4.5 │ 5.2 │ 6.1   │-1.2 │-0.8 │+0.5  │[📊]│
│  GAS    │ 1 day ago    │ 3.8 │ 4.5 │ 6.0   │+1.8 │+2.5 │+4.1  │[📊]│
│  ...    │ ...          │ ... │ ... │ ...   │ ... │ ... │ ...  │[📊]│
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Models Table

**Columns (8 total):**
1. **Ticker** - Stock symbol (FPT, VCB, etc.)
2. **Last Trained** - How long ago (e.g., "2 days ago")
3. **MAPE 7D** - Accuracy % (color-coded)
4. **MAPE 15D** - Accuracy % (color-coded)
5. **MAPE 30D** - Accuracy % (color-coded)
6. **Pred 7D** - Predicted % change (with arrow)
7. **Pred 15D** - Predicted % change (with arrow)
8. **Pred 30D** - Predicted % change (with arrow)
9. **View** - Button to see plot

**Color Coding:**
- **MAPE**: Green (< 5%), Yellow (5-10%), Red (> 10%)
- **Predictions**: Green (positive ↑), Red (negative ↓)

### 2. Plot Modal

When user clicks **[📊]** button, show a simple modal:

```
┌─────────────────────────────────────────────────┐
│ FPT - Model Evaluation                     [✕] │
├─────────────────────────────────────────────────┤
│                                                  │
│         [Display FPT_evaluation.png]            │
│                                                  │
│    Shows actual vs predicted prices              │
│    with metrics overlay                          │
│                                                  │
│                [Close]                           │
└─────────────────────────────────────────────────┘
```

---

## Data Source

**Training outputs are stored in S3** - After training, all artifacts are uploaded to S3 bucket.

Your training script creates these files and uploads them to S3:
- `{ticker}_evaluation.png` - Evaluation plot
- `{ticker}_future_predictions.csv` - Predictions data
- `{ticker}_multi_horizon_model.pkl` - Trained model (optional)
- Training metadata (timestamp, metrics, etc.)

**Backend endpoint:**

```python
GET /api/models

Response:
[
  {
    "ticker": "FPT",
    "last_trained": "2024-11-28T10:30:00Z",
    "mape": {
      "7d": 3.2,
      "15d": 4.1,
      "30d": 5.8
    },
    "predictions": {
      "7d": 4.2,
      "15d": 6.8,
      "30d": 9.1
    },
    "plot_url": "https://s3.amazonaws.com/your-bucket/FPT/FPT_evaluation.png"
  },
  // ... repeat for all VN30 stocks
]
```

**Implementation approach:**
1. Store model metadata in database (ticker, last_trained, mape, predictions)
2. Store artifact URLs (S3 paths for plots, CSVs, etc.)
3. Backend queries database and returns JSON with S3 URLs
4. Frontend displays images directly from S3 (public URLs or signed URLs)

---

## Visual Design

**Keep it simple and clean:**

### Colors
- **MAPE** (accuracy):
  - Green (#10b981): < 5% (excellent)
  - Yellow (#f59e0b): 5-10% (acceptable)
  - Red (#ef4444): > 10% (needs improvement)

- **Predictions**:
  - Green (#10b981): Positive change ↑
  - Red (#ef4444): Negative change ↓

### Table Style
- Header: Dark gray (#333) background, white text
- Rows: Alternate white / light gray (#f5f5f5)
- Hover: Slightly darker (#e8e8e8)
- Border: 1px solid #ddd

### Button
- Blue background (#3b82f6)
- White text
- Rounded (4px)
- Hover: Darker blue (#2563eb)

---

## Implementation Steps

### Frontend (React/Vue/etc.)

1. **Create `/models` page**
2. **Fetch data on load:**
   ```javascript
   fetch('/api/models')
     .then(res => res.json())
     .then(data => setModels(data))
   ```
3. **Render table** with data
4. **Add color logic** for MAPE and predictions
5. **Modal component** for plot display
6. **Click handler** for [View] button

### Backend (Python/FastAPI/Flask)

1. **Create endpoint** `GET /api/models`
2. **Query database** for model metadata (ticker, last_trained, mape, predictions)
3. **Include S3 URLs** for artifacts (plot images)
4. **Return** JSON array with all data

---

## Acceptance Criteria

**Must have for MVP:**

✅ **Table Display**
- [ ] Shows all VN30 stocks with trained models
- [ ] Last trained date displays (relative time like "2 days ago")
- [ ] MAPE values shown for 7D, 15D, 30D
- [ ] Predictions shown for 7D, 15D, 30D
- [ ] MAPE color-coded (green/yellow/red)
- [ ] Predictions have arrows (↑↓) and colors
- [ ] [View] button in each row

✅ **Plot Modal**
- [ ] Clicking [View] opens modal
- [ ] Modal displays evaluation plot image
- [ ] Close button (X) works
- [ ] Clicking outside modal closes it
- [ ] ESC key closes modal

✅ **Data Loading**
- [ ] Backend queries database for model metadata
- [ ] Returns JSON with S3 URLs to frontend
- [ ] Frontend renders correctly
- [ ] Images load from S3

✅ **Responsive**
- [ ] Table scrolls horizontally on mobile
- [ ] Modal is centered and responsive

**That's all!** No complex features, just the essentials.

