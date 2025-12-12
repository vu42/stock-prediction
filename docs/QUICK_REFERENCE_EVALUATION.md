# Quick Reference: Generate Evaluation Results

**One-page guide to generate report section 6.7**

---

## ⚡ Quick Start (3 Commands)

```bash
# 1. Seed model metrics (one time only)
docker exec stock-prediction-api python -m scripts.seed_mock_model_metrics

# 2. Generate evaluation results
docker exec stock-prediction-api python -m scripts.generate_evaluation_results

# 3. View the output
cat output/section_6_7_evaluation.md
```

---

## 📋 Prerequisites Checklist

- [ ] Docker and Docker Compose installed
- [ ] Services running: `docker ps --filter "name=stock-prediction"`
- [ ] Database seeded with stocks

**If services not running:**
```bash
cd backend
docker-compose -f docker/docker-compose.dev.yml up -d
docker exec stock-prediction-api alembic upgrade head
docker exec stock-prediction-api python -m scripts.seed_stocks
```

---

## 📁 Output Files

| File | Location | Purpose |
|------|----------|---------|
| `section_6_7_evaluation.md` | `output/` | Complete markdown section for report |
| `mape_metrics.csv` | `output/` | CSV data for teaching staff |

---

## 🔧 Common Issues

| Problem | Solution |
|---------|----------|
| "No MAPE data found" | Run: `docker exec stock-prediction-api python -m scripts.seed_mock_model_metrics` |
| "Container not found" | Start services: `cd backend && docker-compose -f docker/docker-compose.dev.yml up -d` |
| "Permission denied" | Run from project root directory |

---

## 📞 Need Help?

**Zalo:** `0123-456-789` *(Update with your number)*

**Include when contacting:**
- Error message
- Command you ran
- Output from: `docker ps --filter "name=stock-prediction"`

---

## 📊 Understanding MAPE

- **< 5%** = Excellent (Green)
- **5-10%** = Acceptable (Yellow)  
- **> 10%** = Needs improvement (Red)

Lower is better!

---

**Full guide:** See [HOW_TO_GENERATE_EVALUATION_RESULTS.md](HOW_TO_GENERATE_EVALUATION_RESULTS.md)
