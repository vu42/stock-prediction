# 🎬 DEMO WORKFLOW - Step by Step

## 📋 PREPARATION (One Time Only)

### 1. Start Database
```bash
./start_database.sh
```

**This installs PostgreSQL + creates tables. Run once only!**

---

## 🔌 CONNECT TO DATABASE

### Terminal (psql) - RECOMMENDED FOR DEMO ✅

**Quick connect:**
```bash
psql -U postgres -d stock_prediction
```

**Connection info:**
- Host: localhost
- Port: 5432
- Database: stock_prediction
- User: postgres
- Password: postgres

**One-liner queries (no need to enter psql):**
```bash
# Quick count
psql -U postgres -d stock_prediction -c "SELECT COUNT(*) FROM stock_prices;"

# Recent data
psql -U postgres -d stock_prediction -c "SELECT * FROM stock_prices ORDER BY created_at DESC LIMIT 5;"

# Group by stock
psql -U postgres -d stock_prediction -c "SELECT stock_symbol, COUNT(*) FROM stock_prices GROUP BY stock_symbol;"
```

### GUI Tools (Optional)

**TablePlus / DBeaver / pgAdmin:**
- Host: `localhost`
- Port: `5432`
- Database: `stock_prediction`
- User: `postgres`
- Password: `postgres`

---

## 🎯 DEMO FLOW

### 🔵 STREAMING DEMO (5-10 minutes)

#### Start:
```bash
./start_streaming_demo.sh
```

**What opens:**
- ✅ Kafka UI: http://localhost:8080
- ✅ Airflow UI: http://localhost:8081
- ✅ Database: Ready for queries

**Wait 1-2 minutes for data to accumulate**

#### Show Kafka UI:
1. Open http://localhost:8080
2. Navigate to Topics → `vn30-stock-prices`
3. Show messages flowing (30 ticks/sec)
4. **Check Partitions**: Should see 6 partitions with distributed data

#### Show Partitions (Terminal):
```bash
./check_kafka_partitions.sh
```

**Expected output:**
- 6 partitions (Partition 0-5)
- Messages distributed across partitions
- Each stock routed to specific partition (by hash)

#### Show Database (Prove transformation works!):
```bash
psql -U postgres -d stock_prediction
```

```sql
-- Count records (should increase)
SELECT COUNT(*) FROM stock_prices;

-- Show recent data
SELECT stock_symbol, date, 
       ROUND(close::numeric, 2) as close, 
       created_at
FROM stock_prices 
ORDER BY created_at DESC LIMIT 10;

-- Group by stock
SELECT stock_symbol, COUNT(*) 
FROM stock_prices 
GROUP BY stock_symbol;

-- Exit
\q
```

#### Stop:
```bash
./stop_streaming_demo.sh
```

#### Clean Database:
```bash
./cleanup_database.sh
```

---

### 🟢 BATCH DEMO (10-15 minutes)

#### Start:
```bash
./start_batch_demo.sh
```

**What opens:**
- ✅ Airflow UI: http://localhost:8080
- ✅ Database: Ready for queries

#### Trigger DAG in Airflow:
1. Open http://localhost:8080
2. Login: admin / admin
3. Browse → DAGs
4. Find `vn30_data_crawler` (in batch/ folder)
5. Click ▶️ Play button → Trigger DAG
6. Watch it run (~2-5 minutes)

#### Monitor Database:
```bash
psql -U postgres -d stock_prediction
```

```sql
-- Watch records increase (run every 10 seconds)
SELECT COUNT(*) FROM stock_prices;

-- Check by stock
SELECT stock_symbol, COUNT(*) 
FROM stock_prices 
GROUP BY stock_symbol 
ORDER BY stock_symbol;

-- View VCB historical data
SELECT date, ROUND(close::numeric, 2) as close 
FROM stock_prices 
WHERE stock_symbol = 'VCB' 
ORDER BY date DESC LIMIT 10;

-- Exit
\q
```

#### Show Output Files (Skip training, show pre-trained):
```bash
# List VCB output
ls -lh output/VCB/

# Show evaluation chart
open output/VCB/VCB_evaluation.png

# Show predictions
cat output/VCB/VCB_future_predictions.csv | head -20

# Show prediction chart
open output/VCB/VCB_future.png
```

#### Stop:
```bash
./stop_batch_demo.sh
```

---

## 📊 QUICK SQL QUERIES

### Count records:
```sql
SELECT COUNT(*) FROM stock_prices;
```

### Show recent 10:
```sql
SELECT * FROM stock_prices ORDER BY created_at DESC LIMIT 10;
```

### Group by stock:
```sql
SELECT stock_symbol, COUNT(*) FROM stock_prices GROUP BY stock_symbol;
```

### Clean database:
```sql
DELETE FROM stock_prices;
DELETE FROM crawl_metadata;
```

---

## 🎓 KEY TALKING POINTS

### STREAMING:
- ✅ Kafka architecture for real-time data
- ✅ **6 partitions** for parallel processing & scalability
- ✅ **Hash-based partitioning** (stock symbol → partition)
- ✅ Tick aggregation (intraday → daily OHLCV)
- ✅ Simulated for fast demo (30 ticks/sec)
- ✅ Production-ready transformation logic

### BATCH:
- ✅ Real API integration (VNDirect)
- ✅ Incremental crawling (efficient)
- ✅ Advanced LSTM with 20+ features
- ✅ 30-day predictions
- ✅ Automated scheduling

---

## 🚀 COMPLETE FLOW (15-20 minutes)

1. **Prep** (1 min): `./start_database.sh`
2. **Streaming Demo** (5 min):
   - Start: `./start_streaming_demo.sh`
   - Show Kafka UI
   - Query database
   - Stop: `./stop_streaming_demo.sh`
   - Clean: `./cleanup_database.sh`
3. **Batch Demo** (10 min):
   - Start: `./start_batch_demo.sh`
   - Trigger crawler in Airflow
   - Monitor database
   - Show output files
   - Stop: `./stop_batch_demo.sh`

---

## 🛠️ SCRIPTS SUMMARY

```
start_database.sh             # Setup PostgreSQL (once)
setup_kafka_topic.py          # Setup Kafka topic with partitions
check_kafka_partitions.sh     # Check partition distribution
start_streaming_demo.sh       # Start streaming demo
stop_streaming_demo.sh        # Stop streaming demo
start_batch_demo.sh           # Start batch demo
stop_batch_demo.sh            # Stop batch demo
cleanup_database.sh           # Clean DB between demos
```

---

## 🐛 TROUBLESHOOTING

### Port conflicts:
```bash
# Stop Airflow
pkill -f airflow

# Stop Kafka
docker-compose down

# Restart
./start_streaming_demo.sh  # or start_batch_demo.sh
```

### Database issues:
```bash
# Restart PostgreSQL
brew services restart postgresql@15

# Re-setup
./start_database.sh
```

### Docker not running:
```bash
# Open Docker Desktop
open -a Docker
# Wait 10 seconds, then retry
```

---

**Ready to demo! 🎉**

