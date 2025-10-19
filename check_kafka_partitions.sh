#!/bin/bash
#
# Check Kafka Topic Partitions
# 
# Shows detailed information about topic partitions, including:
# - Number of partitions
# - Messages per partition
# - Consumer lag per partition
#

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "KAFKA TOPIC PARTITIONS CHECK"
echo "========================================"
echo ""

# Check if Kafka is running
if ! docker ps | grep -q vn30-kafka; then
    echo -e "${RED}✗ Kafka container is not running${NC}"
    echo ""
    echo "Start Kafka first:"
    echo "  docker-compose up -d"
    exit 1
fi

echo -e "${GREEN}✓ Kafka is running${NC}"
echo ""

# Get topic name from config
TOPIC_NAME="vn30-stock-prices"

echo "=========================================="
echo "TOPIC: $TOPIC_NAME"
echo "=========================================="
echo ""

# Describe topic
echo -e "${BLUE}1. Topic Configuration:${NC}"
docker exec -it vn30-kafka kafka-topics \
    --bootstrap-server localhost:9092 \
    --describe \
    --topic "$TOPIC_NAME" 2>/dev/null

echo ""
echo "=========================================="
echo -e "${BLUE}2. Messages per Partition:${NC}"
echo "=========================================="

# Get partition count
PARTITIONS=$(docker exec vn30-kafka kafka-topics \
    --bootstrap-server localhost:9092 \
    --describe \
    --topic "$TOPIC_NAME" 2>/dev/null | grep "PartitionCount" | awk '{print $4}')

if [ -z "$PARTITIONS" ]; then
    echo -e "${YELLOW}⚠️  Topic not found or no partitions${NC}"
    exit 0
fi

echo "Total Partitions: $PARTITIONS"
echo ""

# Check messages in each partition
for ((i=0; i<$PARTITIONS; i++)); do
    # Get earliest and latest offset
    EARLIEST=$(docker exec vn30-kafka kafka-run-class kafka.tools.GetOffsetShell \
        --broker-list localhost:9092 \
        --topic "$TOPIC_NAME" \
        --partitions $i \
        --time -2 2>/dev/null | awk -F: '{print $3}')
    
    LATEST=$(docker exec vn30-kafka kafka-run-class kafka.tools.GetOffsetShell \
        --broker-list localhost:9092 \
        --topic "$TOPIC_NAME" \
        --partitions $i \
        --time -1 2>/dev/null | awk -F: '{print $3}')
    
    # Calculate message count
    if [ -n "$EARLIEST" ] && [ -n "$LATEST" ]; then
        COUNT=$((LATEST - EARLIEST))
        echo -e "  Partition $i: ${GREEN}$COUNT messages${NC} (offset: $EARLIEST → $LATEST)"
    else
        echo -e "  Partition $i: ${YELLOW}Unknown${NC}"
    fi
done

echo ""
echo "=========================================="
echo -e "${BLUE}3. Consumer Group Lag:${NC}"
echo "=========================================="

# Check consumer group
CONSUMER_GROUP="stock-prediction-consumer-group"

docker exec vn30-kafka kafka-consumer-groups \
    --bootstrap-server localhost:9092 \
    --describe \
    --group "$CONSUMER_GROUP" 2>/dev/null || echo -e "${YELLOW}⚠️  No active consumer group${NC}"

echo ""
echo "=========================================="
echo -e "${BLUE}4. Partition Leadership:${NC}"
echo "=========================================="

docker exec vn30-kafka kafka-topics \
    --bootstrap-server localhost:9092 \
    --describe \
    --topic "$TOPIC_NAME" 2>/dev/null | grep "Partition:" | awk '{print "  Partition "$2": Leader="$6", Replicas="$8", Isr="$10}'

echo ""
echo "=========================================="
echo "✓ CHECK COMPLETED"
echo "=========================================="
echo ""
echo "View in Kafka UI: http://localhost:8080"
echo ""

