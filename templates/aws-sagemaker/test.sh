#!/bin/bash

# AWS SageMaker Endpoint Testing Script
#
# This script tests a deployed SageMaker endpoint with various payloads.

set -e

# Configuration
MODEL_NAME="${1:-my-model}"
AWS_REGION="${2:-us-east-1}"
ENVIRONMENT="${3:-dev}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Get endpoint name from Terraform
get_endpoint_name() {
    cd terraform
    ENDPOINT_NAME=$(terraform output -raw endpoint_name 2>/dev/null)
    cd ..

    if [ -z "$ENDPOINT_NAME" ]; then
        ENDPOINT_NAME="${MODEL_NAME}-endpoint-${ENVIRONMENT}"
        log_warn "Could not get endpoint name from Terraform, using default: ${ENDPOINT_NAME}"
    fi

    echo "$ENDPOINT_NAME"
}

# Test single prediction
test_single_prediction() {
    local endpoint_name=$1

    log_info "Testing single prediction..."

    PAYLOAD='{"features": [5.1, 3.5, 1.4, 0.2]}'

    aws sagemaker-runtime invoke-endpoint \
        --endpoint-name "$endpoint_name" \
        --content-type application/json \
        --body "$PAYLOAD" \
        --region "$AWS_REGION" \
        response.json

    echo "Request: $PAYLOAD"
    echo "Response:"
    cat response.json | python -m json.tool
    echo ""
}

# Test batch prediction
test_batch_prediction() {
    local endpoint_name=$1

    log_info "Testing batch prediction..."

    PAYLOAD='{"instances": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3], [5.9, 3.0, 5.1, 1.8]]}'

    aws sagemaker-runtime invoke-endpoint \
        --endpoint-name "$endpoint_name" \
        --content-type application/json \
        --body "$PAYLOAD" \
        --region "$AWS_REGION" \
        response.json

    echo "Request: $PAYLOAD"
    echo "Response:"
    cat response.json | python -m json.tool
    echo ""
}

# Test with CSV content type
test_csv_input() {
    local endpoint_name=$1

    log_info "Testing CSV input..."

    PAYLOAD='5.1,3.5,1.4,0.2'

    aws sagemaker-runtime invoke-endpoint \
        --endpoint-name "$endpoint_name" \
        --content-type text/csv \
        --body "$PAYLOAD" \
        --region "$AWS_REGION" \
        response.txt

    echo "Request: $PAYLOAD"
    echo "Response:"
    cat response.txt
    echo ""
}

# Load test
load_test() {
    local endpoint_name=$1
    local num_requests=${2:-10}

    log_info "Running load test with ${num_requests} requests..."

    PAYLOAD='{"features": [5.1, 3.5, 1.4, 0.2]}'

    START_TIME=$(date +%s)

    for i in $(seq 1 $num_requests); do
        aws sagemaker-runtime invoke-endpoint \
            --endpoint-name "$endpoint_name" \
            --content-type application/json \
            --body "$PAYLOAD" \
            --region "$AWS_REGION" \
            "response_${i}.json" &> /dev/null

        if [ $((i % 10)) -eq 0 ]; then
            echo "Completed $i requests..."
        fi
    done

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    log_info "Load test complete!"
    echo "Total requests: ${num_requests}"
    echo "Total time: ${DURATION}s"
    echo "Requests/second: $(echo "scale=2; $num_requests / $DURATION" | bc)"
    echo ""

    # Cleanup
    rm -f response_*.json
}

# Check endpoint status
check_endpoint_status() {
    local endpoint_name=$1

    log_info "Checking endpoint status..."

    aws sagemaker describe-endpoint \
        --endpoint-name "$endpoint_name" \
        --region "$AWS_REGION" \
        --query '{Name: EndpointName, Status: EndpointStatus, CreationTime: CreationTime, LastModifiedTime: LastModifiedTime}' \
        --output table

    echo ""
}

# Get endpoint metrics
get_endpoint_metrics() {
    local endpoint_name=$1

    log_info "Fetching endpoint metrics..."

    # Get current time and 1 hour ago
    END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
    START_TIME=$(date -u -d '1 hour ago' +"%Y-%m-%dT%H:%M:%S")

    echo "Invocations (last hour):"
    aws cloudwatch get-metric-statistics \
        --namespace AWS/SageMaker \
        --metric-name Invocations \
        --dimensions Name=EndpointName,Value="$endpoint_name" Name=VariantName,Value=AllTraffic \
        --start-time "$START_TIME" \
        --end-time "$END_TIME" \
        --period 3600 \
        --statistics Sum \
        --region "$AWS_REGION" \
        --query 'Datapoints[0].Sum' \
        --output text

    echo ""
    echo "Model Latency (last hour, ms):"
    aws cloudwatch get-metric-statistics \
        --namespace AWS/SageMaker \
        --metric-name ModelLatency \
        --dimensions Name=EndpointName,Value="$endpoint_name" Name=VariantName,Value=AllTraffic \
        --start-time "$START_TIME" \
        --end-time "$END_TIME" \
        --period 3600 \
        --statistics Average \
        --region "$AWS_REGION" \
        --query 'Datapoints[0].Average' \
        --output text

    echo ""
}

# Main
main() {
    log_info "Testing SageMaker endpoint for ${MODEL_NAME}"
    echo ""

    ENDPOINT_NAME=$(get_endpoint_name)

    log_info "Endpoint: ${ENDPOINT_NAME}"
    log_info "Region: ${AWS_REGION}"
    echo ""

    # Run tests
    check_endpoint_status "$ENDPOINT_NAME"
    test_single_prediction "$ENDPOINT_NAME"
    test_batch_prediction "$ENDPOINT_NAME"
    test_csv_input "$ENDPOINT_NAME"
    get_endpoint_metrics "$ENDPOINT_NAME"

    # Ask about load test
    echo -n "Run load test? (y/N): "
    read -r answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        echo -n "Number of requests [100]: "
        read -r num_requests
        num_requests=${num_requests:-100}
        load_test "$ENDPOINT_NAME" "$num_requests"
    fi

    # Cleanup
    rm -f response.json response.txt

    log_info "Testing complete!"
}

main
