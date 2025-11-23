#!/bin/bash

# Vercel deployment script
set -e

echo "🚀 Deploying ML Model to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Configuration
PROJECT_NAME=${PROJECT_NAME:-"ml-model"}
MODEL_PATH=${MODEL_PATH:-"../../models/model.pkl"}

echo "📦 Preparing deployment..."

# Copy source code
rm -rf src
cp -r ../../src .

# Copy model if exists and is small enough (<50MB for Vercel)
if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH" 2>/dev/null)
    MAX_SIZE=$((50 * 1024 * 1024))  # 50MB

    if [ "$MODEL_SIZE" -lt "$MAX_SIZE" ]; then
        echo "📊 Copying model ($(($MODEL_SIZE / 1024 / 1024))MB)..."
        mkdir -p models
        cp "$MODEL_PATH" models/model.pkl
    else
        echo "⚠️  Warning: Model is too large ($(($MODEL_SIZE / 1024 / 1024))MB > 50MB)"
        echo "Consider using a smaller model or hosting it externally (e.g., S3, GCS)"
    fi
fi

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."

if [ "$1" = "production" ]; then
    echo "📡 Deploying to production..."
    vercel --prod
else
    echo "📡 Deploying to preview..."
    vercel
fi

echo "✅ Deployment complete!"

# Cleanup
echo "🧹 Cleaning up..."
rm -rf src

echo "✨ Done!"
echo ""
echo "To deploy to production, run: ./deploy.sh production"
