#!/bin/bash
# Deploy fine-tuned model to Ollama

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"
MERGED_DIR="$SCRIPT_DIR/merged_model"
MODEL_NAME="analytics-sql"

echo "=================================================="
echo "🚀 Deploying Fine-tuned Model to Ollama"
echo "=================================================="

# Check if model exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ Error: Model directory not found: $OUTPUT_DIR"
    echo "   Run training first: python training/train_model.py"
    exit 1
fi

# Step 1: Merge LoRA with base model
if [ ! -d "$MERGED_DIR" ]; then
    echo ""
    echo "🔀 Step 1: Merging LoRA adapter with base model..."
    echo "--------------------------------------------------"
    
    python3 "$SCRIPT_DIR/convert_to_gguf.py"
    
    if [ ! -d "$MERGED_DIR" ]; then
        echo "❌ Merge failed"
        exit 1
    fi
    
    echo "✅ Model merged successfully"
else
    echo "✅ Merged model already exists: $MERGED_DIR"
fi

# Step 2: Create Modelfile
echo ""
echo "📝 Step 2: Creating Ollama Modelfile..."
echo "--------------------------------------------------"

MODELFILE="$SCRIPT_DIR/Modelfile"

cat > "$MODELFILE" << EOF
# Fine-tuned SQL Analytics Model
FROM $MERGED_DIR

TEMPLATE """<s>[INST] <<SYS>>
{{ .System }}
<</SYS>>

{{ .Prompt }} [/INST]"""

SYSTEM """You are an expert SQL analytics assistant. You generate clean, efficient SQL queries for SQLite databases.

Key capabilities:
- Always use CLEAN_NUMERIC() function for numeric columns to handle dirty data (commas, currency symbols, percentages)
- Support window functions, CTEs, aggregations, and complex analytics
- Generate pivot tables using CASE statements
- Handle unit conversions (crores, millions, percentages)
- Write safe, read-only SELECT queries only

Best practices:
- Use proper column aliases
- Format SQL for readability
- Add comments for complex queries
- Always validate data types"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER stop "<s>"
PARAMETER stop "</s>"
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
EOF

echo "✅ Modelfile created: $MODELFILE"

# Step 3: Create model in Ollama
echo ""
echo "🎯 Step 3: Creating model in Ollama..."
echo "--------------------------------------------------"

if ollama list | grep -q "$MODEL_NAME"; then
    echo "⚠️  Model '$MODEL_NAME' already exists. Deleting..."
    ollama rm "$MODEL_NAME"
fi

ollama create "$MODEL_NAME" -f "$MODELFILE"

echo "✅ Model created: $MODEL_NAME"

# Step 4: Test the model
echo ""
echo "🧪 Step 4: Testing model..."
echo "--------------------------------------------------"

TEST_PROMPT="What is the total sales by category?"

echo "Test question: $TEST_PROMPT"
echo ""

ollama run "$MODEL_NAME" "$TEST_PROMPT"

# Done!
echo ""
echo "=================================================="
echo "🎉 Deployment Complete!"
echo "=================================================="
echo ""
echo "Model name: $MODEL_NAME"
echo ""
echo "To use this model:"
echo "  export OLLAMA_MODEL=$MODEL_NAME"
echo "  python base_v2.py"
echo ""
echo "Or test directly:"
echo "  ollama run $MODEL_NAME 'your question here'"
echo ""

