# SQL Analytics AI

AI-powered text-to-SQL analytics tool with fine-tuned models for business intelligence.

## 🚀 Quick Start

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Start Ollama (in separate terminal)
ollama serve

# 3. Run analytics tool
export CSV_PATH="/path/to/your/data.csv"
python base_v2.py
```

## 📁 Project Structure

```
python_nn/
├── base_v2.py              # Main CLI application
├── sql_runner.py           # SQL execution engine
├── data_loader.py          # CSV loading & schema inference
├── conversation_manager.py # Multi-turn conversation handling
├── custom_rules.py         # Dynamic rule learning system
├── schema_prompt.txt       # Core SQL generation prompt
├── requirements.txt        # Python dependencies
├── training/               # Fine-tuning pipeline
│   ├── train_simple.py    # Main training script (Mistral-7B)
│   ├── generate_training_data.py    # YouTube → SQL examples
│   ├── generate_synthetic_data.py   # Synthetic data generation
│   ├── deploy_to_ollama.sh          # Model deployment
│   ├── convert_to_gguf.py           # Model conversion
│   ├── test_csv.py        # Benchmark testing
│   ├── train_data/        # Training datasets
│   └── README.md          # Training documentation
└── venv/                  # Virtual environment
```

## 🎯 Features

- **Natural Language to SQL**: Ask questions in plain English
- **Smart Data Cleaning**: Automatic handling of dirty numeric data
- **Complex Analytics**: Pivots, window functions, CTEs, aggregations
- **Unit Conversions**: Crores, millions, percentages
- **Conversation Memory**: Follow-up questions and refinements
- **Custom Rules**: Learn from corrections dynamically
- **Fine-tuned Models**: Mistral-7B optimized for SQL analytics

## 📊 Usage

### Basic Query
```python
Q: What is the total sales?
```

### Complex Analytics
```python
Q: Show category wise revenue with year columns (pivot)
Q: Calculate year over year growth
Q: Give me top 10 customers by revenue
```

### Dynamic Rules
```python
:learn Always use CLEAN_NUMERIC for the Value column
```

### Commands
- `:load <path>` - Switch dataset
- `:history` - View conversation
- `:rules` - Show learned rules
- `:learn <rule>` - Add new rule
- `exit` - Quit

## 🔧 Fine-tuning Your Model

See [training/README.md](training/README.md) for complete guide.

```bash
# 1. Generate training data from YouTube
bash training/add_video.sh 'https://youtube.com/watch?v=VIDEO'

# 2. Generate synthetic data (optional)
python training/generate_synthetic_data.py

# 3. Train model
python training/train_simple.py

# 4. Deploy
bash training/deploy_to_ollama.sh

# 5. Test
python training/test_csv.py
```

## 🧪 Testing

```bash
# Run comprehensive benchmarks
export CSV_PATH="/path/to/data.csv"
python training/test_csv.py
```

## 📦 Dependencies

- Python 3.9+
- Ollama (local LLM)
- pytorch, transformers, peft (for training)
- yt-dlp (for data generation)

Install: `pip install -r requirements.txt`

## 🎓 Model Capabilities

- **Basic**: Aggregations, filtering, grouping
- **Intermediate**: Joins, subqueries, having
- **Advanced**: Window functions, CTEs, pivots
- **Expert**: YoY analysis, running totals, moving averages

## 🤝 Contributing

1. Add your CSV schema to `schema_prompt.txt`
2. Generate training examples
3. Fine-tune and test
4. Share results!

## 📄 License

MIT
