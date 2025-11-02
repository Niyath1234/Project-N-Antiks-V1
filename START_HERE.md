# 🚀 Start Here - SQL Analytics AI

## Quick Start

### 1️⃣ Start Ollama
```bash
ollama serve
```

### 2️⃣ Choose Your Training Method

#### Option A: Active Learning (Antiks-v1) - **Recommended**
```bash
cd /Users/niyathnair/python_nn
source venv/bin/activate
python training/train_antiks_v1.py --iterations 10
```

**Result:** Teacher-level performance in ~5 hours

#### Option B: Static Training
```bash
cd /Users/niyathnair/python_nn
source venv/bin/activate
python training/generate_synthetic_data.py
python training/train_simple.py
bash training/deploy_to_ollama.sh
```

**Result:** Good performance in ~2 hours

### 3️⃣ Test Your Model
```bash
export CSV_PATH="/path/to/your/data.csv"
python base_v2.py
```

---

## Project Overview

### What You Built
- ✅ **Text-to-SQL analytics tool** - Natural language → SQL queries
- ✅ **Smart data cleaning** - Handles dirty numeric data automatically
- ✅ **Complex analytics** - Pivots, window functions, CTEs
- ✅ **Fine-tuning pipeline** - Mistral-7B optimized for SQL
- ✅ **Active learning** - Antiks-v1 with DeepSeek teacher

### Key Components
```
python_nn/
├── Core App
│   ├── base_v2.py              # Main CLI
│   ├── sql_runner.py           # SQL engine
│   ├── data_loader.py          # CSV handling
│   └── conversation_manager.py # Chat memory
│
├── Training
│   ├── train_antiks_v1.py      # Active learning
│   ├── train_simple.py         # Static training
│   ├── deploy_to_ollama.sh     # Deployment
│   └── test_csv.py             # Benchmarks
│
└── Documentation
    ├── ANTIKS_README.md        # Antiks-v1 guide
    ├── training/README.md      # Training docs
    └── DEEPSEEK_SETUP.md       # API setup
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| `ANTIKS_README.md` | Antiks-v1 overview |
| `ANTIKS_TRAINING_QUICKSTART.md` | Quick start |
| `ANTIKS_V1_SUMMARY.md` | Complete details |
| `training/TRAIN_ANTIKS.md` | Technical docs |
| `training/README.md` | Training guide |

---

## Next Steps

1. **Read**: `ANTIKS_TRAINING_QUICKSTART.md`
2. **Train**: `python training/train_antiks_v1.py --iterations 10`
3. **Use**: `python base_v2.py`

---

**Ready to train Antiks-v1? Start here!** 🚀
