# Antiks-v1: SQL Analytics AI 🚀

> **Active Learning Training with DeepSeek Teacher**

Train a powerful SQL analytics assistant using teacher-student active learning. Antiks-v1 learns from DeepSeek's corrections and continuously improves.

## 🎯 What Is Antiks-v1?

Antiks-v1 is an SQL analytics model trained using **active learning**:
- 🤖 **Teacher** (DeepSeek): Generates challenges, evaluates answers, provides corrections
- 🎓 **Student** (Antiks): Learns from mistakes, trains iteratively, improves continuously
- 📈 **Result**: Model that approaches teacher-level performance

## ✨ Key Features

✅ **Active Learning**: Adaptive training based on weaknesses  
✅ **Teacher Validation**: DeepSeek validates every example  
✅ **Complex SQL**: Handles CTEs, window functions, pivots  
✅ **Data Cleaning**: Uses CLEAN_NUMERIC for dirty data  
✅ **Business Focus**: Real-world analytics scenarios  
✅ **Iterative Improvement**: Continuous learning loop  

## 🚀 Quick Start

### 1. System Check
```bash
python3 test_active_learning.py
```

### 2. Start Training
```bash
# Quick test (2 iterations, ~1 hour)
python training/train_antiks_v1.py --iterations 2

# Full training (10 iterations, ~5 hours) 
python training/train_antiks_v1.py --iterations 10
```

### 3. Use Your Model
```bash
ollama run antiks-v1 "Calculate year-over-year revenue growth"
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `ANTIKS_TRAINING_QUICKSTART.md` | Quick start guide |
| `ANTIKS_V1_SUMMARY.md` | Complete overview |
| `training/TRAIN_ANTIKS.md` | Full technical docs |
| `DEEPSEEK_SETUP.md` | DeepSeek API setup |
| `TRAINING_SUMMARY.md` | Training pipeline overview |

## 🏗️ Architecture

```
DeepSeek (Teacher)
    ↓
    Generates SQL challenges
    ↓
Antiks (Student)
    ↓
    Attempts answers
    ↓
DeepSeek evaluates
    ↓
    If wrong: Train on correction
    ↓
    Better model
    ↓
    Repeat until teacher-level
```

## 📊 Training Status

| Component | Status |
|-----------|--------|
| DeepSeek API | ✅ Working |
| Challenge Generation | ✅ Tested |
| Answer Evaluation | ✅ Verified |
| Training Pipeline | ✅ Ready |
| Deployment | ✅ Ready |
| Existing Data | ✅ 549 examples |

## 🎓 Capabilities

After training, Antiks-v1 can:
- Generate complex SQL analytics queries
- Use CLEAN_NUMERIC() for dirty data
- Handle unit conversions (crores→millions)
- Create pivot tables with CASE WHEN
- Use CTEs, window functions, advanced aggregations
- Follow SQLite best practices
- Answer business analytics questions

## 🔧 Configuration

Edit `training/train_antiks_v1.py`:
```python
CONFIG = {
    "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
    "model_name": "Antiks-v1",
    "lora_r": 32,
    "num_epochs": 3,
    "iterations": 10,
    # ... more settings
}
```

## 📈 Expected Results

| Iteration | Avg Score | Training Examples |
|-----------|-----------|-------------------|
| 1 | ~70% | ~30 added |
| 3 | ~80% | ~20 added |
| 5 | ~85% | ~15 added |
| 10 | ~90% | ~5 added |

## 🛠️ Requirements

- Python 3.9+
- CUDA/MPS for acceleration
- DeepSeek API key (OpenRouter)
- 16GB+ RAM recommended
- ~10GB disk space

## 🔗 Related Projects

- `base_v2.py` - Main SQL analytics CLI
- `data_loader.py` - CSV data loader
- `sql_runner.py` - SQL execution engine
- `training/train_simple.py` - Static training pipeline

## 🎉 Ready?

Everything is set up and tested. Start training:

```bash
python training/train_antiks_v1.py --iterations 10
```

Watch Antiks-v1 learn from the teacher and reach teacher-level performance! 🚀

---

**Status**: ✅ Ready to train  
**Next Step**: Run training command above  
**Time**: ~5 hours for full training  
**Goal**: 90%+ performance on SQL analytics tasks

