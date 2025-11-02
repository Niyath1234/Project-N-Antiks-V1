# Fine-tuning Pipeline

Complete guide to training Antiks-v1 SQL analytics model.

## Overview

**Two Training Methods:**

### 1. **Active Learning (Recommended)** - Antiks-v1
Teacher-student active learning with DeepSeek Chat v3:
- Teacher generates challenges
- Student learns from mistakes
- Iterative improvement to teacher-level

### 2. **Static Training** - Traditional
Standard fine-tuning on curated datasets:
- YouTube transcripts (~550 examples)
- Synthetic examples (optional)

---

## Quick Start

### Active Learning (Antiks-v1)

```bash
# Start training with active learning
python training/train_antiks_v1.py --iterations 10
```

See `../ANTIKS_TRAINING_QUICKSTART.md` for full guide.

### Static Training

```bash
# 1. Generate synthetic data (optional)
python training/generate_synthetic_data.py

# 2. Train
python training/train_simple.py

# 3. Deploy
bash training/deploy_to_ollama.sh
```

---

## Files

### Active Learning
| File | Purpose |
|------|---------|
| `train_antiks_v1.py` | Main active learning script |
| `TRAIN_ANTIKS.md` | Full documentation |

### Static Training
| File | Purpose |
|------|---------|
| `train_simple.py` | Main training script |
| `generate_training_data.py` | Extract SQL from YouTube |
| `generate_synthetic_data.py` | Create synthetic examples |
| `deploy_to_ollama.sh` | Convert & deploy to Ollama |
| `convert_to_gguf.py` | Model merging & conversion |
| `test_csv.py` | Benchmark testing |

### Data
| Directory | Contents |
|-----------|----------|
| `train_data/` | Training JSONL files (549 examples) |
| `train_example/` | Sample examples for review |
| `active_learning_data/` | Generated during Antiks training |

---

## Architecture

### Active Learning (Antiks-v1)
- **Teacher**: DeepSeek Chat v3 via OpenRouter
- **Student**: Mistral-7B with LoRA
- **Method**: Iterative challenge-response-evaluate-train
- **Result**: Teacher-level performance

### Static Training
- **Base Model**: Mistral-7B-Instruct
- **Method**: LoRA fine-tuning
- **Data**: ~600 curated examples
- **Time**: 1-2 hours

---

## Hyperparameters

### Antiks-v1
```python
base_model: Mistral-7B-Instruct
lora_r: 32
lora_alpha: 64
num_epochs: 3
learning_rate: 2e-4
max_length: 1024
```

### Static Training
```python
base_model: Mistral-7B-Instruct
lora_r: 16
lora_alpha: 32
num_epochs: 2
learning_rate: 1.5e-4
max_length: 512
```

---

## Expected Results

### Antiks-v1
- ✅ >90% teacher evaluation score
- ✅ >95% syntax correctness
- ✅ Handles complex SQL patterns
- ✅ Iterative improvement

### Static Training
- ✅ >90% syntax correctness
- ✅ >85% execution success
- ✅ Consistent best practices

---

## Documentation

| Document | Purpose |
|----------|---------|
| `../ANTIKS_README.md` | Antiks-v1 overview |
| `../ANTIKS_TRAINING_QUICKSTART.md` | Quick start guide |
| `../ANTIKS_V1_SUMMARY.md` | Complete details |
| `TRAIN_ANTIKS.md` | Technical docs |
| `../DEEPSEEK_SETUP.md` | DeepSeek API setup |
| `../TRAINING_SUMMARY.md` | General overview |

---

## Which Method to Use?

**Use Active Learning if:**
- You want maximum performance
- You have time for iterative training (~5 hours)
- You want teacher-level results

**Use Static Training if:**
- You want quick results (~1-2 hours)
- You have curated datasets
- You prefer traditional fine-tuning

---

## References

- [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [DeepSeek](https://www.deepseek.com)
- [Ollama](https://ollama.ai/)

---

**Start with Antiks-v1 for best results!**

```bash
python training/train_antiks_v1.py --iterations 10
```
