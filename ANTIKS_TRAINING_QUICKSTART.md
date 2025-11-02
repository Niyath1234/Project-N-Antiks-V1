# Antiks-v1 Training: Quick Start Guide

## ✅ System Check
All components tested and working:
- ✅ DeepSeek API (OpenRouter) - Working
- ✅ Challenge generation - Working
- ✅ Answer evaluation - Working
- ✅ Model loading - Ready
- ✅ MPS acceleration - Available

## 🚀 Start Training

### Option 1: Quick Test Run (2 iterations)
```bash
cd /Users/niyathnair/python_nn
source venv/bin/activate
python3 training/train_antiks_v1.py --iterations 2
```

### Option 2: Full Training (10 iterations - Recommended)
```bash
cd /Users/niyathnair/python_nn
source venv/bin/activate
python3 training/train_antiks_v1.py --iterations 10
```

### Option 3: With Initial Training on Existing Data
```bash
cd /Users/niyathnair/python_nn
source venv/bin/activate
python3 training/train_antiks_v1.py --initial_train --iterations 5
```

## 📊 What Happens

### Each Iteration:
1. 🤖 **Teacher** generates 10 challenging SQL questions
2. 🎓 **Student** (Antiks) attempts to answer each
3. ✅ **Teacher** evaluates each answer (score 0-100)
4. 📝 **Wrong answers** (<80 score) become training examples
5. 🔄 **Model trains** on new examples
6. 📈 **Performance improves** over iterations

### Expected Output:
```
Iteration 1:
  Challenge 1: ✅ Score: 95 (no training needed)
  Challenge 2: ❌ Score: 45 (added to training)
  Challenge 3: ❌ Score: 60 (added to training)
  ...
  Saved 7 new training examples
  Training on 7 examples...
  
Iteration 2:
  Challenge 1: ✅ Score: 92 (no training needed)
  Challenge 2: ✅ Score: 88 (no training needed)
  ...
```

## ⏱️ Time Estimates

| Iterations | Time | What You Get |
|------------|------|--------------|
| 2 | ~1 hour | Proof of concept |
| 5 | ~3 hours | Significant improvement |
| 10 | ~5 hours | Teacher-level performance |

## 📁 Output Files

```
output_antiks_v1/
  ├── adapter_model.bin          # LoRA weights
  ├── adapter_config.json        # LoRA config
  └── training_args.bin          # Training settings

checkpoints_antiks_v1_iter_N/    # Per-iteration checkpoints

active_learning_data/
  └── active_learning_iter_N.jsonl  # Training examples per iteration
```

## 🎯 Success Metrics

Watch for:
- ✅ Average score increases over iterations
- ✅ Fewer examples added per iteration (model getting smarter)
- ✅ Training loss decreases
- ✅ Model generates correct SQL more often

## 🔄 After Training

### Deploy to Ollama:
```bash
# Convert to GGUF format
python training/convert_to_gguf.py

# Deploy
bash training/deploy_to_ollama.sh antiks-v1

# Test
ollama run antiks-v1 "Calculate year-over-year revenue growth"
```

### Test Performance:
```bash
python training/test_csv.py
```

## 🐛 Troubleshooting

### API Rate Limits
If hitting rate limits, add delays in code or run fewer iterations at once.

### Memory Issues
If running out of memory:
- Reduce `batch_size` in CONFIG
- Reduce `max_length`
- Close other applications

### Long Training Times
Training on MPS is slower than CUDA. Consider:
- Running overnight
- Using cloud GPU
- Reducing `num_epochs` per iteration

## 📖 Full Documentation

See `training/TRAIN_ANTIKS.md` for complete details.

## 🎉 Ready?

```bash
python3 training/train_antiks_v1.py --iterations 5
```

Let's train Antiks-v1 to teacher-level! 🚀

