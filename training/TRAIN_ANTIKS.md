# Antiks-v1 Active Learning Training Pipeline

## Overview
**Antiks-v1** is trained using **active learning** with DeepSeek as the teacher:
1. Teacher generates challenging SQL questions
2. Student (Mistral-7B) attempts to answer
3. Teacher evaluates the answer
4. Wrong answers become training examples
5. Model trains on new examples and improves
6. Repeat until teacher-level performance

## Architecture

### Teacher: DeepSeek Chat v3
- API: OpenRouter
- Role: Generate challenges, evaluate answers, provide corrections
- Expertise: SQL analytics, best practices, SQLite-specific knowledge

### Student: Antiks-v1 (based on Mistral-7B)
- Base Model: `mistralai/Mistral-7B-Instruct-v0.2`
- LoRA Rank: 32 (higher for complex SQL)
- Target Modules: All attention and MLP layers
- Training: Iterative active learning

## Key Features

### Active Learning Loop
```
Iteration N:
  1. Generate 10 challenging SQL questions
  2. For each question:
     a. Student generates SQL answer
     b. Teacher evaluates (score 0-100)
     c. If score < 80: Add to training data with teacher's correction
  3. Train model on new examples
  4. Check if convergence reached
```

### Challenge Categories
- Revenue/Finance Analysis
- Customer Analytics  
- Time-Series Analysis
- Data Cleaning (CLEAN_NUMERIC)
- Pivot Tables & Cross-tabs
- Advanced Patterns (CTEs, window functions)

## Training Process

### Phase 0: Initial Training (Optional)
Train on existing YouTube data to establish baseline:
```bash
python training/train_antiks_v1.py --initial_train --iterations 5
```

### Phase 1-N: Active Learning
```bash
python training/train_antiks_v1.py --iterations 10
```

This will:
- Run 10 iterations of active learning
- Each iteration: generate 10 challenges
- Train after each iteration on new data
- Save final model as Antiks-v1

## Expected Results

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Avg Score | >85/100 | Teacher evaluation |
| Syntax Pass | >95% | Execution tests |
| Best Practices | >90% | CLEAN_NUMERIC usage |
| Complex Queries | >80% | CTEs, window functions |
| Teacher-Level | ~90% | Convergence threshold |

## Deployment

After training:
```bash
# Convert to GGUF
python training/convert_to_gguf.py

# Deploy to Ollama
bash training/deploy_to_ollama.sh antiks-v1

# Test
ollama run antiks-v1 "How do I calculate year-over-year growth in revenue?"
```

## Monitoring

Watch training logs for:
- Challenge generation success rate
- Score distribution over iterations
- Training examples accumulation
- Loss convergence

## Advantages Over Static Training

✅ **Adaptive**: Focuses on model's weaknesses
✅ **Efficient**: Only trains on mistakes
✅ **Scalable**: Infinite challenge generation
✅ **Quality**: Teacher-validated examples
✅ **Iterative**: Continuous improvement

## Time Estimates

| Phase | Time |
|-------|------|
| Initial Train (optional) | ~1-2 hours |
| Active Learning (10 iter) | ~3-4 hours |
| Per Iteration | ~20 minutes |
| Total (full pipeline) | ~5-6 hours |

## Next Steps After Training

1. Evaluate on test suite
2. Benchmark against teacher
3. Fine-tune if needed
4. Deploy to production
5. Monitor performance

## Configuration

Edit `training/train_antiks_v1.py` CONFIG dict:
- `lora_r`: LoRA rank (32 default, increase for more capacity)
- `num_epochs`: Training epochs per iteration (3 default)
- `iterations`: Number of active learning loops
- `batch_size`: Training batch size

---

**Status**: Ready to train!
**Start Command**: `python training/train_antiks_v1.py --iterations 5`

