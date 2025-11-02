#!/usr/bin/env python3
"""
Monitored version of Antiks-v1 training
Includes real-time logging to training_monitor API
"""
import requests
import sys
from pathlib import Path

# Import the regular training functions
sys.path.insert(0, str(Path(__file__).parent))
from train_antiks_v1 import *

# Training monitor API
MONITOR_API = "http://localhost:5000/api"


def log_to_monitor(event_type: str, data: dict):
    """Log to the monitoring API"""
    try:
        # This will be handled by the training script injecting logs
        pass
    except:
        pass


# Monkey-patch the main training functions to add logging
original_active_learning = active_learning_iteration
original_generate_challenge = generate_challenge_question
original_student_answer = student_answer_question
original_teacher_evaluate = teacher_evaluate


def logged_train_on_difficulty_level(model, tokenizer, difficulty: str, output_dir: Path, 
                                     threshold: float, min_questions: int, max_questions: int) -> Dict:
    """Train on difficulty level with logging"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🎯 Training on {difficulty.upper()} difficulty (target: {threshold}%)")
    logger.info(f"{'='*80}")
    
    scores = []
    training_examples = []
    question_count = 0
    
    while question_count < max_questions:
        question_count += 1
        logger.info(f"\n[{difficulty.upper()}] Question {question_count}/{max_questions}:")
        
        # Generate question
        challenge = generate_challenge_question(difficulty=difficulty)
        if not challenge:
            logger.warning("  ⚠️  Failed to generate challenge, skipping")
            continue
        
        question = challenge['question']
        logger.info(f"  Question: {question}")
        
        # Log challenge
        try:
            requests.post(f"{MONITOR_API}/monitor/log", json={
                "type": "challenge",
                "data": {
                    "iteration": 0,  # Will be set by caller
                    "number": question_count,
                    "question": question,
                    "difficulty": difficulty
                }
            }, timeout=1)
        except:
            pass
        
        # Student attempts answer
        logger.info(f"  ⏳ Generating answer...")
        student_answer = student_answer_question(tokenizer, model, question)
        logger.info(f"  Student answer: {student_answer[:100]}...")
        
        # Log answer
        try:
            requests.post(f"{MONITOR_API}/monitor/log", json={
                "type": "answer",
                "data": {"answer": student_answer}
            }, timeout=1)
        except:
            pass
        
        # Teacher evaluates
        evaluation = teacher_evaluate(student_answer, question)
        score = evaluation.get('score', 0)
        is_correct = evaluation.get('correct', False)
        scores.append(score)
        
        logger.info(f"  Score: {score}/100 {'✅' if is_correct else '❌'}")
        
        # Log evaluation
        try:
            requests.post(f"{MONITOR_API}/monitor/log", json={
                "type": "evaluation",
                "data": evaluation
            }, timeout=1)
        except:
            pass
        
        # Add to training data if wrong
        if not is_correct or score < threshold:
            correct_answer = evaluation.get('corrected_sql') or student_answer
            
            if not correct_answer or not correct_answer.strip():
                logger.warning(f"  ⚠️  No valid SQL available, skipping")
                continue
            
            training_examples.append({
                "messages": [
                    {"role": "system", "content": "You are an expert SQL syntax assistant. Generate SQL queries focusing on syntax patterns and structure. Use generic table/column names if needed. Focus on correct SQL syntax patterns (CTEs, window functions, aggregations), not specific schema."},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": correct_answer}
                ]
            })
            
            logger.info(f"  📝 Added to training data")
        else:
            logger.info(f"  ✅ Answer is correct")
        
        # Check threshold after minimum questions
        if question_count >= min_questions:
            avg_score = sum(scores) / len(scores)
            logger.info(f"\n  📊 Current Average: {avg_score:.1f}% (target: {threshold}%)")
            
            if avg_score >= threshold:
                logger.info(f"  ✅ Threshold reached! Average score: {avg_score:.1f}% >= {threshold}%")
                break
        
        time.sleep(2)
    
    final_avg = sum(scores) / len(scores) if scores else 0
    
    return {
        "difficulty": difficulty,
        "questions_asked": question_count,
        "average_score": final_avg,
        "threshold_met": final_avg >= threshold,
        "training_examples": training_examples,
        "scores": scores
    }


def logged_active_learning_iteration(model, tokenizer, iteration: int, output_dir: Path) -> Dict:
    """Active learning iteration with progressive difficulty and logging"""
    
    # Log iteration start
    try:
        requests.post(f"{MONITOR_API}/monitor/log", json={
            "type": "iteration",
            "data": {"iteration": iteration}
        }, timeout=1)
    except:
        pass
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Iteration {iteration}: Progressive Difficulty Training")
    logger.info(f"{'='*80}")
    
    # Difficulty configuration
    # Max questions set high to ensure threshold can be reached
    # But training stops early when threshold is achieved
    difficulty_config = {
        "basic": {
            "threshold": 80.0,
            "min_questions": 8,
            "max_questions": 50  # Large limit, but stops early at 80%
        },
        "medium": {
            "threshold": 75.0,
            "min_questions": 10,
            "max_questions": 60  # Large limit, but stops early at 75%
        },
        "complex": {
            "threshold": 60.0,
            "min_questions": 12,
            "max_questions": 80  # Large limit, but stops early at 60%
        }
    }
    
    all_training_examples = []
    difficulty_results = []
    
    # Train on each difficulty level in order
    for difficulty in ["basic", "medium", "complex"]:
        config = difficulty_config[difficulty]
        
        result = logged_train_on_difficulty_level(
            model, tokenizer, difficulty, output_dir,
            threshold=config["threshold"],
            min_questions=config["min_questions"],
            max_questions=config["max_questions"]
        )
        
        difficulty_results.append(result)
        all_training_examples.extend(result["training_examples"])
        
        # Train on examples from this difficulty
        if result["training_examples"]:
            temp_file = output_dir / f"temp_{difficulty}_iter_{iteration}.jsonl"
            with open(temp_file, 'w') as f:
                for ex in result["training_examples"]:
                    f.write(json.dumps(ex) + '\n')
            
            train_on_active_learning_data(model, tokenizer, [temp_file], iteration)
            temp_file.unlink()
        
        if not result["threshold_met"]:
            logger.warning(f"  ⚠️  {difficulty} threshold not met ({result['average_score']:.1f}% < {config['threshold']}%), continuing...")
    
    # Save all training examples
    if all_training_examples:
        output_file = output_dir / f"active_learning_iter_{iteration}.jsonl"
        with open(output_file, 'w') as f:
            for ex in all_training_examples:
                f.write(json.dumps(ex) + '\n')
        logger.info(f"\n✅ Saved {len(all_training_examples)} total training examples")
        
        # Print summary
        logger.info(f"\n📊 Difficulty Summary:")
        for r in difficulty_results:
            status = "✅" if r["threshold_met"] else "⚠️"
            logger.info(f"  {status} {r['difficulty'].upper()}: {r['average_score']:.1f}% "
                       f"({r['questions_asked']} questions, {len(r['training_examples'])} examples)")
        
        return {
            "examples": len(all_training_examples),
            "file": output_file,
            "difficulty_results": difficulty_results
        }
    
    logger.info("\nℹ️  No new training examples generated")
    return {"examples": 0, "difficulty_results": difficulty_results}


# Override the function
active_learning_iteration = logged_active_learning_iteration


def main():
    """Main training pipeline with monitoring"""
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=5, help='Number of active learning iterations')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / "active_learning_data"
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*80)
    logger.info("🎓 Antiks-v1 Active Learning Training Pipeline (MONITORED)")
    logger.info("="*80)
    logger.info(f"Approach: Self-correcting SQL validation (smolagents-style)")
    logger.info(f"Student: {CONFIG['base_model']}")
    logger.info(f"Model Name: {CONFIG['model_name']}")
    logger.info(f"Active Learning Iterations: {args.iterations}")
    logger.info(f"Monitor: http://localhost:5000")
    logger.info("="*80)
    
    # Check if monitor is running
    try:
        requests.get(f"{MONITOR_API}/status", timeout=2)
        logger.info("✅ Training monitor connected!")
    except:
        logger.warning("⚠️  Training monitor not running. Run in another terminal: python training/training_monitor.py")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # Load tokenizer and model
    logger.info("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device == "mps":
        model = model.to("mps")
    
    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["target_modules"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Active learning loop (pure - no static data)
    total_new_examples = 0
    iteration_files = []
    
    for iteration in range(1, args.iterations + 1):
        logger.info("\n" + "="*80)
        logger.info(f"Active Learning Iteration {iteration}/{args.iterations}")
        logger.info("="*80)
        
        # Generate challenges and evaluate
        result = active_learning_iteration(model, tokenizer, iteration, output_dir)
        total_new_examples += result.get('examples', 0)
        
        if result.get('file'):
            iteration_files.append(result['file'])
        
        # Log training start
        if iteration_files:
            try:
                requests.post(f"{MONITOR_API}/monitor/log", json={
                    "type": "training",
                    "data": {"iteration": iteration}
                }, timeout=1)
            except:
                pass
        
        # Train on new examples if any
        if iteration_files:
            train_on_active_learning_data(model, tokenizer, iteration_files, iteration)
            iteration_files = []  # Clear after training
    
    # Final save
    final_output_dir = Path(CONFIG["output_dir"])
    final_output_dir.mkdir(exist_ok=True)
    trainer = Trainer(model=model)
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    
    # Log completion
    try:
        requests.post(f"{MONITOR_API}/monitor/log", json={
            "type": "complete",
            "data": {"examples": total_new_examples}
        }, timeout=1)
    except:
        pass
    
    logger.info("\n" + "="*80)
    logger.info("✅ Antiks-v1 Training Complete!")
    logger.info("="*80)
    logger.info(f"Total new examples generated: {total_new_examples}")
    logger.info(f"Model saved to: {final_output_dir}")
    logger.info("\nNext: python training/deploy_to_ollama.sh")


if __name__ == "__main__":
    main()

