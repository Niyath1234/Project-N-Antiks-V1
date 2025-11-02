#!/usr/bin/env python3
"""
Antiks-v1 Training Pipeline: Self-Correcting Active Learning (smolagents approach)
Student generates SQL, validates via execution, and self-corrects based on results
"""
import torch
import json
import time
import sqlite3
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

# Initialize logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: llama-cpp-python for GGUF inference
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not available. GGUF inference will fallback to HuggingFace model.")

CONFIG = {
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct",  # Base Qwen model for training
    "pretrained_gguf": "/Users/niyathnair/Downloads/qwen-2.5-0.5b-finetuned-for-sql-generation-combined-dataset.Q8_0.gguf",  # Pre-trained GGUF for inference
    "use_gguf_for_inference": True,  # Use GGUF model for student answers
    "model_name": "Antiks-v1",
    # Optimized for Mac M4 - higher LoRA rank for better SQL learning
    "lora_r": 32,  # Increased from 16 for better capacity (SQL is complex)
    "lora_alpha": 64,  # 2x rank for better adaptation
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # More modules for better learning
    # Optimized batch sizes for MPS (reduced for memory)
    "batch_size": 4,  # Reduced for MPS memory constraints
    "micro_batch_size": 1,  # Minimal batch size to avoid OOM on MPS
    # More training for better SQL quality
    "num_epochs": 5,  # Increased from 3 for deeper learning
    "learning_rate": 1.5e-4,  # Slightly lower for stability
    "max_length": 512,  # Reduced from 768 to save memory
    "warmup_steps": 50,  # Fewer warmup steps
    "output_dir": "./output_antiks_v1",
    "checkpoint_dir": "./checkpoints_antiks_v1",
}


# SQL Question Bank (syntax-focused, no schema dependencies)
# Expanded with many more questions and variations for comprehensive training
SQL_QUESTION_BANK = {
    "basic": [
        # Basic SELECT operations
        "How do I write SQL to select all records from a table?",
        "How do I write SQL to select specific columns from a table?",
        "How do I write SQL to select all columns using asterisk?",
        "How do I write SQL to filter records using WHERE clause?",
        "How do I write SQL to filter with multiple WHERE conditions?",
        "How do I write SQL to filter records with AND operator?",
        "How do I write SQL to filter records with OR operator?",
        "How do I write SQL to combine AND and OR in WHERE clause?",
        "How do I write SQL to use comparison operators (>, <, =, !=)?",
        "How do I write SQL to filter with greater than operator?",
        "How do I write SQL to filter with less than or equal to?",
        "How do I write SQL to filter with not equal operator?",
        "How do I write SQL to filter with BETWEEN operator?",
        "How do I write SQL to filter with IN operator?",
        "How do I write SQL to filter with NOT IN operator?",
        "How do I write SQL to filter with LIKE pattern matching?",
        "How do I write SQL to filter with LIKE and wildcard percent?",
        "How do I write SQL to filter with LIKE and underscore?",
        "How do I write SQL to handle NULL values with IS NULL?",
        "How do I write SQL to handle NULL values with IS NOT NULL?",
        "How do I write SQL to filter non-null values?",
        # Aggregations
        "How do I write SQL to count records in a table?",
        "How do I write SQL to count non-null values in a column?",
        "How do I write SQL to count distinct values?",
        "How do I write SQL to sum values in a numeric column?",
        "How do I write SQL to sum only positive values?",
        "How do I write SQL to calculate average of a numeric column?",
        "How do I write SQL to calculate average ignoring NULLs?",
        "How do I write SQL to find minimum value in a column?",
        "How do I write SQL to find maximum value in a column?",
        # Sorting and limiting
        "How do I write SQL to sort results using ORDER BY?",
        "How do I write SQL to sort in ascending order?",
        "How do I write SQL to sort in descending order?",
        "How do I write SQL to sort by multiple columns?",
        "How do I write SQL to limit results using LIMIT?",
        "How do I write SQL to get top N records?",
        "How do I write SQL to skip rows using OFFSET?",
        "How do I write SQL to combine LIMIT and OFFSET?",
        # DISTINCT
        "How do I write SQL to select distinct values from a column?",
        "How do I write SQL to select distinct combinations of multiple columns?",
        "How do I write SQL to count distinct values?",
        # Combining clauses
        "How do I write SQL to combine WHERE and ORDER BY?",
        "How do I write SQL to combine WHERE, ORDER BY, and LIMIT?",
        "How do I write SQL to filter, sort, and limit results?",
        # String operations
        "How do I write SQL to concatenate strings?",
        "How do I write SQL to convert text to uppercase?",
        "How do I write SQL to convert text to lowercase?",
        "How do I write SQL to extract substring from text?",
        "How do I write SQL to get length of string?",
        # Date operations (basic)
        "How do I write SQL to filter records by date?",
        "How do I write SQL to extract year from date?",
        "How do I write SQL to extract month from date?",
        "How do I write SQL to extract day from date?",
    ],
    "medium": [
        # GROUP BY and aggregations
        "How do I write SQL using GROUP BY to aggregate data by category?",
        "How do I write SQL using GROUP BY with single column?",
        "How do I write SQL using GROUP BY with multiple columns?",
        "How do I write SQL to group by and count records?",
        "How do I write SQL to group by and sum values?",
        "How do I write SQL to group by and calculate average?",
        "How do I write SQL using HAVING to filter grouped results?",
        "How do I write SQL using HAVING with aggregate functions?",
        "How do I write SQL to filter groups with HAVING COUNT?",
        "How do I write SQL to filter groups with HAVING SUM?",
        "How do I write SQL using multiple aggregations (SUM, COUNT, AVG) together?",
        "How do I write SQL to calculate total and average in same query?",
        # JOINs
        "How do I write SQL using INNER JOIN to combine two tables?",
        "How do I write SQL using LEFT JOIN to combine tables?",
        "How do I write SQL using RIGHT JOIN?",
        "How do I write SQL using FULL OUTER JOIN?",
        "How do I write SQL to join tables on multiple conditions?",
        "How do I write SQL to join three tables?",
        "How do I write SQL to join multiple tables with INNER JOIN?",
        "How do I write SQL using table aliases in JOINs?",
        # CTEs (single)
        "How do I write SQL using a simple CTE (WITH clause) to create a temporary result?",
        "How do I write SQL using WITH clause to simplify query?",
        "How do I write SQL using CTE to pre-calculate values?",
        "How do I write SQL using CTE to filter data before main query?",
        # CASE WHEN
        "How do I write SQL using CASE WHEN for conditional logic?",
        "How do I write SQL using CASE WHEN in SELECT clause?",
        "How do I write SQL using CASE WHEN with multiple conditions?",
        "How do I write SQL using CASE WHEN with ELSE clause?",
        "How do I write SQL to calculate percentage using CASE and aggregations?",
        "How do I write SQL to categorize data using CASE WHEN?",
        # Subqueries
        "How do I write SQL using a subquery in WHERE clause?",
        "How do I write SQL using subquery in SELECT clause?",
        "How do I write SQL using subquery in FROM clause?",
        "How do I write SQL using correlated subquery?",
        "How do I write SQL using EXISTS with subquery?",
        "How do I write SQL using NOT EXISTS with subquery?",
        # Window functions (basic)
        "How do I write SQL using ROW_NUMBER() window function?",
        "How do I write SQL using RANK() window function?",
        "How do I write SQL using DENSE_RANK() window function?",
        "How do I write SQL using PARTITION BY in a window function?",
        "How do I write SQL using window function with ORDER BY?",
        "How do I write SQL to number rows within groups?",
        # Date operations (intermediate)
        "How do I write SQL to filter by date range?",
        "How do I write SQL to calculate date difference?",
        "How do I write SQL to add days to date?",
        "How do I write SQL to format date?",
        # UNION
        "How do I write SQL using UNION to combine results?",
        "How do I write SQL using UNION ALL?",
        # COALESCE and NULL handling
        "How do I write SQL using COALESCE to handle NULLs?",
        "How do I write SQL using IFNULL to replace NULL values?",
    ],
    "complex": [
        # Multiple CTEs (chaining)
        "How do I write SQL that chains two CTEs (WITH clauses) in sequence?",
        "How do I write SQL that chains three CTEs for multi-step data processing?",
        "How do I write SQL using multiple CTEs where second CTE uses first?",
        "How do I write SQL to chain CTEs for progressive data transformation?",
        "How do I write SQL using recursive CTE (WITH RECURSIVE)?",
        # CLEAN_NUMERIC
        "How do I write SQL using CLEAN_NUMERIC() function to clean dirty numeric data?",
        "How do I write SQL using CLEAN_NUMERIC() in WHERE clause?",
        "How do I write SQL using CLEAN_NUMERIC() with aggregations?",
        "How do I write SQL using CLEAN_NUMERIC() with CASE WHEN for data cleaning?",
        "How do I write SQL to clean and sum numeric values?",
        # Advanced window functions
        "How do I write SQL using LAG() window function to access previous row?",
        "How do I write SQL using LEAD() window function to access next row?",
        "How do I write SQL using LAG() with PARTITION BY?",
        "How do I write SQL using LEAD() with PARTITION BY?",
        "How do I write SQL using ROWS BETWEEN for custom window frame?",
        "How do I write SQL using ROWS BETWEEN UNBOUNDED PRECEDING and CURRENT ROW?",
        "How do I write SQL using window function with ROWS BETWEEN UNBOUNDED PRECEDING?",
        "How do I write SQL using ROWS BETWEEN to create sliding window?",
        "How do I write SQL using RANGE BETWEEN in window function?",
        "How do I write SQL using multiple window functions with different PARTITION BY?",
        "How do I write SQL using window function with multiple PARTITION BY columns?",
        "How do I write SQL to rank within groups using PARTITION BY and ORDER BY?",
        "How do I write SQL using window function to calculate running totals?",
        "How do I write SQL using window function to calculate cumulative sums?",
        # Moving averages and trends
        "How do I write SQL to calculate moving averages using window functions?",
        "How do I write SQL to calculate 3-month moving average?",
        "How do I write SQL to calculate 7-day moving average?",
        "How do I write SQL to calculate moving sum using window functions?",
        # Percentiles and statistics
        "How do I write SQL to calculate percentiles using window functions?",
        "How do I write SQL to calculate median using window functions?",
        "How do I write SQL to calculate quartiles?",
        # Complex combinations
        "How do I write SQL that combines CTE, window function, and aggregations?",
        "How do I write SQL using CTE with window function and GROUP BY?",
        "How do I write SQL to chain CTE with window function then aggregate?",
        "How do I write SQL using CTE, JOIN, and window function together?",
        "How do I write SQL to use CTE with CASE WHEN and aggregations?",
        # Pivoting (using CASE WHEN)
        "How do I write SQL to pivot data using CASE WHEN and GROUP BY?",
        "How do I write SQL to create pivot table with multiple categories?",
        "How do I write SQL to transpose rows to columns?",
        # Advanced aggregations
        "How do I write SQL using conditional aggregations with CASE WHEN?",
        "How do I write SQL to count records meeting multiple conditions?",
        "How do I write SQL to sum values conditionally?",
        # Complex filtering
        "How do I write SQL using subquery with window function?",
        "How do I write SQL to filter using window function results?",
        "How do I write SQL using HAVING with window function?",
        # Date operations (advanced)
        "How do I write SQL to calculate year-over-year growth?",
        "How do I write SQL to group by year and month?",
        "How do I write SQL to calculate days between dates?",
        "How do I write SQL to filter by last N months?",
    ]
}


def validate_sql_syntax(sql: str, create_test_db: bool = True) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate SQL SYNTAX ONLY - not column names or schema (syntax-focused training)
    
    Uses EXPLAIN to validate syntax structure without needing actual columns.
    This ensures training focuses on SQL syntax patterns, not specific schema.
    
    Returns:
        (is_valid, error_message, execution_result)
    """
    if not sql or not sql.strip():
        return False, "Empty SQL query", None
    
    # Clean SQL - take first statement only (handle multiple statements)
    sql_clean = sql.strip()
    
    # Split by semicolon and take first statement if multiple exist
    if ';' in sql_clean:
        parts = sql_clean.split(';')
        sql_clean = parts[0].strip()
        if len(parts) > 1:
            logger.warning(f"   Multiple SQL statements detected, using first one only")
    
    sql_clean = sql_clean.rstrip(';').strip()
    
    # Basic safety check
    sql_upper = sql_clean.upper()
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        return False, "Query must start with SELECT or WITH", None
    
    # Block dangerous keywords
    dangerous = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
    if any(kw in sql_upper for kw in dangerous):
        return False, f"Query contains dangerous keyword (only SELECT/WITH allowed)", None
    
    # SYNTAX-ONLY VALIDATION: Use EXPLAIN to check syntax without executing
    # This validates SQL structure without needing actual columns/tables
    conn = None
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # Create a dummy table for EXPLAIN to work with
        # (EXPLAIN needs at least a table reference, but we don't care about column names)
        cursor.execute("CREATE TABLE table_name (id INTEGER)")
        
        # Replace table/column names in SQL with dummy names for EXPLAIN
        # This allows EXPLAIN to validate syntax structure without schema dependency
        sql_for_explain = sql_clean
        
        # Replace common table name patterns with our dummy table
        import re
        # Replace "FROM table_name" or "FROM any_table" with "FROM table_name"
        sql_for_explain = re.sub(r'\bFROM\s+\w+', 'FROM table_name', sql_for_explain, flags=re.IGNORECASE)
        
        # For JOINs, replace table references
        sql_for_explain = re.sub(r'\bJOIN\s+\w+', 'JOIN table_name', sql_for_explain, flags=re.IGNORECASE)
        
        # Try EXPLAIN to validate syntax structure only
        try:
            cursor.execute(f"EXPLAIN {sql_for_explain}")
            results = cursor.fetchall()
            
            # If EXPLAIN succeeds, SQL syntax structure is valid
            # (Even if columns don't exist, EXPLAIN validates the query structure)
            return True, None, "SQL syntax structure is valid (EXPLAIN succeeded)"
        except sqlite3.Error as e:
            error_msg = str(e)
            # Filter out "no such column" errors - we only care about syntax errors
            if "no such column" in error_msg.lower() or "no such table" in error_msg.lower():
                # These are schema errors, not syntax errors - syntax is valid
                return True, None, "SQL syntax structure is valid (schema-independent check passed)"
            else:
                # Real syntax error (missing keyword, wrong structure, etc.)
                return False, f"SQL syntax error: {error_msg}", None
                
    except Exception as e:
        return False, f"Validation error: {str(e)}", None
    finally:
        if conn:
            conn.close()
    
    return False, "Unknown error", None


# Global variable to hold GGUF model for inference
_gguf_model: Optional[Any] = None


def _load_gguf_model(gguf_path: str) -> Optional[Any]:
    """Load GGUF model for inference"""
    global _gguf_model
    
    if _gguf_model is not None:
        return _gguf_model
    
    if not LLAMA_CPP_AVAILABLE:
        logger.warning("llama-cpp-python not available. Cannot load GGUF model.")
        return None
    
    if not Path(gguf_path).exists():
        logger.warning(f"GGUF file not found: {gguf_path}")
        return None
    
    try:
        logger.info(f"Loading GGUF model from: {gguf_path}")
        _gguf_model = Llama(
            model_path=gguf_path,
            n_ctx=2048,  # Context window
            n_threads=4,  # CPU threads
            verbose=False
        )
        logger.info("✅ GGUF model loaded successfully!")
        return _gguf_model
    except Exception as e:
        logger.error(f"Failed to load GGUF model: {e}")
        return None


def model_generate_text(tokenizer, model, prompt: str, max_tokens: int = 512, temperature: float = 0.7, use_gguf: bool = False, gguf_path: str = None) -> str:
    """Generate text using the model (for question generation and self-correction)"""
    
    # Try GGUF if enabled
    if use_gguf and gguf_path:
        gguf_model = _load_gguf_model(gguf_path)
        if gguf_model:
            try:
                response_obj = gguf_model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["<|im_end|>", "\n\n\n"],
                    echo=False
                )
                return response_obj["choices"][0]["text"].strip()
            except Exception as e:
                logger.warning(f"GGUF generation failed: {e}, falling back")
    
    # HuggingFace model generation (for inference only - set to eval mode)
    was_training = model.training
    model.eval()  # Set to eval mode for inference
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model_device = next(model.parameters()).device
    if model_device.type != device:
        model = model.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Restore training mode if it was training before
    if was_training:
        model.train()
    
    return response.strip()


def generate_challenge_question(difficulty: str = "basic", iteration: int = 1) -> Dict:
    """Generate a SQL question from the question bank (no external API)
    
    Difficulty levels:
    - basic: Simple SELECT, WHERE, basic aggregations
    - medium: JOINs, GROUP BY, CTEs, window functions
    - complex: Multiple CTEs, advanced window functions, CLEAN_NUMERIC
    """
    
    questions = SQL_QUESTION_BANK.get(difficulty, SQL_QUESTION_BANK['basic'])
    question = random.choice(questions)
    
    # Determine expected patterns based on difficulty
    if difficulty == "basic":
        patterns = ["SELECT", "WHERE", "aggregation"]
    elif difficulty == "medium":
        patterns = ["GROUP BY", "JOIN", "CTE", "window_function"]
    else:
        patterns = ["CTE_chaining", "window_function", "CLEAN_NUMERIC", "advanced"]
    
    return {
        "question": question,
        "difficulty": difficulty,
        "category": f"{difficulty}_query",
        "expected_patterns": patterns
    }


def estimate_token_size(question: str) -> int:
    """Estimate required tokens based on question complexity"""
    
    question_lower = question.lower()
    
    # Base tokens for simple SELECT
    base_tokens = 128
    
    # Complexity indicators that need more tokens
    if any(word in question_lower for word in ['cte', 'with', 'common table expression', 'chaining', 'multiple cte']):
        base_tokens += 128  # CTEs are multi-line
    
    if any(word in question_lower for word in ['window function', 'over', 'partition by', 'rows between', 'lag', 'lead', 'rank', 'row_number']):
        base_tokens += 96  # Window functions add complexity
    
    if any(word in question_lower for word in ['multiple', 'several', 'various', 'complex', 'advanced']):
        base_tokens += 64
    
    if any(word in question_lower for word in ['pivot', 'crosstab', 'case when', 'conditional']):
        base_tokens += 64  # CASE WHEN can be verbose
    
    if any(word in question_lower for word in ['join', 'multiple join', 'left join', 'inner join']):
        base_tokens += 48
    
    if any(word in question_lower for word in ['subquery', 'nested', 'correlated']):
        base_tokens += 64
    
    # Cap at reasonable maximum
    return min(base_tokens, 768)  # Max 768 tokens (safety limit)


def student_answer_question(tokenizer, model, question: str, use_gguf: bool = False, gguf_path: str = None) -> str:
    """Student model generates SQL answer
    
    Can use either HuggingFace model or GGUF model for inference
    """
    
    # Estimate required tokens based on question complexity
    estimated_tokens = estimate_token_size(question)
    logger.info(f"  🤖 Student model generating answer... (estimated: {estimated_tokens} tokens)")
    
    # Try GGUF model first if enabled
    if use_gguf and gguf_path:
        gguf_model = _load_gguf_model(gguf_path)
        if gguf_model:
            # Format prompt optimized for SQL text-to-SQL conversion
            prompt = f"""<|im_start|>system
You are an expert SQL generator. Convert natural language questions into valid SQLite SQL queries.

CRITICAL RULES:
1. Output ONLY SQL code - no explanations, no comments, no text
2. Start with WITH, SELECT, or CREATE
3. Use generic placeholder names: table_name, column_name, amount_column
4. Ensure valid SQLite syntax
5. Focus on correct SQL structure

Example:
User: "How do I count records where amount is greater than 100?"
Assistant: SELECT COUNT(*) FROM table_name WHERE amount_column > 100;

Now convert this question to SQL:
<|im_start|>user
{question}
<|im_start|>assistant
"""
            
            try:
                response_obj = gguf_model(
                    prompt,
                    max_tokens=estimated_tokens,
                    temperature=0.1,
                    stop=["<|im_end|>", "\n\n\n"],
                    echo=False
                )
                response = response_obj["choices"][0]["text"].strip()
                logger.info("  ✅ Used GGUF model for inference")
                
                # Extract SQL from response
                import re
                sql_match = re.search(r'(SELECT|WITH|CREATE|INSERT|UPDATE|DELETE)[\s\S]*', response, re.IGNORECASE)
                if sql_match:
                    response = sql_match.group(0).strip()
                    # Remove explanations after SQL
                    response = response.split('\n\n')[0].split('Note:')[0].strip()
                    return response
                return response
            except Exception as e:
                logger.warning(f"GGUF inference failed: {e}, falling back to HuggingFace model")
    
    # Fallback to HuggingFace model - optimized prompt for text-to-SQL
    messages = [
        {
            "role": "system", 
            "content": "You are an expert SQL generator. Convert natural language to valid SQLite SQL. Output ONLY SQL code starting with WITH, SELECT, or CREATE. Use generic placeholder names: table_name, column_name, amount_column. No explanations, no comments, just SQL."
        },
        {
            "role": "user", 
            "content": f"Convert this question to SQL:\n\n{question}\n\nGenerate ONLY the SQL query:"
        }
    ]
    
    # Use tokenizer's chat template if available, otherwise format manually
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback format for Qwen
        prompt = f"<|im_start|>system\n{messages[0]['content']}\n<|im_start|>user\n{messages[1]['content']}\n<|im_start|>assistant\n"
    
    # Set model to eval mode for inference (but preserve training state)
    was_training = model.training
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model_device = next(model.parameters()).device
    if model_device.type != device:
        model = model.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=estimated_tokens,  # Dynamic based on question complexity
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Restore training mode if it was training before
    if was_training:
        model.train()
    
    # Extract SQL - handle multiple formats
    import re
    
    # Method 1: Extract from ```sql code blocks
    sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
    if sql_match:
        sql = sql_match.group(1).strip()
        # Remove any trailing text after SQL
        sql = re.sub(r'\n.*$', '', sql, flags=re.MULTILINE)
        # Take first statement only (split on semicolon)
        if ';' in sql:
            sql = sql.split(';')[0].strip()
        return sql
    
    # Method 2: Extract SQL starting with SELECT, WITH, or CREATE (handle multi-line SQL)
    sql_match = re.search(r'(SELECT|WITH|CREATE|INSERT|UPDATE|DELETE)[\s\S]*', response, re.IGNORECASE)
    if sql_match:
        sql = sql_match.group(0).strip()
        # CRITICAL: Take first statement only - split on semicolon if multiple statements
        if ';' in sql:
            # Split and take first statement
            parts = sql.split(';')
            sql = parts[0].strip()
            if len(parts) > 1:
                logger.warning(f"  ⚠️  Multiple SQL statements detected, using first one only")
        
        # Split by double newline to separate SQL from explanations
        sql = sql.split('\n\n')[0]
        # Remove lines that start with explanation words
        lines = sql.split('\n')
        sql_lines = []
        stop_words = ['Note:', 'This', 'The query', 'Explanation', 'Here', 'You', 'It', 'For', 'The above']
        for line in lines:
            line_stripped = line.strip()
            # Stop if line starts with explanation word
            if any(line_stripped.startswith(word) for word in stop_words):
                break
            sql_lines.append(line)
        sql = '\n'.join(sql_lines).strip()
        # Remove trailing semicolon and any remaining explanation
        sql = re.sub(r';\s*(Note:|This|The|You|It|For).*$', '', sql, flags=re.IGNORECASE)
        return sql.strip()
    
    # Method 3: If response starts with SQL keywords, take first paragraph
    if re.match(r'^\s*(SELECT|WITH|CREATE)', response, re.IGNORECASE):
        sql = response.split('\n\n')[0].split('Note:')[0].split('This')[0]
        sql = sql.strip()
        # Take first statement only
        if ';' in sql:
            sql = sql.split(';')[0].strip()
        # Remove any trailing non-SQL text
        sql = re.sub(r'(\n|;)\s*[A-Z][a-z]+.*$', '', sql)
        return sql.strip()
    
    # Fallback: Return cleaned response (remove common prefixes)
    cleaned = re.sub(r'^(Here|Here is|The SQL|SQL query|Query):\s*', '', response, flags=re.IGNORECASE)
    cleaned = cleaned.split('\n\n')[0].strip()
    
    # Take first statement only
    if ';' in cleaned:
        cleaned = cleaned.split(';')[0].strip()
    
    # VALIDATION: Check if this is actually SQL, not just text
    if not re.match(r'^\s*(SELECT|WITH|CREATE|INSERT|UPDATE|DELETE)', cleaned, re.IGNORECASE):
        # Not SQL - return empty/invalid
        logger.warning(f"  ⚠️  Model generated non-SQL response: {cleaned[:50]}...")
        return ""  # Return empty to trigger teacher to provide correction
    
    return cleaned


def teacher_evaluate(answer: str, question: str, tokenizer=None, model=None, use_gguf: bool = False, gguf_path: str = None) -> Dict:
    """Self-correcting evaluation: Validate SQL via execution, then self-correct if needed (smolagents approach)
    
    Flow:
    1. Check if SQL is syntactically valid (execute on test DB)
    2. If invalid, use model to generate corrected SQL
    3. Re-validate corrected SQL
    4. Score based on syntax correctness and pattern matching
    """
    
    # Handle empty/invalid answers
    if not answer or not answer.strip():
        return {
            "correct": False,
            "score": 0,
            "feedback": "No SQL generated - model failed to produce code",
            "corrected_sql": None,
            "issues": ["No SQL code produced"]
        }
    
    # Step 1: Validate SQL SYNTAX ONLY (not column names - syntax-focused training)
    is_valid, error_msg, exec_result = validate_sql_syntax(answer, create_test_db=True)
    
    if is_valid:
        # SQL syntax structure is valid - check if it matches question requirements
        # Focus on SQL syntax patterns, not column names
        score = 90  # High score for valid SQL syntax
        feedback = "SQL syntax structure is valid"
        issues = []
        
        # Check if SQL demonstrates expected SYNTAX PATTERNS from question
        answer_upper = answer.upper()
        question_lower = question.lower()
        
        # Pattern matching - SYNTAX PATTERNS ONLY (not column-specific)
        if "group by" in question_lower and "GROUP BY" not in answer_upper:
            score -= 20
            issues.append("Missing GROUP BY syntax pattern")
        if "join" in question_lower and "JOIN" not in answer_upper:
            score -= 15
            issues.append("Missing JOIN syntax pattern")
        if "cte" in question_lower or "with" in question_lower:
            if "WITH" not in answer_upper:
                score -= 20
                issues.append("Missing CTE/WITH syntax pattern")
        if "window function" in question_lower or "over" in question_lower:
            if "OVER" not in answer_upper:
                score -= 20
                issues.append("Missing window function syntax pattern (OVER clause)")
        if "clean_numeric" in question_lower:
            if "CLEAN_NUMERIC" not in answer_upper:
                score -= 15
                issues.append("Missing CLEAN_NUMERIC function syntax")
        if "case when" in question_lower or "conditional" in question_lower:
            if "CASE" not in answer_upper or "WHEN" not in answer_upper:
                score -= 15
                issues.append("Missing CASE WHEN syntax pattern")
        
        if issues:
            feedback = f"SQL syntax is valid but missing expected syntax patterns: {', '.join(issues)}"
        else:
            feedback = "SQL syntax structure is valid and includes expected patterns"
        
        return {
            "correct": score >= 80,
            "score": max(score, 0),
            "feedback": feedback,
            "corrected_sql": None if score >= 80 else answer,  # Keep original if minor issues
            "issues": issues if issues else ["none"]
        }
    else:
        # SQL is invalid - use model to self-correct
        logger.info(f"   SQL validation failed: {error_msg}")
        logger.info(f"   Attempting self-correction...")
        
        if tokenizer and model:
            correction_prompt = f"""The following SQL query has an error. Fix it to make it valid SQLite syntax.

Question: {question}

Invalid SQL:
```sql
{answer}
```

Error: {error_msg}

Provide ONLY the corrected SQL query, no explanations. Start with SELECT or WITH:"""
            
            # Use model's chat template
            messages = [
                {"role": "system", "content": "You are an expert SQL generator. Fix SQL syntax errors. Output ONLY corrected SQL code."},
                {"role": "user", "content": correction_prompt}
            ]
            
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = f"<|im_start|>system\n{messages[0]['content']}\n<|im_start|>user\n{messages[1]['content']}\n<|im_start|>assistant\n"
            
            corrected_sql = model_generate_text(
                tokenizer, model, prompt_text, 
                max_tokens=512, temperature=0.1,
                use_gguf=use_gguf, gguf_path=gguf_path
            )
            
            # Extract SQL from corrected response
            sql_match = re.search(r'(SELECT|WITH|CREATE)[\s\S]*', corrected_sql, re.IGNORECASE)
            if sql_match:
                corrected_sql = sql_match.group(0).strip()
                corrected_sql = corrected_sql.split('\n\n')[0].split('Note:')[0].strip()
            
            # Re-validate corrected SQL
            is_corrected_valid, corrected_error, _ = validate_sql_syntax(corrected_sql, create_test_db=True)
            
            if is_corrected_valid:
                return {
                    "correct": True,
                    "score": 70,  # Lower score because it needed correction
                    "feedback": f"SQL corrected: {error_msg}. Corrected version is valid.",
                    "corrected_sql": corrected_sql,
                    "issues": [error_msg]
                }
            else:
                # Correction failed - return original with error info
                return {
                    "correct": False,
                    "score": 20,
                    "feedback": f"SQL syntax error: {error_msg}. Auto-correction also failed.",
                    "corrected_sql": corrected_sql if corrected_sql else answer,
                    "issues": [error_msg, "Auto-correction failed"]
                }
        else:
            # No model available for correction
            return {
                "correct": False,
                "score": 0,
                "feedback": f"SQL syntax error: {error_msg}",
                "corrected_sql": answer,  # Keep original as fallback
                "issues": [error_msg]
            }


def train_on_difficulty_level(model, tokenizer, difficulty: str, output_dir: Path, 
                                iteration: int, post_training_threshold: float = 90.0) -> Dict:
    """Train on a difficulty level: collect volume of examples, train, then test
    
    Strategy: Initially focus on VOLUME of training examples
    - Model needs sufficient data to learn from
    - Collect mistakes (non-correct answers) until we have enough
    - Then train and test the improved model
    
    Flow:
    1. Collect minimum volume of training examples (all mistakes)
    2. Train model on collected examples
    3. Test trained model with fresh questions
    4. If post-training accuracy >= threshold, done; else collect more and repeat
    
    Args:
        difficulty: 'basic' (only basic SQL for now)
        post_training_threshold: Target accuracy after training (default 90%)
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🎯 Training on {difficulty.upper()} SQL")
    logger.info(f"📊 Strategy: Collect volume of mistakes → Train → Test (target: {post_training_threshold}%)")
    logger.info(f"{'='*80}")
    
    min_training_examples = 50  # Increased minimum for better SQL learning
    batch_size = 25  # More questions per batch for volume
    max_total_questions = 150  # More total questions for comprehensive training
    test_questions = 15  # More test questions for better accuracy measurement
    
    all_training_examples = []  # Accumulate across batches
    total_questions_asked = 0
    
    for batch_num in range(1, (max_total_questions // batch_size) + 1):
        logger.info(f"\n📦 Batch {batch_num}: Collecting training examples...")
        logger.info(f"  Current dataset: {len(all_training_examples)} examples")
        logger.info(f"  Target: At least {min_training_examples} examples before first training")
        
        # Step 1: Collect training examples (focus on VOLUME of mistakes)
        batch_training_examples = []
        
        for q_num in range(1, batch_size + 1):
            total_questions_asked += 1
            
            if total_questions_asked > max_total_questions:
                break
                
            logger.info(f"\n  Question {total_questions_asked}/{max_total_questions} (Batch {batch_num}):")
            
            # Generate question from bank (no external API)
            try:
                challenge = generate_challenge_question(difficulty=difficulty)
                if not challenge:
                    logger.warning("  ⚠️  Failed to generate challenge, skipping")
                    continue
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to generate challenge: {e}, skipping")
                continue
            
            question = challenge['question']
            logger.info(f"  Question: {question}")
            
            # Student attempts answer (use GGUF if configured)
            use_gguf = CONFIG.get("use_gguf_for_inference", False)
            gguf_path = CONFIG.get("pretrained_gguf", None)
            student_answer = student_answer_question(tokenizer, model, question, use_gguf=use_gguf, gguf_path=gguf_path)
            logger.info(f"  Student answer: {student_answer[:100]}...")
            
            # Self-correcting evaluation (smolagents approach)
            try:
                evaluation = teacher_evaluate(
                    student_answer, question,
                    tokenizer=tokenizer, model=model,
                    use_gguf=use_gguf, gguf_path=gguf_path
                )
                score = evaluation.get('score', 0)
                is_correct = evaluation.get('correct', False)
            except Exception as e:
                logger.warning(f"  ⚠️  Evaluation failed: {e}, using default")
                evaluation = {"correct": False, "score": 0, "corrected_sql": student_answer}
                score = 0
                is_correct = False
            
            logger.info(f"  Score: {score}/100 {'✅' if is_correct else '❌'}")
            
            # Add ALL non-correct answers to training data
            # Volume matters - we need sufficient examples to learn from
            if not is_correct or score < 100:
                correct_answer = evaluation.get('corrected_sql') or student_answer
                
                if not correct_answer or not correct_answer.strip():
                    logger.warning(f"  ⚠️  No valid SQL available, skipping")
                    continue
                
                # Optimized format for SQL text-to-SQL conversion training
                example = {
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are an expert SQL generator. Convert natural language questions into valid SQLite SQL queries. Output ONLY SQL code. Use generic placeholder names: table_name, column_name, amount_column."
                        },
                        {
                            "role": "user", 
                            "content": question
                        },
                        {
                            "role": "assistant", 
                            "content": correct_answer
                        }
                    ]
                }
                batch_training_examples.append(example)
                all_training_examples.append(example)
                logger.info(f"  📝 Added to training data (total: {len(all_training_examples)} examples)")
            
            time.sleep(2)
        
        # Step 2: Train if we have minimum volume OR if this is a subsequent batch after training
        should_train = (len(all_training_examples) >= min_training_examples) or (batch_num > 1)
        
        if should_train and all_training_examples:
            logger.info(f"\n🔧 Training on {len(all_training_examples)} total examples...")
            temp_file = output_dir / f"temp_{difficulty}_batch_{batch_num}_iter_{iteration}.jsonl"
            with open(temp_file, 'w') as f:
                for ex in all_training_examples:
                    f.write(json.dumps(ex) + '\n')
            
            train_on_active_learning_data(model, tokenizer, [temp_file], iteration)
            temp_file.unlink()
            logger.info(f"  ✅ Training complete on {len(all_training_examples)} examples!")
        
        # Step 3: Test trained model (only if we've trained)
        if should_train and all_training_examples:
            logger.info(f"\n🧪 Testing trained model with {test_questions} fresh questions...")
            test_scores = []
            use_gguf = CONFIG.get("use_gguf_for_inference", False)
            gguf_path = CONFIG.get("pretrained_gguf", None)
            
            for test_num in range(1, test_questions + 1):
                challenge = generate_challenge_question(difficulty=difficulty)
                if not challenge:
                    continue
                
                question = challenge['question']
                student_answer = student_answer_question(tokenizer, model, question, use_gguf=use_gguf, gguf_path=gguf_path)
                evaluation = teacher_evaluate(student_answer, question, tokenizer=tokenizer, model=model, use_gguf=use_gguf, gguf_path=gguf_path)
                
                score = evaluation.get('score', 0)
                is_correct = evaluation.get('correct', False)
                test_scores.append(score)
                
                logger.info(f"  Test {test_num}/{test_questions}: {score}/100 {'✅' if is_correct else '❌'}")
                time.sleep(2)
            
            # Step 4: Check if threshold met
            if test_scores:
                post_training_accuracy = sum(test_scores) / len(test_scores)
                logger.info(f"\n📊 Post-Training Accuracy: {post_training_accuracy:.1f}% (target: {post_training_threshold}%)")
                
                if post_training_accuracy >= post_training_threshold:
                    logger.info(f"  ✅ Threshold reached! Model achieved {post_training_accuracy:.1f}% >= {post_training_threshold}%")
                    return {
                        "difficulty": difficulty,
                        "batches_collected": batch_num,
                        "total_questions": total_questions_asked,
                        "total_training_examples": len(all_training_examples),
                        "post_training_accuracy": post_training_accuracy,
                        "threshold_met": True,
                        "test_scores": test_scores
                    }
                else:
                    logger.info(f"  ⚠️  Not yet: {post_training_accuracy:.1f}% < {post_training_threshold}%")
                    logger.info(f"  📦 Collecting more examples to improve...")
        
        if total_questions_asked >= max_total_questions:
            break
    
    # If max questions reached
    if all_training_examples:
        # Final training and test
        logger.info(f"\n🔧 Final training on {len(all_training_examples)} examples...")
        temp_file = output_dir / f"temp_{difficulty}_final_iter_{iteration}.jsonl"
        with open(temp_file, 'w') as f:
            for ex in all_training_examples:
                f.write(json.dumps(ex) + '\n')
        
        train_on_active_learning_data(model, tokenizer, [temp_file], iteration)
        temp_file.unlink()
        
        # Final test
        logger.info(f"\n🧪 Final test with {test_questions} questions...")
        test_scores = []
        use_gguf = CONFIG.get("use_gguf_for_inference", False)
        gguf_path = CONFIG.get("pretrained_gguf", None)
        for test_num in range(1, test_questions + 1):
            challenge = generate_challenge_question(difficulty=difficulty)
            if not challenge:
                continue
            question = challenge['question']
            student_answer = student_answer_question(tokenizer, model, question, use_gguf=use_gguf, gguf_path=gguf_path)
            evaluation = teacher_evaluate(student_answer, question, tokenizer=tokenizer, model=model, use_gguf=use_gguf, gguf_path=gguf_path)
            test_scores.append(evaluation.get('score', 0))
            time.sleep(2)
        
        final_accuracy = sum(test_scores) / len(test_scores) if test_scores else 0
    else:
        final_accuracy = 0
        test_scores = []
    
    logger.warning(f"\n⚠️  Max questions reached. Final accuracy: {final_accuracy:.1f}%")
    
    return {
        "difficulty": difficulty,
        "batches_collected": batch_num,
        "total_questions": total_questions_asked,
        "total_training_examples": len(all_training_examples),
        "post_training_accuracy": final_accuracy,
        "threshold_met": final_accuracy >= post_training_threshold,
        "test_scores": test_scores
    }


def active_learning_iteration(model, tokenizer, iteration: int, output_dir: Path) -> Dict:
    """One active learning iteration - focus on BASIC SQL only
    
    Flow:
    1. Collect training examples (mistakes)
    2. Train model on examples
    3. Test trained model with fresh questions
    4. Repeat until post-training accuracy >= 90%
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Iteration {iteration}: Basic SQL Training (Post-Training Target: 90%)")
    logger.info(f"{'='*80}")
    
    # Only focus on basic SQL for now
    result = train_on_difficulty_level(
        model, tokenizer, 
        difficulty="basic",
        output_dir=output_dir,
        iteration=iteration,
        post_training_threshold=90.0
    )
    
    logger.info(f"\n📊 Training Summary:")
    logger.info(f"  Batches trained: {result['batches_trained']}")
    logger.info(f"  Total examples: {result['total_examples']}")
    logger.info(f"  Post-training accuracy: {result['post_training_accuracy']:.1f}%")
    logger.info(f"  Threshold met: {'✅' if result['threshold_met'] else '⚠️'}")
    
    return {
        "examples": result["total_examples"],
        "post_training_accuracy": result["post_training_accuracy"],
        "threshold_met": result["threshold_met"],
        "difficulty_results": [result]
    }


def train_on_active_learning_data(model, tokenizer, data_files: List[Path], iteration: int):
    """Fine-tune on active learning data (mistakes corrected by teacher)"""
    
    if not data_files:
        logger.warning("No data files to train on")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Training on {len(data_files)} active learning data files")
    logger.info(f"{'='*80}")
    
    # Load active learning data
    dataset = load_dataset('json', data_files=[str(f) for f in data_files], split='train')
    
    def format_prompt(example):
        if "messages" not in example:
            return {"text": ""}
        
        messages = example["messages"]
        text = ""
        
        # Use Qwen chat template format
        for msg in messages:
            if msg["role"] == "system":
                text += f"<|im_start|>system\n{msg['content']}\n<|im_end|>\n"
            elif msg["role"] == "user":
                text += f"<|im_start|>user\n{msg['content']}\n<|im_end|>\n"
            elif msg["role"] == "assistant":
                text += f"<|im_start|>assistant\n{msg['content']}\n<|im_end|>\n"
        
        return {"text": text}
    
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=CONFIG["max_length"],
            padding=False,
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    # Ensure data collator properly handles the data format
    # Verify that inputs will have proper format
    if len(tokenized) > 0:
        sample = tokenized[0]
        logger.info(f"Sample input keys: {sample.keys()}")
        logger.info(f"Sample input_ids shape: {sample.get('input_ids', 'N/A')}")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Optimize for Mac M4 MPS
    use_bf16 = (device == "mps")  # bfloat16 works better on MPS than fp16
    use_fp16 = (device == "cuda")
    
    # Adjust gradient accumulation for memory efficiency
    effective_batch_size = CONFIG["batch_size"]
    micro_batch = CONFIG["micro_batch_size"]
    grad_accumulation = max(1, effective_batch_size // micro_batch)
    
    training_args = TrainingArguments(
        output_dir=str(CONFIG["checkpoint_dir"]) + f"_iter_{iteration}",
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,  # Keep only 2 checkpoints to save space
        fp16=use_fp16,
        bf16=use_bf16,  # Use bfloat16 on MPS for better performance
        logging_dir=str(Path(CONFIG["output_dir"]) / "logs"),
        report_to="none",
        optim="adamw_torch",
        gradient_checkpointing=False,  # Disable for MPS compatibility (causes gradient issues)
        dataloader_num_workers=0,  # MPS works better with 0 workers
        dataloader_pin_memory=False,  # Disable pin_memory for MPS
        remove_unused_columns=False,
        max_steps=-1,  # Use epochs instead
        lr_scheduler_type="cosine",  # Better learning rate schedule
        weight_decay=0.01,  # L2 regularization for better generalization
        max_grad_norm=1.0,  # Gradient clipping to prevent memory spikes
    )
    
    logger.info(f"📊 Training config: batch_size={micro_batch}, gradient_accumulation={grad_accumulation}, effective_batch={micro_batch * grad_accumulation}")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    
    # Ensure model is in training mode and LoRA adapters are trainable
    model.train()
    model.config.use_cache = False
    
    # Clear MPS cache before training
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Explicitly enable gradients for all LoRA parameters
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and not param.requires_grad:
            param.requires_grad = True
            logger.warning(f"   Enabled gradients for {name}")
    
    # Verify that LoRA parameters require gradients
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        logger.error("❌ No trainable parameters found! LoRA adapters may not be configured correctly.")
        # Print parameter names for debugging
        logger.error("Parameter names:")
        for name, param in model.named_parameters():
            logger.error(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
        raise RuntimeError("Model has no trainable parameters. Check LoRA configuration.")
    
    logger.info(f"✅ Training with {trainable_params:,} trainable parameters")
    
    try:
        trainer.train()
    finally:
        # Clear MPS cache after training
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    logger.info("✅ Training on active learning data complete!")


def main():
    """Main training pipeline"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=5, help='Number of active learning iterations')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / "active_learning_data"
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*80)
    logger.info("🎓 Antiks-v1 Self-Correcting Active Learning Pipeline")
    logger.info("="*80)
    logger.info(f"Approach: Self-correcting SQL validation (smolagents-style)")
    logger.info(f"Student: {CONFIG['base_model']}")
    logger.info(f"Model Name: {CONFIG['model_name']}")
    logger.info(f"Active Learning Iterations: {args.iterations}")
    logger.info("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # Load tokenizer and model
    # Resolve output_dir - try both project root and script directory
    output_path = Path(CONFIG["output_dir"])
    if output_path.is_absolute():
        final_output_dir = output_path
    elif (Path.cwd() / output_path / "adapter_config.json").exists():
        # Model in project root
        final_output_dir = Path.cwd() / output_path
    else:
        # Model in script directory (relative path)
        final_output_dir = (script_dir / output_path).resolve()
    
    # Check if trained model exists (resume training)
    if final_output_dir.exists() and (final_output_dir / "adapter_config.json").exists():
        logger.info("\n🔄 Found existing trained model - RESUMING training!")
        logger.info(f"Loading from: {final_output_dir}")
        
        tokenizer = AutoTokenizer.from_pretrained(str(final_output_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load base model with optimized dtype for M4
        if device == "mps":
            dtype = torch.bfloat16
        elif device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
            
        base_model = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model"],
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        
        if device == "mps":
            base_model = base_model.to("mps")
            # MPS doesn't have empty_cache(), memory is managed automatically
        
        # Load LoRA adapters from previous training
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, str(final_output_dir))
        logger.info("✅ Loaded previous training weights - continuing from where you left off!")
        
    else:
        logger.info("\n🚀 Starting fresh training from base model...")
        logger.info(f"No existing model found at: {final_output_dir}")
        
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Optimize dtype for Mac M4 MPS
        if device == "mps":
            dtype = torch.bfloat16  # bfloat16 is optimal for MPS
        elif device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model"],
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        
        if device == "mps":
            model = model.to("mps")
            # MPS doesn't have empty_cache(), memory is managed automatically
        
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
        
        # Train on new examples if any
        if iteration_files:
            train_on_active_learning_data(model, tokenizer, iteration_files, iteration)
            iteration_files = []  # Clear after training
    
    # Final save
    final_output_dir = (script_dir / CONFIG["output_dir"]).resolve()
    final_output_dir.mkdir(exist_ok=True, parents=True)
    trainer = Trainer(model=model)
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    
    logger.info("\n" + "="*80)
    logger.info("✅ Antiks-v1 Training Complete!")
    logger.info("="*80)
    logger.info(f"Total new examples generated: {total_new_examples}")
    logger.info(f"Model saved to: {final_output_dir}")
    logger.info("\nNext: python training/deploy_to_ollama.sh")


if __name__ == "__main__":
    main()

