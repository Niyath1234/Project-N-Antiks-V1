#!/usr/bin/env python3
"""
Comprehensive tests for the annual enterprise survey CSV
Tests both base and fine-tuned models
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from sql_runner import create_connection, register_numeric_udfs
from data_loader import load_csv_to_sqlite
import ollama


# Real test cases based on your CSV schema
CSV_TEST_CASES = [
    {
        "question": "What is the total value?",
        "expected_concepts": ["SUM", "CLEAN_NUMERIC", "Value"],
        "complexity": "basic",
        "category": "aggregation"
    },
    {
        "question": "Show total value by Variable_category",
        "expected_concepts": ["SUM", "GROUP BY", "CLEAN_NUMERIC"],
        "complexity": "intermediate",
        "category": "grouping"
    },
    {
        "question": "Get top 10 Variable_category by total value",
        "expected_concepts": ["SUM", "ORDER BY", "LIMIT", "DESC"],
        "complexity": "intermediate",
        "category": "ranking"
    },
    {
        "question": "Show total value in crores",
        "expected_concepts": ["SUM", "division", "10000000"],
        "complexity": "intermediate",
        "category": "unit_conversion"
    },
    {
        "question": "Variable_category wise value with Year columns",
        "expected_concepts": ["CASE", "WHEN", "Year", "GROUP BY"],
        "complexity": "advanced",
        "category": "pivot"
    },
    {
        "question": "Show total by Industry_name_NZSIOC",
        "expected_concepts": ["GROUP BY", "Industry_name_NZSIOC"],
        "complexity": "intermediate",
        "category": "grouping"
    },
    {
        "question": "What are the top 5 Variable_category?",
        "expected_concepts": ["SUM", "ORDER BY", "LIMIT"],
        "complexity": "intermediate",
        "category": "ranking"
    },
    {
        "question": "Show industry wise breakdown with total in millions",
        "expected_concepts": ["GROUP BY", "division", "1000000"],
        "complexity": "intermediate",
        "category": "grouping_unit"
    },
    {
        "question": "Give me Variable_name wise value by Year",
        "expected_concepts": ["GROUP BY", "Variable_name", "Year"],
        "complexity": "intermediate",
        "category": "multi_grouping"
    },
    {
        "question": "Show percentage of each Variable_category",
        "expected_concepts": ["SUM", "percentage", "division"],
        "complexity": "advanced",
        "category": "percentage_calc"
    }
]


def test_model_performance(model_name: str, csv_path: str) -> Dict:
    """Test a model and return detailed results"""
    
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}\n")
    
    # Setup database
    conn = create_connection(":memory:")
    register_numeric_udfs(conn)
    table_name, cols = load_csv_to_sqlite(conn, csv_path, table_name='dataset')
    
    # Build schema context
    schema_text = f"Table: {table_name}\nColumns: {', '.join(cols[:10])}"
    if len(cols) > 10:
        schema_text += f"... (+{len(cols)-10} more)"
    
    print(f"Schema: {schema_text}")
    print()
    
    results = []
    
    for i, test in enumerate(CSV_TEST_CASES, 1):
        print(f"Test {i}/{len(CSV_TEST_CASES)}: {test['question']}")
        
        try:
            # Generate SQL using the base_v2 approach
            prompt = f"""You are a SQL expert for this dataset.

Table: {table_name}
Columns: {', '.join(cols)}

Key rules:
- Always use CLEAN_NUMERIC(Value) for the Value column
- Generate safe, read-only SELECT queries only
- For pivots, use CASE WHEN statements (SQLite has no PIVOT)
- Return ONLY the SQL query, no explanations

Question: {test['question']}"""

            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
            sql_raw = response['message']['content'].strip()
            
            # Extract SQL
            import re
            match = re.search(r"```sql\n([\s\S]*?)```", sql_raw, re.DOTALL | re.IGNORECASE)
            sql = match.group(1).strip() if match else sql_raw.strip()
            
            # Remove comments after semicolons
            sql = sql.split(';')[0].strip()
            if not sql.endswith(';'):
                sql += ';'
            
            # Test syntax
            syntax_valid = False
            syntax_error = ""
            try:
                cursor = conn.cursor()
                cursor.execute(f"EXPLAIN {sql}")
                cursor.fetchall()
                syntax_valid = True
            except Exception as e:
                syntax_error = str(e)[:100]
            
            # Test execution
            execution_valid = False
            exec_error = ""
            result_count = 0
            try:
                if syntax_valid:
                    cursor = conn.cursor()
                    cursor.execute(sql)
                    results_data = cursor.fetchall()
                    result_count = len(results_data)
                    execution_valid = True
            except Exception as e:
                exec_error = str(e)[:100]
            
            # Check for required keywords
            sql_upper = sql.upper()
            keywords_found = [kw for kw in test['expected_concepts'] if kw.upper() in sql_upper]
            keyword_score = len(keywords_found) / len(test['expected_concepts']) if test['expected_concepts'] else 1.0
            
            # Check CLEAN_NUMERIC usage (critical)
            uses_clean_numeric = "CLEAN_NUMERIC" in sql_upper
            is_basic_agg = test['category'] in ['aggregation', 'grouping']
            
            # Calculate score
            score = 0
            if syntax_valid:
                score += 0.4
            if execution_valid:
                score += 0.4
            score += keyword_score * 0.15
            if uses_clean_numeric or not is_basic_agg:
                score += 0.05  # Bonus for CLEAN_NUMERIC or complex queries that might not need it
            
            score_pct = score * 100
            
            # Status icon
            if score_pct >= 80:
                status = "✅"
            elif score_pct >= 50:
                status = "⚠️"
            else:
                status = "❌"
            
            print(f"  {status} Score: {score_pct:.0f}% | Syntax: {'✓' if syntax_valid else '✗'} | Execute: {'✓' if execution_valid else '✗'} | CleanNum: {'✓' if uses_clean_numeric else '✗'}")
            
            if not syntax_valid and syntax_error:
                print(f"    Error: {syntax_error[:60]}")
            if not execution_valid and exec_error:
                print(f"    Exec Error: {exec_error[:60]}")
            
            results.append({
                "question": test['question'],
                "category": test['category'],
                "complexity": test['complexity'],
                "score": score_pct,
                "syntax_valid": syntax_valid,
                "execution_valid": execution_valid,
                "keyword_score": keyword_score * 100,
                "uses_clean_numeric": uses_clean_numeric,
                "result_count": result_count,
                "keywords_found": keywords_found
            })
            
        except Exception as e:
            print(f"  ❌ FAILED: {str(e)[:80]}")
            results.append({
                "question": test['question'],
                "category": test['category'],
                "complexity": test['complexity'],
                "score": 0,
                "syntax_valid": False,
                "execution_valid": False,
                "keyword_score": 0,
                "uses_clean_numeric": False,
                "result_count": 0,
                "keywords_found": []
            })
    
    conn.close()
    
    # Calculate summary
    avg_score = sum(r["score"] for r in results) / len(results)
    syntax_rate = sum(1 for r in results if r["syntax_valid"]) / len(results) * 100
    exec_rate = sum(1 for r in results if r["execution_valid"]) / len(results) * 100
    clean_num_rate = sum(1 for r in results if r["uses_clean_numeric"]) / len(results) * 100
    
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Average Score: {avg_score:.1f}%")
    print(f"Syntax Pass: {syntax_rate:.1f}%")
    print(f"Execution Success: {exec_rate:.1f}%")
    print(f"CLEAN_NUMERIC Usage: {clean_num_rate:.1f}%")
    
    # By category
    print(f"\nBy Category:")
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r['score'])
    
    for cat, scores in sorted(categories.items()):
        avg = sum(scores) / len(scores)
        print(f"  {cat.replace('_', ' ').title():<25}: {avg:.1f}%")
    
    # By complexity
    print(f"\nBy Complexity:")
    complexities = {}
    for r in results:
        comp = r['complexity']
        if comp not in complexities:
            complexities[comp] = []
        complexities[comp].append(r['score'])
    
    for comp, scores in sorted(complexities.items()):
        avg = sum(scores) / len(scores)
        print(f"  {comp.capitalize():<25}: {avg:.1f}%")
    
    return {
        "model": model_name,
        "avg_score": avg_score,
        "syntax_rate": syntax_rate,
        "exec_rate": exec_rate,
        "clean_num_rate": clean_num_rate,
        "results": results
    }


def main():
    csv_path = os.environ.get('CSV_PATH', '/Users/niyathnair/Downloads/annual-enterprise-survey-2024-financial-year-provisional.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        print("Set CSV_PATH environment variable")
        return
    
    print("🧪 Comprehensive SQL Model Testing")
    print(f"CSV: annual-enterprise-survey-2024-financial-year-provisional.csv")
    print(f"Test cases: {len(CSV_TEST_CASES)}")
    
    # Test both models
    models = ["llama3.2", "analytics-sql"]
    all_results = {}
    
    for model_name in models:
        try:
            results = test_model_performance(model_name, csv_path)
            all_results[model_name] = results
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            continue
    
    # Comparison if both models were tested
    if len(all_results) >= 2:
        print("\n\n" + "="*80)
        print("📊 SIDE-BY-SIDE COMPARISON")
        print("="*80)
        
        model_names = list(all_results.keys())
        base_name = model_names[0]
        tuned_name = model_names[1] if len(model_names) > 1 else None
        
        print(f"\n{'Metric':<35} | {'Base':<20} | {'Fine-tuned':<20} | {'Delta'}")
        print("-" * 80)
        
        base_results = all_results[base_name]
        tuned_results = all_results[tuned_name] if tuned_name else None
        
        metrics = [
            ("Average Score", "avg_score", "%"),
            ("Syntax Pass Rate", "syntax_rate", "%"),
            ("Execution Success", "exec_rate", "%"),
            ("CLEAN_NUMERIC Usage", "clean_num_rate", "%")
        ]
        
        for metric_name, field, unit in metrics:
            base_val = base_results[field]
            tuned_val = tuned_results[field] if tuned_results else 0
            delta = tuned_val - base_val
            sign = "+" if delta >= 0 else ""
            
            print(f"{metric_name:<35} | {base_val:<20.1f}{unit} | {tuned_val:<20.1f}{unit} | {sign}{delta:.1f}{unit}")
        
        if tuned_results:
            improvement = ((tuned_results['avg_score'] - base_results['avg_score']) / base_results['avg_score'] * 100) if base_results['avg_score'] > 0 else 0
            print(f"\n🎯 Overall Improvement: {improvement:+.1f}%")
            
            if improvement > 0:
                print("✅ Fine-tuning improved performance!")
            elif improvement < -5:
                print("❌ Fine-tuning decreased performance significantly")
            else:
                print("⚠️  Fine-tuning shows minimal change")
        
        print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()

