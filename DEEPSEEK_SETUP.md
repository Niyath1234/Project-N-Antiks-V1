# DeepSeek API Integration Setup

## Overview
Successfully integrated DeepSeek Chat v3 as the teacher model for the SQL analytics training pipeline via OpenRouter API.

## Configuration

### API Details
- **Provider**: OpenRouter
- **Model**: `deepseek/deepseek-chat-v3-0324`
- **API Key**: `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`
- **Endpoint**: `https://openrouter.ai/api/v1/chat/completions`

### Files Modified/Created
1. ✅ **teacher_pipeline.py** - Updated to use OpenRouter API
2. ✅ **requirements.txt** - Added `requests>=2.31.0`
3. ✅ **test_deepseek.py** - Simple test script
4. ✅ **deepseek_cli.py** - Interactive CLI for testing
5. ✅ **DEEPSEEK_SETUP.md** - This documentation

## Usage

### Interactive CLI
```bash
# Activate venv
source venv/bin/activate

# Run interactive chat
python3 deepseek_cli.py
```

### Test Script
```bash
python3 test_deepseek.py
```

### Teacher Pipeline
```bash
# Run teacher-student training with DeepSeek validation
python3 training/teacher_pipeline.py --iterations 3
```

### Programmatic Usage
```python
from training.teacher_pipeline import query_deepseek

response = query_deepseek("Your question here")
print(response)
```

## Verification

### Basic Query Test ✅
- Question: "What is 2+2?"
- Response: Correct answer received

### SQL Validation Test ✅
- Question: "How many records are there?"
- SQL: `SELECT COUNT(*) FROM records`
- Validation: `{'valid': True, 'score': 100, 'feedback': 'CORRECT'}`

## Key Features

### Teacher Pipeline Functions
1. **query_deepseek()** - Base API interaction
2. **validate_sql_deepseek()** - SQL validation with structured feedback
3. **generate_hard_sql()** - Generate challenging SQL examples
4. **teacher_student_iteration()** - Complete iteration workflow

### CLI Features
- Interactive chat interface
- Commands: `exit`, `quit`, `q`
- Real-time API responses
- Clean formatting

## API Specifications

### Request Format
```python
{
    "model": "deepseek/deepseek-chat-v3-0324",
    "messages": [
        {"role": "user", "content": "your question"}
    ],
    "temperature": 0.1  # For consistent validation
}
```

### Headers
```
Authorization: Bearer sk-or-v1-...
Content-Type: application/json
HTTP-Referer: https://github.com/Niyath1234
X-Title: Python_NN_Project
```

## Next Steps

1. **Run full teacher-student pipeline**:
   ```bash
   python3 training/teacher_pipeline.py --iterations 5
   ```

2. **Combine training data**:
   ```bash
   cat training/train_data/*.jsonl training/validated_data/*.jsonl > all_data.jsonl
   ```

3. **Train model**:
   ```bash
   python3 training/train_simple.py
   ```

## Troubleshooting

### API Errors
- Verify API key is correctly set
- Check OpenRouter account balance
- Review rate limits

### Import Errors
- Ensure venv is activated: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

### JSON Parsing Errors
- DeepSeek responses are parsed with regex extraction
- Validation includes fallback error handling

## Credits
- **DeepSeek**: https://www.deepseek.com
- **OpenRouter**: https://openrouter.ai
- Integration completed successfully! 🎉

