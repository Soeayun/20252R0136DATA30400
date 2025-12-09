# LLM Refinement Setup Guide

## 🔑 API Key Configuration

### Step 1: Install python-dotenv
```bash
pip install python-dotenv
```

### Step 2: Create .env file
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# .env file content:
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

### Step 3: Verify Setup
```bash
# Check if .env is properly ignored by git
git status

# .env should NOT appear in the output (it's gitignored)
```

## 🚀 Usage

### Run LLM Refinement
```bash
# Make sure you've run main.py first to generate ambiguous_doc_ids
python main.py

# Then run LLM refinement
python run_llm_refinement.py
```

## 📊 Expected Output

```
[1] Loading data...
  - Loaded 29,487 training documents

[2] Loading existing core classes and ambiguous doc IDs...
  - Loaded core classes for 29,487 documents
  - Loaded candidates for 29,487 documents
  - Loaded 10,679 ambiguous doc IDs (ratio ≤ 2)

[3] Running LLM refinement...
  Settings:
    - Max API calls: 1,000
    - Batch size: 10 documents per call
    - Parallel calls: 50 (async processing)
    - Target: 10,679 ambiguous docs (ratio ≤ 2)
    - Task: Select 0-3 true core classes from up to 10 candidates
    - Expected API calls: 1068
    - Expected speedup: ~50x faster with parallel processing

⏸️  Press Enter to start LLM refinement (this will use API credits)...

[Processing...]
⏱️  Processing time: 21.4s (46.7 batches/sec)

✅ Refinement Complete!
   API calls made: 1000
   Documents refined: 10,000
   Average time per call: 0.02s
```

## ⚠️ Important Notes

1. **Cost Estimation:**
   - Model: GPT-4o-mini
   - ~1,000 API calls
   - Estimated cost: ~$0.50-1.00
   
2. **API Call Limit:**
   - Budget: 1,000 calls
   - Can process: 10,000 documents
   - Coverage: 93.6% of ambiguous docs

3. **Security:**
   - Never commit `.env` file to git
   - `.env` is automatically gitignored
   - Share `.env.example` instead

## 🔧 Troubleshooting

### Error: "OPENAI_API_KEY not found"
```bash
# Make sure .env file exists
ls -la .env

# Check .env content (should have OPENAI_API_KEY=sk-...)
cat .env

# Verify python-dotenv is installed
pip show python-dotenv
```

### Error: "Invalid API key"
- Check if API key is correctly copied (no extra spaces)
- Verify key starts with `sk-`
- Test on OpenAI website: https://platform.openai.com/api-keys
