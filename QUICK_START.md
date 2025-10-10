# Quick Start Guide - Cohere Multilingual RAG

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Setup
```bash
python test_cohere.py
```
Expected: `✅ All Cohere embedding tests passed!`

### Step 3: Run the Application
```bash
streamlit run src/rag_agent.py
```

## 🌍 Using Multilingual Embeddings

### In the Streamlit UI:

1. **Select Embedding Model** (Sidebar)
   - Choose: `cohere/embed-multilingual-v3.0` ⭐ (Recommended)
   - Alternative: `cohere/embed-multilingual-light-v3.0` (Faster)

2. **Upload Documents** (Any Language!)
   - 📄 PDFs (with tables, images, complex layouts)
   - 📊 Excel files
   - 🖼️ Images (with OCR and vision AI)

3. **Ask Questions** (Any Language!)
   - Query in the same language as documents
   - Query in different language (cross-lingual search)
   - Mix and match!

## 🎯 Model Selection Guide

| Model | Use Case | Dimensions | Speed |
|-------|----------|------------|-------|
| `cohere/embed-multilingual-v3.0` | Best quality, many languages | 1024 | Moderate |
| `cohere/embed-multilingual-light-v3.0` | Fast, many languages | 384 | Fast |
| `cohere/embed-english-v3.0` | Best English quality | 1024 | Moderate |
| `nomic-ai/nomic-embed-text-v1.5` | Local, English only | 768 | Fast |

## 💡 Example Scenarios

### Scenario 1: International Business
```
Documents: French contract + Spanish invoice + German email
Question: "What are the payment deadlines?" (English)
Result: Finds relevant info across all languages! ✨
```

### Scenario 2: Academic Research
```
Documents: Papers in Chinese, Japanese, Korean
Question: "研究方法是什么?" (What's the methodology? in Chinese)
Result: Cross-references all papers! 📚
```

### Scenario 3: Customer Support
```
Documents: Support tickets in Arabic, Hindi, Portuguese
Question: "ما هي المشاكل الشائعة؟" (What are common issues? in Arabic)
Result: Aggregates insights from all tickets! 🎫
```

## ⚙️ Advanced Features

### Query Decomposition
- ✅ Enabled by default
- Breaks complex queries into sub-queries
- Better retrieval for multi-part questions

### Vision AI for Images
- 🔍 Groq: Cloud-based, fast
- 🏠 Ollama: Local, no rate limits
- Automatically describes charts, diagrams, photos

### Vector Databases
- **ChromaDB**: Easy, in-memory, good for experimentation
- **FAISS**: Fast, scalable, good for production

## 🔧 Configuration Options (Sidebar)

- **Vector Database**: ChromaDB / FAISS
- **Embedding Model**: Select from local or Cohere models
- **LLM Provider**: Cerebras / Groq / Ollama
- **LLM Model**: Various options per provider
- **Chunk Size**: 200-2000 characters
- **Chunk Overlap**: 0-500 characters
- **Top K**: Number of retrieved chunks (1-10)
- **Use Vision Model**: Toggle image understanding
- **Use Query Decomposition**: Toggle complex query handling

## 📊 Monitoring

### Check Embedding Status
- Look for: "✅ Stored X chunks from Y items"
- Shows successful processing

### View Retrieved Context
- Click "📖 Retrieved Context" in chat responses
- See source documents, pages, and similarity scores

### Response Time
- Shown in chat: "⏱️ Response time: X.XXs"
- Monitor performance

## 🆘 Common Issues

### Issue: Slow embedding generation
**Solution**: Switch to light model (`embed-multilingual-light-v3.0`)

### Issue: Out of memory
**Solution**:
- Reduce chunk size to 500-800
- Use ChromaDB instead of FAISS
- Process fewer documents at once

### Issue: Poor retrieval quality
**Solution**:
- Increase Top K to 5-7
- Enable query decomposition
- Use full model instead of light

### Issue: Language not recognized
**Solution**:
- Ensure using multilingual model
- Cohere supports 100+ languages automatically

## 📈 Best Practices

1. **Choose the right model**
   - Multilingual docs → `embed-multilingual-v3.0`
   - English only → `embed-english-v3.0` or local models
   - Speed priority → light models

2. **Optimize chunk sizes**
   - Tables: Larger chunks (1500-2000)
   - Regular text: Medium chunks (800-1000)
   - Short docs: Smaller chunks (500-700)

3. **Use appropriate overlap**
   - High overlap (300-500): Better context preservation
   - Low overlap (100-200): Faster processing, less redundancy

4. **Leverage query decomposition**
   - Enable for complex, multi-part questions
   - Disable for simple, direct queries

## 🎓 Learn More

- Full documentation: [COHERE_INTEGRATION.md](COHERE_INTEGRATION.md)
- Cohere docs: https://docs.cohere.com/docs/embeddings
- Test script: `python test_cohere.py`

---

**Happy multilingual document chatting! 🌍✨**
