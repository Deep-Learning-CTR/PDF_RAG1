# Cohere Multilingual Embeddings Integration Guide

## Overview

Your RAG system now supports **Cohere's multilingual embedding models**, enabling you to process documents and queries in **100+ languages**! This integration allows you to work seamlessly with documents in Arabic, Chinese, French, Spanish, German, Japanese, Korean, and many more languages.

## What Has Been Added

### 1. **New Embedding Model Options**

You can now select from the following Cohere models in the Streamlit UI:

- **cohere/embed-english-v3.0**: High-quality embeddings for English text
- **cohere/embed-multilingual-v3.0**: ⭐ Best choice for multilingual documents (1024 dimensions)
- **cohere/embed-english-light-v3.0**: Faster, smaller English embeddings
- **cohere/embed-multilingual-light-v3.0**: Fast multilingual embeddings (384 dimensions)

### 2. **Supported Languages**

The Cohere multilingual models support 100+ languages, including:
- **Major Languages**: English, Spanish, French, German, Italian, Portuguese, Dutch, Polish, Russian, Arabic, Chinese, Japanese, Korean, Hindi, Turkish, Indonesian, and many more
- **Total Coverage**: Over 100 languages with high-quality embeddings

### 3. **Modified Files**

#### [.env](.env)
```env
COHERE_API_KEY=RntiFbyLSiNoVjmv2yd1BXWpJf8o8UXWApyNzuTn
```

#### [requirements.txt](requirements.txt)
Added: `cohere`

#### [src/rag_agent.py](src/rag_agent.py)
- Added Cohere client initialization
- Updated `load_embedding_model()` to handle Cohere models
- Updated `embed_chunks()` to support both local and Cohere API embeddings
- Updated `search_query()` to use appropriate embedding method for queries
- Added UI options for Cohere models
- Added informational sidebar with language support details

## How to Use

### 1. **Start the Application**

```bash
streamlit run src/rag_agent.py
```

### 2. **Select Cohere Embedding Model**

In the sidebar under "Embedding Model", select one of the Cohere options:
- For **multilingual documents**: Choose `cohere/embed-multilingual-v3.0`
- For **English only**: Choose `cohere/embed-english-v3.0`
- For **faster processing**: Use the "light" versions

### 3. **Upload Your Documents**

Upload documents in any supported language (PDFs, Excel files, images). The system will:
- Extract text, tables, and images
- Generate embeddings using Cohere's API
- Store them in your selected vector database (ChromaDB or FAISS)

### 4. **Ask Questions**

Query your documents in any language! The Cohere embeddings will understand:
- Questions in the same language as the document
- Cross-lingual queries (ask in English about French documents, etc.)
- Mixed-language contexts

## Example Use Cases

### 1. **Multilingual Document Search**
```
Documents:
- Contract in French
- Report in Spanish
- Email in German

Query: "What are the payment terms?" (in English)
→ Cohere embeddings will find relevant sections across all languages!
```

### 2. **International Business Documents**
```
Documents:
- Chinese business proposal
- Japanese financial statement
- Korean contract

Query: "财务数据是什么?" (What is the financial data? in Chinese)
→ Works seamlessly across all documents!
```

### 3. **Academic Research Papers**
```
Documents:
- Papers in English, French, German
- Mixed-language citations

Query: "Quelles sont les conclusions principales?" (What are the main conclusions? in French)
→ Retrieves relevant information regardless of source language!
```

## Technical Details

### Embedding Dimensions
- **Full models** (v3.0): 1024 dimensions - Higher quality, better semantic understanding
- **Light models** (v3.0): 384 dimensions - Faster processing, lower memory usage

### API Batching
- Cohere API accepts up to 96 texts per request
- The system automatically batches your documents for efficient processing
- Progress is shown during embedding generation

### Input Types
- **Documents** (`search_document`): Used when embedding your document chunks
- **Queries** (`search_query`): Used when embedding user questions
- This distinction optimizes retrieval quality

## Performance Considerations

### Cohere API vs Local Models

**Cohere API Advantages:**
- ✅ Multilingual support (100+ languages)
- ✅ Higher quality embeddings
- ✅ No local GPU/CPU intensive processing
- ✅ Always up-to-date with latest models

**Cohere API Trade-offs:**
- ⚠️ Requires internet connection
- ⚠️ API rate limits may apply
- ⚠️ Slightly slower than local models for small batches

**Local Models (Sentence Transformers):**
- ✅ Fast for English-only documents
- ✅ No internet required
- ✅ No rate limits
- ⚠️ Limited language support
- ⚠️ Requires local compute resources

## Troubleshooting

### Error: "API Key Invalid"
**Solution**: Verify your API key in [.env](.env) file:
```env
COHERE_API_KEY=your_actual_api_key_here
```

### Error: "Rate Limit Exceeded"
**Solution**:
- Wait a few minutes before retrying
- Consider using the "light" models for faster processing
- Check your Cohere account's rate limits

### Slow Embedding Generation
**Solution**:
- Use light models for faster processing
- Process documents in smaller batches
- Consider using local models for English-only documents

## Testing

Run the test script to verify Cohere integration:

```bash
python test_cohere.py
```

Expected output:
```
Testing Cohere Multilingual Embeddings...
✅ Testing cohere/embed-multilingual-v3.0
   Shape: (7, 1024)
   Successfully embedded 7 texts in different languages!
✅ All Cohere embedding tests passed!
```

## API Key Security

⚠️ **Important**: Never commit your `.env` file to version control!

Add to your `.gitignore`:
```gitignore
.env
*.env
```

## Resources

- **Cohere Documentation**: https://docs.cohere.com/docs/embeddings
- **Supported Languages**: https://docs.cohere.com/docs/supported-languages
- **API Reference**: https://docs.cohere.com/reference/embed

## Next Steps

1. **Experiment with different models**: Try both full and light versions to find the best balance for your use case
2. **Test multilingual queries**: Upload documents in different languages and test cross-lingual retrieval
3. **Monitor API usage**: Keep track of your Cohere API usage in your dashboard
4. **Optimize chunk sizes**: Adjust chunk sizes for better semantic coherence in multilingual contexts

---

**Need Help?** Check the sidebar in the Streamlit app for:
- ℹ️ Embedding Models Info
- ℹ️ PDF Processing Info
- ℹ️ Vector Database Info
