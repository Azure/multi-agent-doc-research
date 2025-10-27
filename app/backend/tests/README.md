# File Upload System Test

Simple test for the unified file upload plugin that simulates frontend behavior.

## Usage

### Run with pytest
```bash
cd app/backend
pytest tests/test_file_upload_system.py -v
```

### Run manual test
```bash
cd app/backend  
python tests/test_file_upload_system.py
```

## What it tests

1. **Upload request detection** - Detects when users want to upload files
2. **File upload process** - Tests chunking and vector storage 
3. **File status checking** - Verifies files exist in database
4. **Document search** - Tests searching uploaded content
5. **Error handling** - Tests with invalid files

## Test files

Uses PDF files in `app/backend/ai_search/` directory automatically.

## Requirements

- Azure AI Search and OpenAI services configured
- Environment variables set for Azure credentials
- PDF files in `ai_search/` directory for testing
