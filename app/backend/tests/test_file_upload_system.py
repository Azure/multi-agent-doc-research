"""
Simple File Upload Test

Tests the unified file upload plugin by simulating frontend behavior.
Sends local PDF files to the backend and verifies the complete workflow.

Usage:
    # Run all tests
    pytest test_file_upload_system.py -v
    
    # Run manual test 
    python test_file_upload_system.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import pytest

# Add the parent directory to sys.path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from services_sk.unified_file_upload_plugin import UnifiedFileUploadPlugin
from services_sk.ai_search_plugin import AISearchPlugin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileUploadTest:
    """Simple test client for file upload functionality."""
    
    def __init__(self):
        self.upload_plugin = UnifiedFileUploadPlugin()
        self.search_plugin = AISearchPlugin()
        self.test_files_dir = Path(__file__).parent / "test_files"
        
    def get_test_files(self):
        """Get available PDF files for testing."""
        return [str(f.absolute()) for f in self.test_files_dir.glob("*.pdf") if f.exists()]
    
    async def test_upload_request(self, message: str):
        """Test upload request detection."""
        result = await self.upload_plugin.handle_upload_request(message)
        return json.loads(result)
    
    async def test_file_upload(self, file_paths, **kwargs):
        """Test file upload process."""
        result = await self.upload_plugin.upload_documents(
            file_paths=json.dumps(file_paths),
            **kwargs
        )
        return json.loads(result)
    
    async def test_file_status(self, file_names):
        """Test file status checking."""
        file_names_str = ", ".join([Path(f).name for f in file_names])
        result = await self.upload_plugin.check_docs_status(file_names_str)
        return json.loads(result)
    
    async def test_search(self, query, company="Test Company"):
        """Test document search (runs sync search in thread to avoid blocking)."""
        return self.search_plugin.search_documents(
            query=query,
            search_type="semantic",
            company=company,
            top_k=3,
            include_content=True
        )


@pytest.fixture
def test_client():
    return FileUploadTest()


@pytest.fixture
def sample_files(test_client):
    files = test_client.get_test_files()
    if not files:
        pytest.skip("No PDF files found for testing")
    return files


@pytest.mark.asyncio
async def test_upload_workflow(test_client, sample_files):
    """Test complete upload workflow."""
    test_file = sample_files[0]
    
    # Test upload request detection
    response = await test_client.test_upload_request("I want to upload documents")
    assert response["status"] == "upload_requested"
    
    # Test file upload
    upload_result = await test_client.test_file_upload(
        [test_file],
        document_type="TEST_DOC",
        company="Test Corp",
        force_upload="true",
        is_debug=True
    )
    assert upload_result["status"] == "completed"
    
    # Test file status
    status_result = await test_client.test_file_status([test_file])
    assert status_result["status"] == "success"
    
    # Test search
    search_result = await test_client.test_search("revenue", "Test Corp")
    assert "status" in search_result


# Manual test runner
async def run_manual_test():
    """Simple manual test for direct execution."""
    print("🧪 File Upload Test")
    print("=" * 30)
    
    client = FileUploadTest()
    files = client.get_test_files()
    
    if not files:
        print("❌ No PDF files found in ai_search directory")
        return
    
    print(f"📁 Found {len(files)} test files")
    test_file = files[0]
    
    # Test upload request
    print("\n1. Testing upload request detection...")
    response = await client.test_upload_request("I want to upload files")
    print(f"   Result: {response['status']}")
    
    # Test file upload  
    print("\n2. Testing file upload...")
    upload_result = await client.test_file_upload(
        [test_file],
        document_type="MANUAL_TEST",
        company="Manual Test Corp",
        force_upload="true"
    )
    print(f"   Result: {upload_result['status']}")
    print(f"   Successful: {upload_result.get('successful_uploads', 0)}")
    
    # Test file status
    print("\n3. Testing file status...")
    status_result = await client.test_file_status([test_file])
    print(f"   Result: {status_result['status']}")
    print(f"   Files: {status_result.get('files', [])}")
    
    # Test search
    print("\n4. Testing document search...")
    search_result = await client.test_search("financial", "Manual Test Corp")
    if search_result.get('documents'):
        print(f"   Found: {len(search_result['documents'])} documents")
    else:
        print("   No search results")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    asyncio.run(run_manual_test())
