"""
Shared test fixtures and utilities for DST data builder tests.

This conftest.py provides:
- Reusable test data fixtures
- Mock factories for OpenAI API
- Helper functions for common assertions
- Shared test utilities
"""

import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from omegaconf import OmegaConf


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def basic_config():
    """Basic configuration for SimpleDSTGenerator"""
    return OmegaConf.create({
        "generator": {"type": "single"},
        "model": {"name": "gpt-4o", "temperature": 0.1, "max_tokens": 4000},
        "max_retries": 1,
    })


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_input_data():
    """Sample input data structure"""
    return {
        "video_uid": "test_video_001",
        "inferred_knowledge": "Task involves assembling components with proper alignment",
        "parsed_video_anns": {
            "all_step_descriptions": (
                "[0.0s-10.0s] Pick up first component\n"
                "[10.0s-20.0s] Align component with base\n"
                "[20.0s-30.0s] Secure component"
            )
        }
    }


@pytest.fixture
def valid_dst_structure():
    """Valid DST structure for testing"""
    return {
        "steps": [
            {
                "step_id": "S1",
                "name": "Assemble components",
                "timestamps": {"start_ts": 0.0, "end_ts": 30.0},
                "substeps": [
                    {
                        "sub_id": "S1.1",
                        "name": "Pick up component",
                        "timestamps": {"start_ts": 0.0, "end_ts": 10.0},
                        "actions": [
                            {
                                "act_id": "S1.1.a",
                                "name": "Grasp component",
                                "timestamps": {"start_ts": 0.0, "end_ts": 5.0}
                            }
                        ]
                    }
                ]
            }
        ]
    }


@pytest.fixture
def invalid_dst_missing_timestamps():
    """Invalid DST structure missing timestamps"""
    return {
        "steps": [
            {
                "step_id": "S1",
                "name": "Step without timestamps"
            }
        ]
    }


# ============================================================================
# Temporary Data Fixtures
# ============================================================================

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory with test JSON files"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def populated_test_dir(test_data_dir, sample_input_data):
    """Test directory with 3 sample JSON files"""
    for i in range(3):
        test_file = test_data_dir / f"sample_{i}.json"
        data = sample_input_data.copy()
        data["video_uid"] = f"test_video_{i:03d}"
        test_file.write_text(json.dumps(data))
    return test_data_dir


@pytest.fixture
def single_test_file(test_data_dir, sample_input_data):
    """Single test file with sample data"""
    test_file = test_data_dir / "test.json"
    test_file.write_text(json.dumps(sample_input_data))
    return test_file


# ============================================================================
# Mock Factories
# ============================================================================

class MockOpenAIResponse:
    """Factory for creating mock OpenAI API responses"""
    
    @staticmethod
    def create_success(content):
        """Create a successful API response"""
        class Message:
            def __init__(self, content):
                self.content = content
        
        class Choice:
            def __init__(self, message):
                self.message = message
        
        class Response:
            def __init__(self, choices):
                self.choices = choices
        
        return Response([Choice(Message(content))])
    
    @staticmethod
    def create_json_response(data_dict):
        """Create response with JSON content"""
        return MockOpenAIResponse.create_success(json.dumps(data_dict))


@pytest.fixture
def mock_openai_factory():
    """Factory for creating OpenAI mocks"""
    return MockOpenAIResponse


@pytest.fixture
def mock_api_client(monkeypatch):
    """Mock OpenAIAPIClient with controllable responses"""
    responses = []
    
    async def fake_generate(prompt, model, temperature, max_tokens):
        if responses:
            response = responses.pop(0)
            return response
        # Default success
        return True, '{"steps": []}'
    
    class MockClient:
        def __init__(self):
            self.responses = responses
            self.generate_completion = fake_generate
        
        def add_response(self, success, content):
            """Add a response to the queue"""
            self.responses.append((success, content))
    
    return MockClient()


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_files(directory: Path, count: int = 3, base_data: dict = None):
    """
    Helper to create multiple test JSON files.
    
    Args:
        directory: Path to directory
        count: Number of files to create
        base_data: Base data dict to use (will be copied and modified)
    
    Returns:
        List of created file paths
    """
    if base_data is None:
        base_data = {
            "video_uid": "test",
            "inferred_knowledge": "test knowledge",
            "parsed_video_anns": {"all_step_descriptions": "test steps"}
        }
    
    files = []
    for i in range(count):
        file_path = directory / f"test_{i}.json"
        data = base_data.copy()
        data["video_uid"] = f"test_{i:03d}"
        file_path.write_text(json.dumps(data))
        files.append(file_path)
    
    return files


def assert_valid_dst(dst_structure):
    """Assert that a DST structure has required fields"""
    assert "steps" in dst_structure
    assert isinstance(dst_structure["steps"], list)
    if dst_structure["steps"]:
        step = dst_structure["steps"][0]
        assert "step_id" in step
        assert "name" in step


def assert_valid_dst_output(dst_output):
    """Assert that a DSTOutput has required fields"""
    assert dst_output is not None
    result_dict = dst_output.to_dict()
    assert "dst" in result_dict
    assert "metadata" in result_dict
    assert "counts" in result_dict["metadata"]
    assert_valid_dst(result_dict["dst"])


# ============================================================================
# Parametrization Helpers
# ============================================================================

# Common test data for parametrized tests
VALID_JSON_SAMPLES = [
    ('{"steps": []}', {"steps": []}),
    ('{"steps": [{"step_id": "S1"}]}', {"steps": [{"step_id": "S1"}]}),
    ('  {"steps": []}  ', {"steps": []}),  # With whitespace
]

INVALID_JSON_SAMPLES = [
    '{"steps": [unclosed',
    'not json at all',
    '{"trailing": "comma",}',
    '',
]

MARKDOWN_WRAPPED_JSON = [
    ('```json\n{"steps": []}\n```', {"steps": []}),
    ('```\n{"steps": []}\n```', {"steps": []}),
    ('Some text\n{"steps": []}\nMore text', {"steps": []}),
]
