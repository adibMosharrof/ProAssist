"""
Tests for JSONParsingValidator

This validator is critical as it processes ALL GPT API responses.
These tests ensure robust JSON parsing with detailed error messages.
"""

import pytest
import json
from dst_data_builder.validators.json_parsing_validator import JSONParsingValidator

# Test data for parametrized tests (from conftest.py)
VALID_JSON_SAMPLES = [
    ('{"steps": []}', {"steps": []}),
    ('{"steps": [{"step_id": "S1"}]}', {"steps": [{"step_id": "S1"}]}),
    ('  {"steps": []}  ', {"steps": []}),
]

INVALID_JSON_SAMPLES = [
    '{"steps": [unclosed',
    'not json at all',
    '',
]

MARKDOWN_WRAPPED_JSON = [
    ('```json\n{"steps": []}\n```', {"steps": []}),
    ('```\n{"steps": []}\n```', {"steps": []}),
    ('Here is the JSON:\n{"steps": []}\nHope that helps!', {"steps": []}),
]


@pytest.mark.parametrize("json_str,expected", VALID_JSON_SAMPLES)
def test_valid_json_strings(json_str, expected):
    """Test parsing various valid JSON strings"""
    validator = JSONParsingValidator()
    ok, msg = validator.validate(json_str)
    
    assert ok, f"Should parse valid JSON, got error: {msg}"
    assert msg == ""
    assert validator.parsed_result == expected


@pytest.mark.parametrize("raw,expected", MARKDOWN_WRAPPED_JSON)
def test_json_with_markdown_wrapping(raw, expected):
    """Test parsing JSON wrapped in markdown or embedded in text"""
    validator = JSONParsingValidator()
    ok, msg = validator.validate(raw)
    
    assert ok, f"Should extract JSON, got error: {msg}"
    assert validator.parsed_result == expected


@pytest.mark.parametrize("invalid_json", INVALID_JSON_SAMPLES)
def test_invalid_json_syntax(invalid_json):
    """Test error handling for various invalid JSON formats"""
    validator = JSONParsingValidator()
    ok, msg = validator.validate(invalid_json)
    
    assert not ok, "Should fail on invalid JSON"
    assert msg != "", "Should provide error message"


def test_json_with_trailing_commas():
    """Test cleaning and parsing JSON with trailing commas"""
    validator = JSONParsingValidator()
    raw = '{"steps": [{"id": "S1", "name": "test",},],}'
    
    ok, msg = validator.validate(raw)
    
    assert ok, f"Should clean trailing commas, got error: {msg}"
    assert validator.parsed_result["steps"][0]["id"] == "S1"


def test_backward_compatibility_dict_input(valid_dst_structure):
    """Test backward compatibility - already parsed dict input"""
    validator = JSONParsingValidator()
    
    ok, msg = validator.validate(valid_dst_structure)
    
    assert ok
    assert validator.parsed_result == valid_dst_structure


def test_complex_nested_structure(valid_dst_structure):
    """Test parsing complex nested DST structure"""
    validator = JSONParsingValidator()
    raw = json.dumps(valid_dst_structure)
    
    ok, msg = validator.validate(raw)
    
    assert ok
    assert validator.parsed_result == valid_dst_structure


def test_invalid_input_type():
    """Test error on invalid input type"""
    validator = JSONParsingValidator()
    ok, msg = validator.validate(12345)  # int, not str or dict
    
    assert not ok
    assert "invalid input type" in msg.lower()


def test_json_with_unicode():
    """Test parsing JSON with unicode characters"""
    validator = JSONParsingValidator()
    raw = '{"steps": [{"name": "测试 Test ñ"}]}'
    
    ok, msg = validator.validate(raw)
    
    assert ok
    assert "测试" in validator.parsed_result["steps"][0]["name"]


def test_multiple_markdown_fences():
    """Test handling multiple code fences - uses first valid one"""
    validator = JSONParsingValidator()
    # Test with clean markdown-wrapped JSON (validator now extracts first JSON block)
    raw = '```json\n{"steps": [{"step_id": "S1"}]}\n```'
    
    ok, msg = validator.validate(raw)
    
    assert ok
    assert validator.parsed_result == {"steps": [{"step_id": "S1"}]}

    validator = JSONParsingValidator()
    raw = '```json\n{"steps": []}\n```\nSome text\n```\nmore text\n```'
    
    ok, msg = validator.validate(raw)
    
    assert ok
    assert validator.parsed_result == {"steps": []}
