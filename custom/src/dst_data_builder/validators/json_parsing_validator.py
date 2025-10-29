"""
JSON Parsing Validator - Parses and validates JSON responses from GPT API
"""

import json
import re
import logging
from typing import Dict, Any, Tuple, Optional, Union

from dst_data_builder.validators.base_validator import BaseValidator


class JSONParsingValidator(BaseValidator):
    """
    Validator that parses raw JSON responses from GPT API.
    
    This validator transforms string input into Dict output and provides
    detailed error messages for retry prompts.
    
    Usage:
        validator = JSONParsingValidator()
        success, error_msg = validator.validate(raw_string)
        if success:
            parsed_dict = validator.parsed_result
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._parsed_result: Optional[Dict[str, Any]] = None
    
    def validate(self, data: Union[str, Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Parse and validate JSON input.
        
        Args:
            data: Raw JSON string from API or already-parsed dict
            
        Returns:
            Tuple of (success, error_message)
            - On success: parsed result available via .parsed_result property
            - On failure: detailed error message for retry prompt
        """
        # If already a dict, it's been parsed (backward compatibility)
        if isinstance(data, dict):
            self._parsed_result = data
            return True, ""
        
        # If not a string, invalid input
        if not isinstance(data, str):
            return False, f"Invalid input type: expected str or dict, got {type(data).__name__}"
        
        # Clean and parse JSON
        return self._parse_json(data)
    
    def _parse_json(self, raw_response: str) -> Tuple[bool, str]:
        """Parse raw JSON response with detailed error reporting."""
        
        # Step 1: Extract JSON if embedded in text
        json_content = raw_response.strip()
        
        if not json_content.startswith("{"):
            # Try to find JSON in the response
            json_match = re.search(r"\{.*\}", json_content, re.DOTALL)
            if json_match:
                json_content = json_match.group()
                self.logger.info(f"Extracted JSON from response: {len(json_content)} chars")
            else:
                return False, (
                    "No valid JSON object found in response.\n"
                    "Expected: Response should contain a JSON object starting with '{' and ending with '}'.\n"
                    f"Received: {raw_response[:200]}..."
                )
        
        # Step 2: Clean common JSON formatting issues
        json_content = self._clean_json_response(json_content)
        
        # Step 3: Attempt to parse
        try:
            parsed = json.loads(json_content)
            self._parsed_result = parsed
            self.logger.debug("✅ Successfully parsed JSON")
            return True, ""
            
        except json.JSONDecodeError as e:
            error_msg = self._format_parse_error(e, json_content)
            self._parsed_result = None
            self.logger.warning(f"JSON parsing failed: {error_msg}")
            return False, error_msg
    
    def _clean_json_response(self, content: str) -> str:
        """Clean common JSON formatting issues in GPT responses."""
        # Remove markdown code blocks
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*$", "", content)
        
        # Fix trailing commas before closing braces/brackets
        content = re.sub(r",(\s*[}\]])", r"\1", content)
        
        return content.strip()
    
    def _format_parse_error(self, error: json.JSONDecodeError, content: str) -> str:
        """
        Format detailed JSON parsing error for retry prompt.
        
        Provides:
        - Exact line and column of error
        - Error type and message
        - Context around the error (50 chars before/after)
        - Visual pointer to error location
        """
        # Calculate context window
        context_start = max(0, error.pos - 50) if error.pos else 0
        context_end = min(len(content), error.pos + 50) if error.pos else 100
        
        context = content[context_start:context_end]
        
        # Calculate pointer position in context
        pointer_pos = (error.pos - context_start) if error.pos else 0
        pointer = " " * pointer_pos + "^"
        
        # Get the problematic line
        lines = content.split('\n')
        error_line = lines[error.lineno - 1] if error.lineno <= len(lines) else ""
        
        error_msg = (
            f"JSON Parse Error at line {error.lineno}, column {error.colno}:\n"
            f"\n"
            f"Error Type: {error.msg}\n"
            f"\n"
            f"Problematic Line:\n"
            f"  {error_line}\n"
            f"  {' ' * (error.colno - 1)}^\n"
            f"\n"
            f"Context (50 chars around error):\n"
            f"  ...{context}...\n"
            f"  {pointer}\n"
            f"\n"
            f"Common Issues:\n"
            f"  - Missing or extra commas\n"
            f"  - Unquoted keys or values\n"
            f"  - Unclosed brackets/braces\n"
            f"  - Invalid escape sequences\n"
            f"\n"
            f"Please fix the JSON syntax at the indicated position."
        )
        
        return error_msg
    
    @property
    def parsed_result(self) -> Optional[Dict[str, Any]]:
        """Get the parsed JSON result after successful validation."""
        return self._parsed_result
