"""Chunk quality validation for RAG pipeline."""

import logging
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChunkValidationConfig:
    """Configuration for chunk quality validation.

    Attributes:
        min_length: Minimum number of characters (default: 100)
        min_meaningful_ratio: Minimum ratio of alphanumeric chars (default: 0.3)
        max_special_char_ratio: Maximum ratio of special chars (default: 0.7)
        whitelist_patterns: Regex patterns to always allow (e.g., code blocks)
    """
    min_length: int = 100
    min_meaningful_ratio: float = 0.3
    max_special_char_ratio: float = 0.7
    whitelist_patterns: Optional[List[str]] = None

    def __post_init__(self):
        """Set default whitelist patterns if none provided."""
        if self.whitelist_patterns is None:
            self.whitelist_patterns = [
                r'```[\s\S]*?```',  # Code blocks
                r'^\s*def\s+\w+',   # Python functions
                r'^\s*class\s+\w+', # Python classes
            ]


class ChunkValidator:
    """Validates chunk quality before storage in vector database."""

    def __init__(self, config: Optional[ChunkValidationConfig] = None):
        """Initialize validator with configuration.

        Args:
            config: Validation configuration (uses defaults if not provided)
        """
        self.config = config or ChunkValidationConfig()
        logger.info(
            f"Initialized ChunkValidator: min_length={self.config.min_length}, "
            f"min_meaningful_ratio={self.config.min_meaningful_ratio}, "
            f"max_special_char_ratio={self.config.max_special_char_ratio}"
        )

    def is_valid_chunk(self, text: str) -> Tuple[bool, str]:
        """Check if chunk meets quality standards.

        Args:
            text: Chunk text to validate

        Returns:
            Tuple of (is_valid, reason) where reason explains why invalid
        """
        if not text:
            return False, "empty"

        text_stripped = text.strip()

        # Check if empty after stripping
        if not text_stripped:
            return False, "empty"

        # Check whitelist patterns first
        for pattern in self.config.whitelist_patterns:
            if re.search(pattern, text, re.MULTILINE):
                logger.debug(f"Chunk whitelisted by pattern: {pattern}")
                return True, "whitelisted"

        # Check table artifact pattern before length (table artifacts are often short)
        if re.match(r'^[\|\s\?\-]+$', text_stripped):
            return False, "table_artifact"

        # Check minimum length
        if len(text_stripped) < self.config.min_length:
            return False, f"too_short({len(text_stripped)})"

        # Calculate character ratios
        stats = self._calculate_stats(text_stripped)

        # Check meaningful content ratio
        if stats["meaningful_ratio"] < self.config.min_meaningful_ratio:
            return False, f"low_meaningful_ratio({stats['meaningful_ratio']:.2f})"

        # Check special character ratio
        if stats["special_char_ratio"] > self.config.max_special_char_ratio:
            return False, f"high_special_char_ratio({stats['special_char_ratio']:.2f})"

        return True, "valid"

    def _calculate_stats(self, text: str) -> Dict[str, Any]:
        """Calculate content statistics for validation.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with ratios and counts
        """
        total_chars = len(text)
        alphanumeric_count = sum(c.isalnum() for c in text)
        special_char_count = sum(not c.isalnum() and not c.isspace() for c in text)

        return {
            "total_chars": total_chars,
            "alphanumeric_count": alphanumeric_count,
            "special_char_count": special_char_count,
            "meaningful_ratio": alphanumeric_count / total_chars if total_chars > 0 else 0,
            "special_char_ratio": special_char_count / total_chars if total_chars > 0 else 0,
        }

    def get_validation_stats(self, text: str) -> Dict[str, Any]:
        """Get detailed validation statistics for debugging.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with validation decision and statistics
        """
        is_valid, reason = self.is_valid_chunk(text)
        stats = self._calculate_stats(text.strip())

        return {
            "is_valid": is_valid,
            "reason": reason,
            "length": len(text.strip()),
            "statistics": stats,
            "thresholds": {
                "min_length": self.config.min_length,
                "min_meaningful_ratio": self.config.min_meaningful_ratio,
                "max_special_char_ratio": self.config.max_special_char_ratio,
            }
        }
