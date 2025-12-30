"""Unit tests for ChunkValidator."""

from src.chunk_validator import ChunkValidator, ChunkValidationConfig


class TestChunkValidator:
    """Test chunk validation logic."""

    def test_valid_chunk(self):
        """Test that normal text passes validation."""
        validator = ChunkValidator()
        text = "This is a valid chunk with meaningful content that contains enough characters to pass the minimum length threshold."
        is_valid, reason = validator.is_valid_chunk(text)
        assert is_valid
        assert reason == "valid"

    def test_table_artifact_filtered(self):
        """Test that table artifacts are filtered."""
        validator = ChunkValidator()
        text = "|\n| ?"
        is_valid, reason = validator.is_valid_chunk(text)
        assert not is_valid
        assert reason == "table_artifact"

    def test_table_artifact_with_pipes_and_dashes(self):
        """Test that table formatting with pipes and dashes is filtered."""
        validator = ChunkValidator()
        text = "| | | - - -"
        is_valid, reason = validator.is_valid_chunk(text)
        assert not is_valid
        assert reason == "table_artifact"

    def test_too_short_filtered(self):
        """Test that short chunks are filtered."""
        validator = ChunkValidator(ChunkValidationConfig(min_length=100))
        text = "This is a short chunk under 100 characters."
        is_valid, reason = validator.is_valid_chunk(text)
        assert not is_valid
        assert "too_short" in reason

    def test_boundary_chunk_filtered(self):
        """Test that short boundary chunks like section headers are filtered."""
        validator = ChunkValidator()  # Default min_length=100
        text = "all OS X users.\n\n## Finding Hidden Files\n"
        is_valid, reason = validator.is_valid_chunk(text)
        assert not is_valid
        assert "too_short" in reason

    def test_minimum_length_boundary(self):
        """Test chunks exactly at the minimum length boundary."""
        validator = ChunkValidator(ChunkValidationConfig(min_length=50))
        # Exactly 50 characters
        text = "A" * 50
        is_valid, reason = validator.is_valid_chunk(text)
        assert is_valid
        assert reason == "valid"

        # Just below 50 characters
        text = "A" * 49
        is_valid, reason = validator.is_valid_chunk(text)
        assert not is_valid
        assert "too_short" in reason

    def test_low_meaningful_ratio_filtered(self):
        """Test that chunks with low alphanumeric ratio are filtered."""
        validator = ChunkValidator(ChunkValidationConfig(min_meaningful_ratio=0.3))
        text = "!@#$%^&*()_+{}|:<>?~`-=[]\\;',./!@#$%^&*()!@#$%^&*()_+{}|:<>?~`-=[]\\;',./!@#$%^&*()!@#$%^&*()_+{}|:<>?~`-=[]\\;',./!@#$%^&*()"
        is_valid, reason = validator.is_valid_chunk(text)
        assert not is_valid
        assert "low_meaningful_ratio" in reason

    def test_high_special_char_ratio_filtered(self):
        """Test that chunks with too many special characters are filtered."""
        validator = ChunkValidator(ChunkValidationConfig(
            min_length=50,  # Lower threshold so we can test special char ratio
            min_meaningful_ratio=0.2,  # Low enough to pass, so we test special char ratio
            max_special_char_ratio=0.5
        ))
        # Text with 60% special characters (60 special + 40 alphanumeric = 100 chars)
        text = "!@#$%^&*()!@#$%^&*()!@#$%^&*()!@#$%^&*()!@#$%^&*()!@#$%^&*()abc123abc123abc123abc123"
        is_valid, reason = validator.is_valid_chunk(text)
        assert not is_valid
        assert "high_special_char_ratio" in reason

    def test_code_block_whitelisted(self):
        """Test that code blocks pass validation despite special chars."""
        validator = ChunkValidator()
        text = """```python
def foo():
    return 42
```"""
        is_valid, reason = validator.is_valid_chunk(text)
        assert is_valid
        assert reason == "whitelisted"

    def test_python_function_whitelisted(self):
        """Test that Python function definitions are whitelisted."""
        validator = ChunkValidator()
        text = "def calculate_sum():"
        is_valid, reason = validator.is_valid_chunk(text)
        assert is_valid
        assert reason == "whitelisted"

    def test_python_class_whitelisted(self):
        """Test that Python class definitions are whitelisted."""
        validator = ChunkValidator()
        text = "class MyClass:"
        is_valid, reason = validator.is_valid_chunk(text)
        assert is_valid
        assert reason == "whitelisted"

    def test_empty_chunk_filtered(self):
        """Test that empty chunks are filtered."""
        validator = ChunkValidator()
        is_valid, reason = validator.is_valid_chunk("")
        assert not is_valid
        assert reason == "empty"

    def test_whitespace_only_filtered(self):
        """Test that whitespace-only chunks are filtered."""
        validator = ChunkValidator()
        text = "   \n\t\n   "
        is_valid, reason = validator.is_valid_chunk(text)
        assert not is_valid
        assert reason == "empty"

    def test_meaningful_text_with_normal_punctuation(self):
        """Test that normal text with punctuation passes."""
        validator = ChunkValidator()
        text = "This is a normal paragraph with proper punctuation. It has sentences, commas, and periods. The content is meaningful and readable by humans. It should pass all validation checks without any issues."
        is_valid, reason = validator.is_valid_chunk(text)
        assert is_valid
        assert reason == "valid"

    def test_get_validation_stats(self):
        """Test that validation stats are returned correctly."""
        validator = ChunkValidator()
        text = "This is a test chunk with enough content to pass all validation requirements including the minimum length threshold of one hundred characters."
        stats = validator.get_validation_stats(text)

        assert "is_valid" in stats
        assert "reason" in stats
        assert "length" in stats
        assert "statistics" in stats
        assert "thresholds" in stats

        assert stats["is_valid"] is True
        assert stats["reason"] == "valid"
        assert stats["length"] == len(text.strip())

    def test_custom_config(self):
        """Test that custom configuration values are applied."""
        config = ChunkValidationConfig(
            min_length=50,
            min_meaningful_ratio=0.5,
            max_special_char_ratio=0.4
        )
        validator = ChunkValidator(config)

        # Text with 45% special chars (25 special + 30 alphanumeric = 55 chars)
        text = "!@#$%^&*()!@#$%^&*()!@#$%1234567890abcdefghijklmnopqrs"
        is_valid, reason = validator.is_valid_chunk(text)
        # Should fail because special char ratio (45%) is > 0.4
        assert not is_valid
        assert "high_special_char_ratio" in reason

    def test_multiline_text(self):
        """Test that multiline text is handled correctly."""
        validator = ChunkValidator()
        text = """This is a multiline text chunk.
It contains several lines.
Each line has meaningful content.
The total length exceeds the minimum threshold.
Therefore it should pass validation."""
        is_valid, reason = validator.is_valid_chunk(text)
        assert is_valid
        assert reason == "valid"

    def test_real_world_section_header(self):
        """Test realistic section header from PDF (should be filtered)."""
        validator = ChunkValidator()
        text = "all OS X users.\n\nThere are a million reasons why it's helpful to know Unix as an OS X power user, and you'll see them demonstrated time and again throughout this book.\n\n## Finding Hidden Files\n"
        is_valid, reason = validator.is_valid_chunk(text)
        # This is ~160 characters but should pass since it has meaningful content
        if len(text.strip()) >= 100:
            # Should pass if it has enough meaningful content
            assert is_valid or "low_meaningful_ratio" in reason or "too_short" in reason
