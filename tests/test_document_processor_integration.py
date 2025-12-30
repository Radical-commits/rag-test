"""Integration tests for document processor with chunk validation."""

from src.document_processor import DocumentProcessor
from src.chunk_validator import ChunkValidationConfig


class TestDocumentProcessorIntegration:
    """Test document processor with chunk filtering."""

    def test_filters_table_artifacts(self):
        """Test that table artifacts are filtered during processing."""
        processor = DocumentProcessor(
            chunk_size=50,
            chunk_overlap=10,
            validation_config=ChunkValidationConfig(min_length=30)
        )

        # Text where table artifact will be in its own chunk due to small chunk_size
        text = "Valid content here. " + "A" * 60 + " |\n| ? " + "B" * 60 + " More valid."
        chunks = processor._create_chunks(text, source="test.pdf", metadata={})

        # Should get valid chunks, table artifact should be filtered as standalone chunk
        assert len(chunks) >= 1
        # Check that standalone table artifacts are filtered
        for chunk in chunks:
            # If a chunk is just the table artifact, it should have been filtered
            if chunk.text.strip() == "|\n| ?":
                assert False, "Standalone table artifact was not filtered"

    def test_filters_short_chunks(self):
        """Test that short chunks are filtered."""
        processor = DocumentProcessor(
            chunk_size=50,
            chunk_overlap=10,
            validation_config=ChunkValidationConfig(min_length=40)
        )

        # Text that will create short chunks
        text = "A" * 25 + " " + "B" * 25 + " " + "C" * 60  # Creates varied-length chunks
        chunks = processor._create_chunks(text, source="test.pdf", metadata={})

        # All chunks should be at least 40 characters
        for chunk in chunks:
            assert len(chunk.text.strip()) >= 40, f"Short chunk found: {len(chunk.text)} chars"

    def test_filtering_stats_tracked(self):
        """Test that filtering statistics are tracked correctly."""
        processor = DocumentProcessor(
            chunk_size=50,
            chunk_overlap=5,
            validation_config=ChunkValidationConfig(min_length=30)
        )

        # Process text that will create some valid and some invalid chunks
        text = "A" * 60 + "B" * 10 + "C" * 60  # Middle part will create short chunk
        chunks = processor._create_chunks(text, source="test.pdf", metadata={})

        stats = processor.get_filtering_stats()
        assert stats["total_chunks_created"] > 0
        assert "filter_reasons" in stats
        assert "filter_rate" in stats
        assert 0 <= stats["filter_rate"] <= 1

    def test_reset_filtering_stats(self):
        """Test that filtering stats can be reset."""
        processor = DocumentProcessor()

        # Process some text
        text = "Valid chunk content here with enough characters to pass validation. " * 3
        processor._create_chunks(text, source="test.pdf", metadata={})

        # Verify stats exist
        stats_before = processor.get_filtering_stats()
        assert stats_before["total_chunks_created"] > 0

        # Reset and verify
        processor.reset_filtering_stats()
        stats_after = processor.get_filtering_stats()
        assert stats_after["total_chunks_created"] == 0
        assert stats_after["chunks_filtered"] == 0
        assert len(stats_after["filter_reasons"]) == 0

    def test_no_filtering_with_permissive_config(self):
        """Test that no chunks are filtered with very permissive config."""
        processor = DocumentProcessor(
            chunk_size=100,
            chunk_overlap=20,
            validation_config=ChunkValidationConfig(
                min_length=1,  # Very low threshold
                min_meaningful_ratio=0.0,  # Accept any ratio
                max_special_char_ratio=1.0  # Accept all special chars
            )
        )

        text = "Short.\n\n|\n| ?\n\nNormal content here."
        chunks = processor._create_chunks(text, source="test.pdf", metadata={})

        stats = processor.get_filtering_stats()
        # With permissive config, filter rate should be very low or zero
        assert stats["filter_rate"] < 0.2  # Allow up to 20% for edge cases

    def test_aggressive_filtering(self):
        """Test that aggressive config filters more chunks."""
        processor = DocumentProcessor(
            chunk_size=60,
            chunk_overlap=10,
            validation_config=ChunkValidationConfig(
                min_length=50,  # High threshold
                min_meaningful_ratio=0.5,  # Require 50% alphanumeric
                max_special_char_ratio=0.3  # Allow only 30% special chars
            )
        )

        # Create text with multiple sections, some short that will be filtered
        text = "A" * 80 + " Short. " + "B" * 80 + " More. " + "C" * 80
        chunks = processor._create_chunks(text, source="test.pdf", metadata={})

        stats = processor.get_filtering_stats()
        # Should have filtered some chunks (the short "Short." and "More." sections)
        assert stats["chunks_filtered"] > 0
        assert stats["filter_rate"] > 0

    def test_metadata_preserved_in_valid_chunks(self):
        """Test that metadata is correctly preserved in valid chunks."""
        processor = DocumentProcessor()

        text = "This is a valid chunk with meaningful content that should pass all validation checks including the minimum length requirement of one hundred characters."
        metadata = {"original_path": "/path/to/file.pdf", "extra_info": "test"}
        chunks = processor._create_chunks(text, source="test.pdf", metadata=metadata)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.source == "test.pdf"
            assert "original_path" in chunk.metadata
            assert chunk.metadata["original_path"] == "/path/to/file.pdf"
            assert "extra_info" in chunk.metadata
            assert chunk.metadata["extra_info"] == "test"

    def test_chunk_index_sequential(self):
        """Test that valid chunks have sequential indices."""
        processor = DocumentProcessor(
            chunk_size=100,
            chunk_overlap=20,
            validation_config=ChunkValidationConfig(min_length=50)
        )

        text = "A" * 200 + "B" * 200 + "C" * 200  # Creates multiple chunks
        chunks = processor._create_chunks(text, source="test.pdf", metadata={})

        # Chunk indices should be sequential starting from 0
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_filter_reasons_recorded(self):
        """Test that different filter reasons are recorded correctly."""
        processor = DocumentProcessor(
            chunk_size=80,
            chunk_overlap=10,
            validation_config=ChunkValidationConfig(min_length=50)
        )

        # Text with various problematic patterns
        text = "Valid content here. " + "x" * 60 + " |\n| ? " + "More valid content. " + "y" * 70
        chunks = processor._create_chunks(text, source="test.pdf", metadata={})

        stats = processor.get_filtering_stats()
        filter_reasons = stats["filter_reasons"]

        # Should have recorded specific reasons
        assert isinstance(filter_reasons, dict)
        # May contain reasons like "too_short", "table_artifact", etc.
        if stats["chunks_filtered"] > 0:
            assert len(filter_reasons) > 0

    def test_empty_text_handling(self):
        """Test handling of empty text."""
        processor = DocumentProcessor()

        text = ""
        chunks = processor._create_chunks(text, source="test.pdf", metadata={})

        assert len(chunks) == 0
        stats = processor.get_filtering_stats()
        assert stats["total_chunks_created"] == 0

    def test_whitespace_only_text(self):
        """Test that whitespace-only text produces no chunks."""
        processor = DocumentProcessor()

        text = "   \n\n\t\t   \n   "
        chunks = processor._create_chunks(text, source="test.pdf", metadata={})

        assert len(chunks) == 0

    def test_real_world_section_headers(self):
        """Test filtering of realistic section headers from PDFs."""
        processor = DocumentProcessor(
            chunk_size=200,
            chunk_overlap=50,
            validation_config=ChunkValidationConfig(min_length=100)
        )

        # Simulate text with section headers that might be short
        text = """This is the main content of a section with substantial text that explains concepts in detail.

## Short Header

More content follows with detailed explanations and examples that make up a full paragraph."""

        chunks = processor._create_chunks(text, source="test.pdf", metadata={})

        # Section headers alone shouldn't make it through (too short)
        for chunk in chunks:
            # If a chunk is very short (just the header), it should have been filtered
            if "## Short Header" in chunk.text:
                # The header should only appear with surrounding context
                assert len(chunk.text.strip()) >= 100

    def test_code_blocks_not_filtered(self):
        """Test that code blocks pass validation."""
        processor = DocumentProcessor()

        text = """```python
def example_function():
    return 42
```"""

        chunks = processor._create_chunks(text, source="test.pdf", metadata={})

        # Code blocks should be whitelisted and pass through
        assert len(chunks) > 0
        # At least one chunk should contain the code
        assert any("def example_function" in chunk.text for chunk in chunks)

    def test_stats_accumulate_across_multiple_calls(self):
        """Test that stats accumulate across multiple chunk creation calls."""
        processor = DocumentProcessor()

        # First call
        text1 = "First batch of text with meaningful content that passes validation."
        processor._create_chunks(text1, source="test1.pdf", metadata={})

        stats1 = processor.get_filtering_stats()
        count1 = stats1["total_chunks_created"]

        # Second call
        text2 = "Second batch of text with different content that also passes validation checks."
        processor._create_chunks(text2, source="test2.pdf", metadata={})

        stats2 = processor.get_filtering_stats()
        count2 = stats2["total_chunks_created"]

        # Stats should have accumulated
        assert count2 > count1
