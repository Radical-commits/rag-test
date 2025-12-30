"""Document processing module using Docling for PDF parsing and chunking."""

import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from .chunk_validator import ChunkValidator, ChunkValidationConfig
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with metadata."""

    text: str
    source: str  # Filename
    page: int
    chunk_index: int
    metadata: Dict[str, Any]


class DocumentProcessor:
    """Handles PDF document processing and text chunking."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        validation_config: Optional[ChunkValidationConfig] = None
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
            validation_config: Chunk quality validation configuration
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.validator = ChunkValidator(validation_config)

        # Configure Docling with XET mode disabled (set in app.py)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR
        pipeline_options.do_table_structure = True  # Enable table structure detection
        pipeline_options.generate_page_images = False  # Keep image generation disabled
        pipeline_options.generate_picture_images = False  # Keep picture generation disabled

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        logger.info(f"Initialized DocumentProcessor (chunk_size={chunk_size}, overlap={chunk_overlap}, OCR=enabled, tables=enabled, validation=enabled)")

    def process_pdf(self, pdf_path: Path) -> tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        Process a single PDF file and extract text chunks.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (chunks, batch_stats) where batch_stats contains filtering metrics

        Raises:
            ValueError: If the file is not a PDF or cannot be read
        """
        if not pdf_path.exists():
            raise ValueError(f"File not found: {pdf_path}")

        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF: {pdf_path}")

        logger.info(f"Processing PDF: {pdf_path.name}")

        try:
            # Use Docling to convert the PDF
            result = self.converter.convert(str(pdf_path))

            # Extract text from the document
            full_text = result.document.export_to_markdown()

            # Log text extraction details
            text_length = len(full_text)
            logger.info(f"Extracted {text_length} characters of text from {pdf_path.name}")

            if text_length == 0:
                logger.warning(f"No text extracted from {pdf_path.name}! Document may be empty or image-only.")
                # Return empty chunks and empty stats
                return [], {
                    "chunks_created": 0,
                    "chunks_filtered": 0,
                    "filter_reasons": {}
                }

            # Create chunks from the text (now returns tuple with stats)
            chunks, batch_stats = self._create_chunks(
                text=full_text,
                source=pdf_path.name,
                metadata={"original_path": str(pdf_path)}
            )

            logger.info(
                f"Created {len(chunks)} chunks from {pdf_path.name} "
                f"({batch_stats['chunks_created']} total, {batch_stats['chunks_filtered']} filtered)"
            )
            return chunks, batch_stats

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path.name}: {e}")
            raise

    def process_multiple_pdfs(self, pdf_paths: List[Path]) -> List[DocumentChunk]:
        """
        Process multiple PDF files.

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            Combined list of DocumentChunk objects from all PDFs
        """
        all_chunks = []

        for pdf_path in pdf_paths:
            try:
                chunks = self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue

        logger.info(f"Processed {len(pdf_paths)} PDFs, total {len(all_chunks)} chunks")
        return all_chunks

    def _create_chunks(
        self,
        text: str,
        source: str,
        metadata: Dict[str, Any]
    ) -> tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        Split text into overlapping chunks and validate quality.

        Args:
            text: Full text to chunk
            source: Source filename
            metadata: Additional metadata to attach

        Returns:
            Tuple of (valid_chunks, batch_stats) where batch_stats = {
                "chunks_created": int,
                "chunks_filtered": int,
                "filter_reasons": dict
            }
        """
        chunks = []
        text_length = len(text)
        start = 0
        chunk_index = 0

        # Initialize batch stats
        batch_stats = {
            "chunks_created": 0,
            "chunks_filtered": 0,
            "filter_reasons": {}
        }

        logger.debug(f"Starting chunking of {text_length} characters")

        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)

            # If this is not the last chunk, try to break at a sentence or paragraph boundary
            if end < text_length:
                # Look for paragraph break first
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break > start:
                    end = paragraph_break + 2
                else:
                    # Look for sentence break
                    sentence_break = max(
                        text.rfind('. ', start, end),
                        text.rfind('! ', start, end),
                        text.rfind('? ', start, end)
                    )
                    if sentence_break > start:
                        end = sentence_break + 2

            # Extract chunk text
            chunk_text = text[start:end].strip()

            # Track chunk creation
            batch_stats["chunks_created"] += 1

            # Validate chunk quality
            is_valid, reason = self.validator.is_valid_chunk(chunk_text)

            if is_valid:
                chunk = DocumentChunk(
                    text=chunk_text,
                    source=source,
                    page=chunk_index // 3 + 1,  # Rough page estimation
                    chunk_index=chunk_index,
                    metadata={**metadata, "start_pos": start, "end_pos": end}
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Track filtering statistics
                batch_stats["chunks_filtered"] += 1
                if reason not in batch_stats["filter_reasons"]:
                    batch_stats["filter_reasons"][reason] = 0
                batch_stats["filter_reasons"][reason] += 1

                logger.debug(
                    f"Filtered chunk: reason={reason}, "
                    f"length={len(chunk_text)}, "
                    f"sample='{chunk_text[:100]}'"
                )

            # Move to next chunk with overlap
            # Ensure we always move forward to prevent infinite loop
            if end >= text_length:
                break

            next_start = end - self.chunk_overlap
            if next_start <= start:
                # Safety: if overlap would cause us to go backwards or stay in place,
                # move forward by at least 1 character
                next_start = start + 1

            start = next_start

        logger.debug(f"Chunking complete: {len(chunks)} chunks created")
        return chunks, batch_stats
