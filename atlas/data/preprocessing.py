"""
Data preprocessing utilities.

This module provides functions for cleaning, normalizing, and preprocessing
text data before tokenization.
"""

import re
import unicodedata
from pathlib import Path
from typing import List, Union, Iterator, Optional
import logging

logger = logging.getLogger(__name__)


def clean_text(
    text: str,
    lowercase: bool = False,
    remove_extra_whitespace: bool = True,
    normalize_unicode: bool = True,
    remove_control_chars: bool = True,
) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text to clean
        lowercase: Convert to lowercase
        remove_extra_whitespace: Collapse multiple spaces into one
        normalize_unicode: Normalize Unicode characters (NFD -> NFC)
        remove_control_chars: Remove control characters
        
    Returns:
        Cleaned text
        
    Example:
        >>> text = "Hello   World!  \\n\\n  This is   a test."
        >>> clean_text(text)
        'Hello World! This is a test.'
    """
    if not text:
        return ""
    
    # Normalize Unicode
    if normalize_unicode:
        text = unicodedata.normalize('NFC', text)
    
    # Remove control characters (except newline and tab)
    if remove_control_chars:
        text = ''.join(
            char for char in text
            if unicodedata.category(char)[0] != 'C' or char in '\n\t'
        )
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove extra whitespace
    if remove_extra_whitespace:
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n\n+', '\n\n', text)
        # Strip leading/trailing whitespace
        text = text.strip()
    
    return text


def chunk_text(
    text: str,
    max_tokens: int,
    tokenizer,
    overlap: int = 0,
) -> List[str]:
    """
    Split text into chunks with a maximum token count.
    
    This is useful for processing long documents that exceed the model's
    context window. Chunks can optionally overlap to preserve context
    at boundaries.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
        tokenizer: Tokenizer instance for counting tokens
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
        
    Example:
        >>> from atlas.tokenizer import Tokenizer
        >>> tokenizer = Tokenizer()
        >>> text = "Long document text here..."
        >>> chunks = chunk_text(text, max_tokens=100, tokenizer=tokenizer, overlap=10)
    """
    if not text.strip():
        return []
    
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError(
            f"overlap must be non-negative and less than max_tokens, "
            f"got overlap={overlap}, max_tokens={max_tokens}"
        )
    
    # Tokenize the entire text
    tokens = tokenizer.encode(text)
    
    if len(tokens) <= max_tokens:
        # Text fits in one chunk
        return [text]
    
    # Create chunks with overlap
    chunks = []
    stride = max_tokens - overlap
    
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        
        # Decode this chunk
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move to next chunk
        start_idx += stride
        
        # If we've reached the end, break
        if end_idx == len(tokens):
            break
    
    logger.info(f"Split text into {len(chunks)} chunks (max_tokens={max_tokens}, overlap={overlap})")
    
    return chunks


def load_text_file(
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    clean: bool = True,
    **clean_kwargs,
) -> str:
    """
    Load text from a file with optional cleaning.
    
    Args:
        file_path: Path to text file
        encoding: File encoding
        clean: Whether to clean the text
        **clean_kwargs: Additional arguments for clean_text()
        
    Returns:
        Loaded text content
        
    Example:
        >>> text = load_text_file("data/document.txt", clean=True)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading text from {file_path}")
    
    with open(file_path, 'r', encoding=encoding) as f:
        text = f.read()
    
    if clean:
        text = clean_text(text, **clean_kwargs)
    
    return text


def load_jsonl(
    file_path: Union[str, Path],
    text_field: str = 'text',
    encoding: str = 'utf-8',
) -> List[str]:
    """
    Load text from a JSONL file.
    
    JSONL (JSON Lines) format has one JSON object per line. This function
    extracts a specified text field from each object.
    
    Args:
        file_path: Path to JSONL file
        text_field: Name of the field containing text
        encoding: File encoding
        
    Returns:
        List of text strings
        
    Example:
        >>> texts = load_jsonl("data/documents.jsonl", text_field="content")
    """
    import json
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading JSONL from {file_path}")
    
    texts = []
    with open(file_path, 'r', encoding=encoding) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
                if text_field not in obj:
                    logger.warning(
                        f"Line {line_num}: Missing field '{text_field}', skipping"
                    )
                    continue
                
                text = obj[text_field]
                if isinstance(text, str):
                    texts.append(text)
                else:
                    logger.warning(
                        f"Line {line_num}: Field '{text_field}' is not a string, skipping"
                    )
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}, skipping")
    
    logger.info(f"Loaded {len(texts)} texts from {file_path}")
    
    return texts


def iterate_documents(
    file_paths: Union[str, Path, List[Union[str, Path]]],
    file_format: str = 'txt',
    clean: bool = True,
    **kwargs,
) -> Iterator[str]:
    """
    Iterate over documents from one or more files.
    
    Supports different file formats and yields one document at a time,
    which is memory-efficient for large corpora.
    
    Args:
        file_paths: Path or list of paths to files
        file_format: Format of the files ('txt' or 'jsonl')
        clean: Whether to clean text
        **kwargs: Additional format-specific arguments
        
    Yields:
        Document text strings
        
    Example:
        >>> for doc in iterate_documents(["file1.txt", "file2.txt"]):
        ...     print(len(doc))
    """
    # Convert to list of paths
    if isinstance(file_paths, (str, Path)):
        file_paths = [Path(file_paths)]
    else:
        file_paths = [Path(p) for p in file_paths]
    
    for file_path in file_paths:
        if file_format == 'txt':
            # For txt files, yield the entire file as one document
            text = load_text_file(file_path, clean=clean)
            yield text
        
        elif file_format == 'jsonl':
            # For JSONL, yield each entry as a separate document
            texts = load_jsonl(file_path, **kwargs)
            for text in texts:
                if clean:
                    text = clean_text(text)
                yield text
        
        else:
            raise ValueError(f"Unsupported file format: {file_format}")


def count_tokens(text: str, tokenizer) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: Input text
        tokenizer: Tokenizer instance
        
    Returns:
        Number of tokens
    """
    return len(tokenizer.encode(text))


def filter_by_length(
    texts: List[str],
    tokenizer,
    min_tokens: int = 1,
    max_tokens: Optional[int] = None,
) -> List[str]:
    """
    Filter texts by token count.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance
        min_tokens: Minimum token count (inclusive)
        max_tokens: Maximum token count (inclusive), or None for no limit
        
    Returns:
        Filtered list of texts
        
    Example:
        >>> texts = ["Short", "This is a longer sentence", "Medium text"]
        >>> filtered = filter_by_length(texts, tokenizer, min_tokens=3, max_tokens=10)
    """
    filtered = []
    
    for text in texts:
        token_count = count_tokens(text, tokenizer)
        
        if token_count < min_tokens:
            continue
        
        if max_tokens is not None and token_count > max_tokens:
            continue
        
        filtered.append(text)
    
    logger.info(
        f"Filtered {len(texts)} texts -> {len(filtered)} texts "
        f"(min_tokens={min_tokens}, max_tokens={max_tokens})"
    )
    
    return filtered
