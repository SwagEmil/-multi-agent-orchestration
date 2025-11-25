#!/usr/bin/env python3
"""
Clean YouTube transcript for RAG ingestion.

Removes:
- Filler words and sounds ("Heat", "N", etc.)
- Repetitive phrases
- Timestamps
- Excessive whitespace

Usage:
    python scripts/clean_transcript.py input.txt output.md
"""

import sys
import re
from pathlib import Path

def clean_transcript(input_file: str, output_file: str):
    """Clean YouTube transcript"""
    
    print(f"\n🧹 Cleaning transcript: {Path(input_file).name}\n")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract metadata (first 6 lines)
    lines = content.split('\n', 6)
    metadata = '\n'.join(lines[:6])
    transcript = lines[6] if len(lines) > 6 else ""
    
    original_length = len(transcript)
    
    # Clean transcript
    transcript = remove_youtube_artifacts(transcript)
    transcript = remove_repetitive_sounds(transcript)
    transcript = remove_excessive_filler(transcript)
    transcript = normalize_whitespace(transcript)
    transcript = fix_sentence_structure(transcript)
    
    cleaned_length = len(transcript)
    reduction = ((original_length - cleaned_length) / original_length) * 100 if original_length > 0 else 0
    
    # Format as Markdown
    output_content = f"{metadata}\n\n## Cleaned Transcript\n\n{transcript}"
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    print(f"✅ Original: {original_length} chars")
    print(f"✅ Cleaned: {cleaned_length} chars")
    print(f"✅ Reduced by: {reduction:.1f}%")
    print(f"✅ Saved to: {output_file}\n")

def remove_youtube_artifacts(text: str) -> str:
    """Remove YouTube-specific artifacts"""
    # Remove timestamps
    text = re.sub(r'\[\d{1,2}:\d{2}:\d{2}\]', '', text)
    text = re.sub(r'\d{1,2}:\d{2}', '', text)
    return text

def remove_repetitive_sounds(text: str) -> str:
    """Remove repetitive sounds at start/end of transcript"""
    # Remove repeated "Heat. Heat. N." patterns
    text = re.sub(r'^(Heat\.\s*|N\.\s*|Meow\.\s*|Yow\.\s*|Y\.\s*|Hey\.\s*|Oh,\s*down\w*\.\s*|Woo!\s*|down\s+){3,}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Heat\.\s*|N\.\s*){3,}$', '', text, flags=re.IGNORECASE)
    
    # Remove "down down up" patterns
    text = re.sub(r'(down,?\s*){3,}(up,?\s*){0,3}', '', text, flags=re.IGNORECASE)
    
    return text

def remove_excessive_filler(text: str) -> str:
    """Remove excessive filler words while keeping natural speech"""
    # Don't be too aggressive - just remove obvious repetition
    
    # Remove standalone "um", "uh" when repeated
    text = re.sub(r'\b(um|uh),?\s+\1\b', r'\1', text, flags=re.IGNORECASE)
    
    return text

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace"""
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove trailing spaces
    text = re.sub(r' +\n', '\n', text)
    # Normalize spaces
    text = re.sub(r'  +', ' ', text)
    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    return text.strip()

def fix_sentence_structure(text: str) -> str:
    """Fix sentence structure"""
    # Ensure space after periods
    text = re.sub(r'\.(\w)', r'. \1', text)
    # Ensure space after commas
    text = re.sub(r',(\w)', r', \1', text)
    
    return text

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python clean_transcript.py input.txt output.md")
        sys.exit(1)
    
    clean_transcript(sys.argv[1], sys.argv[2])
