#!/usr/bin/env python3
"""
Improved YouTube transcript cleaner - removes ALL noise artifacts.

Usage:
    python scripts/clean_transcript_v2.py input.txt output.md
"""

import sys
import re
from pathlib import Path

def clean_transcript(input_file: str, output_file: str):
    """Clean YouTube transcript thoroughly"""
    
    print(f"\n🧹 DEEP CLEANING: {Path(input_file).name}\n")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract metadata (first 6 lines)
    lines = content.split('\n', 6)
    metadata = '\n'.join(lines[:6])
    transcript = lines[6] if len(lines) > 6 else ""
    
    original_length = len(transcript)
    
    # Aggressive cleaning
    transcript = remove_all_youtube_noise(transcript)
    transcript = normalize_whitespace(transcript)
    
    cleaned_length = len(transcript)
    reduction = ((original_length - cleaned_length) / original_length) * 100 if original_length > 0 else 0
    
    # Format as Markdown
    output_content = f"{metadata}\n\n## Cleaned Transcript\n\n{transcript}"
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    print(f"✅ Original: {original_length:,} chars")
    print(f"✅ Cleaned: {cleaned_length:,} chars")
    print(f"✅ Reduced by: {reduction:.1f}%")
    print(f"✅ Saved to: {output_file}\n")

def remove_all_youtube_noise(text: str) -> str:
    """Aggressively remove ALL YouTube artifacts"""
    
    # Remove repetitive noise at start (much more aggressive)
    # Match any combination of: Heat, N, Meow, Yow, Y, Hey, Uh, down, up, Woo, etc.
    noise_pattern = r'^(Heat\.?\s*|N\.?\s*|Meow\.?\s*|Yow!?\s*|Y\.?\s*|Hey\.?\s*|Oh,?\s*|Uh,?\s*|down\w*\.?\s*|up\.?\s*|Woo!?\s*|Awesome\.?\s*){1,200}'
    text = re.sub(noise_pattern, '', text, flags=re.IGNORECASE)
    
    # Remove similar noise at end
    text = re.sub(r'(Heat\.?\s*|N\.?\s*|down\.?\s*|up\.?\s*|Yeah\.?\s*){1,50}$', '', text, flags=re.IGNORECASE)
    
    # Remove timestamps [00:15:30] or 00:15
    text = re.sub(r'\[\d{1,2}:\d{2}:\d{2}\]', '', text)
    text = re.sub(r'\d{1,2}:\d{2}', '', text)
    
    # Remove music notes and symbols
    text = re.sub(r'[♪♫🎵🎶]', '', text)
    
    # Remove excessive "um", "uh" when standalone or repeated
    text = re.sub(r'\b(um|uh|er|ah),?\s+', ' ', text, flags=re.IGNORECASE)
    
    # Remove "down down up" type patterns
    text = re.sub(r'\b(down|up)(\s+(down|up)){2,}', '', text, flags=re.IGNORECASE)
    
    return text

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace"""
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove trailing spaces
    text = re.sub(r' +\n', '\n', text)
    # Normalize spaces (but not TOO aggressive)
    text = re.sub(r'  +', ' ', text)
    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Ensure space after periods
    text = re.sub(r'\.(\w)', r'. \1', text)
    # Ensure space after commas
    text = re.sub(r',(\w)', r', \1', text)
    
    return text.strip()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python clean_transcript_v2.py input.txt output.md")
        sys.exit(1)
    
    clean_transcript(sys.argv[1], sys.argv[2])
