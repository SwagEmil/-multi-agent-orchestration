#!/usr/bin/env python3
"""
ULTRA AGGRESSIVE YouTube transcript cleaner - removes ALL garbage.

Usage:
    python scripts/clean_transcript_ultra.py input.txt output.md
"""

import sys
import re
from pathlib import Path

def clean_transcript(input_file: str, output_file: str):
    """Ultra aggressively clean YouTube transcript"""
    
    print(f"\n🔥 ULTRA CLEANING: {Path(input_file).name}\n")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract metadata
    lines = content.split('\n', 6)
    metadata = '\n'.join(lines[:6])
    transcript = lines[6] if len(lines) > 6 else ""
    
    original_length = len(transcript)
    
    # Find where actual content starts (after all the noise)
    # Look for characteristic opening phrases
    start_markers = [
        r'Welcome back to',
        r'Hello everyone',
        r'Welcome to',
        r'Hi everyone',
        r'Good morning',
        r'Thanks for joining'
    ]
    
    start_pos = 0
    for marker in start_markers:
        match = re.search(marker, transcript, re.IGNORECASE)
        if match:
            # Look backwards from the match to find where to actually start
            # Skip any single-word noise before it
            before_match = transcript[:match.start()]
            # Find last sentence-like structure before the match
            clean_start = max(
                before_match.rfind('. ') + 2,
                before_match.rfind('! ') + 2,
                before_match.rfind('? ') + 2,
                match.start()
            )
            start_pos = clean_start
            print(f"✂️  Found content start at position {start_pos}")
            break
    
    if start_pos > 0:
        transcript = transcript[start_pos:]
    
    # Remove noise at the end (after last sentence)
    end_markers = [
        r'Thank you(\s+so\s+much)?\.?\s*$',
        r'See you (tomorrow|next time|soon)\.?\s*$',
        r'Goodbye\.?\s*$'
    ]
    
    for marker in end_markers:
        match = re.search(marker, transcript, re.IGNORECASE)
        if match:
            # Include the marker but cut everything after
            transcript = transcript[:match.end()]
            print(f"✂️  Found content end")
            break
    
    # Remove any remaining noise patterns at start/end
    transcript = re.sub(r'^([A-Z][a-z]*\.?\s*){1,50}(?=Welcome|Hello|Hi|Good)', '', transcript)
    transcript = re.sub(r'(([A-Z][a-z]*\.?\s*){5,}|Woo!?)$', '', transcript)
    
    # Clean up formatting
    transcript = normalize_text(transcript)
    
    cleaned_length = len(transcript)
    reduction = ((original_length - cleaned_length) / original_length) * 100 if original_length > 0 else 0
    
    # Format output
    output_content = f"{metadata}\n\n## Cleaned Transcript\n\n{transcript}"
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    print(f"\n✅ Original: {original_length:,} chars")
    print(f"✅ Cleaned: {cleaned_length:,} chars")
    print(f"✅ Reduced by: {reduction:.1f}%")
    print(f"✅ Saved to: {output_file}\n")

def normalize_text(text: str) -> str:
    """Normalize whitespace and formatting"""
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\.(\w)', r'. \1', text)
    text = re.sub(r',(\w)', r', \1', text)
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normalize spaces
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r' +\n', '\n', text)
    
    return text.strip()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python clean_transcript_ultra.py input.txt output.md")
        sys.exit(1)
    
    clean_transcript(sys.argv[1], sys.argv[2])
