#!/usr/bin/env python3
"""
Clean extracted PDF markdown files for RAG ingestion.
Removes headers, footers, TOC, page numbers, and other PDF artifacts.

Usage:
    python scripts/clean_pdf_extracts.py --input data/final --output data/final
"""

import argparse
from pathlib import Path
import re

class PDFCleaner:
    def __init__(self):
        # Common PDF artifacts to remove
        self.header_patterns = [
            r'^Agents Companion$',
            r'^February \d{4}\d+$',
            r'^Embeddings & Vector Stores\d+$',
            r'^Prompt Engineering\d+$',
            r'^Solving Domain Specific Problems.*\d+$',
            r'^Operationalizing.*\d+$',
            r'^Introduction to Agents\d+$',
        ]
        
        self.removal_patterns = [
            # Table of contents entries
            r'^(Introduction|Acknowledgements|Table of contents|Endnotes|Summary)\s*\d*$',
            # Page numbers at end of lines
            r'\d+$',
            # Isolated numbers on their own line (page numbers)
            r'^\d{1,3}$',
            # "Figure X:" captions without content
            r'^Figure \d+:?\s*$',
            # "Table X:" captions without content  
            r'^Table \d+:?\s*$',
        ]
    
    def is_toc_line(self, line: str) -> bool:
        """Check if line is part of table of contents"""
        # TOC patterns: "Topic Name 42" or "Topic Name...... 42"
        if re.match(r'^[A-Z].*\d{1,3}$', line.strip()):
            return True
        if '...' in line and line.strip()[-2:].isdigit():
            return True
        return False
    
    def clean_text(self, text: str, filename: str) -> str:
        """Clean PDF extracted text"""
        lines = text.split('\n')
        cleaned_lines = []
        in_toc = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                cleaned_lines.append('')
                continue
            
            # Detect TOC start
            if 'Table of contents' in stripped or 'Contents' in stripped:
                in_toc = True
                continue
            
            # Detect TOC end (usually when real content starts)
            if in_toc and len(stripped) > 100:  # Long lines = content, not TOC
                in_toc = False
            
            # Skip TOC lines
            if in_toc or self.is_toc_line(stripped):
                continue
            
            # Skip header patterns
            skip = False
            for pattern in self.header_patterns:
                if re.match(pattern, stripped, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue
            
            # Skip removal patterns
            skip = False
            for pattern in self.removal_patterns:
                if re.match(pattern, stripped, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue
            
            # Remove inline page numbers at end of line
            line = re.sub(r'\s+\d{1,3}$', '', line)
            
            cleaned_lines.append(line)
        
        # Join and clean excessive newlines
        result = '\n'.join(cleaned_lines)
        result = re.sub(r'\n{4,}', '\n\n\n', result)
        
        return result.strip()
    
    def process_file(self, input_path: Path, output_path: Path):
        """Clean a single markdown file"""
        print(f"\n📄 Cleaning: {input_path.name}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_size = len(content)
        
        # Clean
        cleaned = self.clean_text(content, input_path.name)
        
        cleaned_size = len(cleaned)
        reduction = ((original_size - cleaned_size) / original_size) * 100
        
        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        
        print(f"   Original: {original_size:,} chars")
        print(f"   Cleaned:  {cleaned_size:,} chars")
        print(f"   Reduced:  {reduction:.1f}%")
        print(f"   ✅ Saved to: {output_path.name}")
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all PDF-extracted markdown files"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"PDF MARKDOWN CLEANER")
        print(f"{'='*70}\n")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        
        # Find PDF-extracted files (exclude livestream transcripts)
        exclude_patterns = ['day2_', 'day3_', 'day4_', 'day5_']
        md_files = []
        
        for f in input_path.glob('*.md'):
            if not any(f.name.startswith(pattern) for pattern in exclude_patterns):
                md_files.append(f)
        
        if not md_files:
            print(f"\n⚠️  No PDF-extracted markdown files found")
            return
        
        print(f"\n📊 Found {len(md_files)} PDF files to clean\n")
        
        for md_file in md_files:
            output_file = output_path / md_file.name
            self.process_file(md_file, output_file)
        
        print(f"\n{'='*70}")
        print(f"CLEANING COMPLETE")
        print(f"{'='*70}\n")
        print(f"✅ Cleaned {len(md_files)} files")
        print(f"\n💡 Files ready for RAG ingestion\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/final', help='Input directory')
    parser.add_argument('--output', default='data/final', help='Output directory')
    
    args = parser.parse_args()
    
    cleaner = PDFCleaner()
    cleaner.process_directory(args.input, args.output)

if __name__ == "__main__":
    main()
