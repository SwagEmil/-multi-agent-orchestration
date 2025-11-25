#!/usr/bin/env python3
"""
AGGRESSIVE PDF cleaning - removes all metadata, headers, acknowledgements, etc.
Only keeps technical content.

Usage:
    python scripts/clean_pdf_aggressive.py --input data/final --output data/final
"""

import argparse
from pathlib import Path
import re

class AggressivePDFCleaner:
    def __init__(self):
        # Sections to completely remove
        self.skip_sections = [
            'acknowledgements',
            'table of contents',
            'contents',
            'endnotes',
            'authors:',
            'editors & curators',
            'content contributors',
            'designer',
            'about the authors',
            'contributors',
        ]
        
        # Line patterns to remove
        self.remove_patterns = [
            # Month/Year stamps
            r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
            # Document titles with numbers
            r'^(Agents Companion|Agents|Embeddings & Vector Stores|Prompt Engineering|Introduction to Agents|Solving Domain.*|Operationalizing.*)\s*\d*$',
            # Page numbers
            r'^\d{1,3}$',
            # Figure/Table captions alone
            r'^(Figure|Table)\s+\d+:?\s*$',
            # Designer/Author credits
            r'^(Designer|Author|Editor|Curator)s?:',
            # Names (Author: First Last)
            r'^[A-Z][a-z]+ [A-Z][a-z]+( [A-Z][a-z]+)?$',
            # Isolated metadata
            r'^(Haymaker|Michael Lanning|Steven Johnson|Hussain Chinoy)$',
        ]
    
    def should_skip_line(self, line: str) -> bool:
        """Check if line should be skipped"""
        stripped = line.strip().lower()
        
        # Skip empty
        if not stripped:
            return False
        
        # Skip section headers we don't want
        for section in self.skip_sections:
            if section in stripped:
                return True
        
        # Skip patterns
        for pattern in self.remove_patterns:
            if re.search(pattern, line.strip(), re.IGNORECASE):
                return True
        
        return False
    
    def clean_text(self, text: str) -> str:
        """Aggressively clean PDF text"""
        lines = text.split('\n')
        cleaned_lines = []
        skip_until_content = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Keep empty lines for paragraph breaks
            if not stripped:
                # Don't add multiple consecutive empty lines
                if cleaned_lines and cleaned_lines[-1] != '':
                    cleaned_lines.append('')
                continue
            
            # Skip lines we don't want
            if self.should_skip_line(line):
                continue
            
            # Skip very short lines (likely artifacts)
            if len(stripped) < 3 and not stripped.isdigit():
                continue
            
            # Remove inline artifacts
            # Remove February 2025 type stamps within lines
            line = re.sub(r'\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}', '', line)
            
            # Remove inline page numbers at line end
            line = re.sub(r'\s+\d{1,3}$', '', line)
            
            # Remove document title fragments
            line = re.sub(r'(Agents Companion|Embeddings & Vector Stores|Prompt Engineering)', '', line)
            
            cleaned_lines.append(line.strip())
        
        # Join and clean up
        result = '\n'.join(cleaned_lines)
        
        # Remove excessive newlines (more than 2)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        # Remove leading/trailing whitespace
        result = result.strip()
        
        return result
    
    def extract_title_from_filename(self, filename: str) -> str:
        """Extract clean title from filename"""
        # Remove .md extension
        title = filename.replace('.md', '')
        # Replace special chars
        title = title.replace('&', 'and')
        # Title case
        title = title.title()
        return title
    
    def process_file(self, input_path: Path, output_path: Path):
        """Clean a single file"""
        print(f"\n📄 Deep cleaning: {input_path.name}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_size = len(content)
        
        # Clean
        cleaned = self.clean_text(content)
        
        # Create new document with minimal metadata
        title = self.extract_title_from_filename(input_path.name)
        
        final_content = f"""# {title}

**Type:** Technical Documentation  
**Format:** Cleaned for RAG ingestion

---

{cleaned}
"""
        
        final_size = len(final_content)
        reduction = ((original_size - final_size) / original_size) * 100
        
        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print(f"   Original: {original_size:,} chars")
        print(f"   Cleaned:  {final_size:,} chars")
        print(f"   Reduced:  {reduction:.1f}%")
        print(f"   ✅ Saved")
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all PDF files"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"AGGRESSIVE PDF CLEANER")
        print(f"{'='*70}\n")
        print(f"Removing: metadata, headers, acknowledgements, artifacts...")
        print(f"Keeping: technical content only\n")
        
        # Exclude livestream files
        exclude = ['day2_', 'day3_', 'day4_', 'day5_']
        md_files = [f for f in input_path.glob('*.md') 
                   if not any(f.name.startswith(p) for p in exclude)]
        
        if not md_files:
            print(f"\n⚠️  No PDF files found")
            return
        
        print(f"📊 Found {len(md_files)} PDF files\n")
        
        for md_file in md_files:
            output_file = output_path / md_file.name
            self.process_file(md_file, output_file)
        
        print(f"\n{'='*70}")
        print(f"DEEP CLEANING COMPLETE")
        print(f"{'='*70}\n")
        print(f"✅ Cleaned {len(md_files)} files")
        print(f"\n💡 Technical content extracted, ready for RAG\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/final')
    parser.add_argument('--output', default='data/final')
    args = parser.parse_args()
    
    cleaner = AggressivePDFCleaner()
    cleaner.process_directory(args.input, args.output)

if __name__ == "__main__":
    main()
