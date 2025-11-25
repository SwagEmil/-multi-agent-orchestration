#!/usr/bin/env python3
"""
Extract text from PDFs and convert to RAG-ready markdown format.

Usage:
    python scripts/extract_pdfs_for_rag.py --input data/processed/final --output data/final
"""

import argparse
from pathlib import Path
import PyPDF2
import re

class PDFToRAGConverter:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """Clean extracted PDF text"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers (lines with just numbers)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        
        # Fix hyphenation at line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Remove isolated special characters
        text = re.sub(r'\n[^\w\s]{1,3}\n', '\n', text)
        
        return text.strip()
    
    def extract_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF"""
        print(f"\n📄 Processing: {pdf_path.name}")
        
        text_content = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                print(f"   Pages: {total_pages}")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                
                full_text = '\n\n'.join(text_content)
                cleaned_text = self.clean_text(full_text)
                
                print(f"   Extracted: {len(cleaned_text):,} characters")
                
                return cleaned_text
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def convert_to_markdown(self, pdf_path: Path, extracted_text: str):
        """Convert extracted text to RAG-ready markdown"""
        
        # Create markdown filename
        md_filename = pdf_path.stem + '.md'
        output_path = self.output_dir / md_filename
        
        # Create markdown document with metadata
        markdown_content = f"""# {pdf_path.stem.replace('_', ' ').title()}

**Source:** {pdf_path.name}  
**Type:** PDF Document  
**Format:** Extracted and cleaned for RAG ingestion

---

{extracted_text}
"""
        
        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"   ✅ Saved to: {output_path.name}")
        
        return output_path
    
    def process_all_pdfs(self):
        """Process all PDFs in input directory"""
        
        print(f"\n{'='*70}")
        print(f"PDF TO RAG CONVERTER")
        print(f"{'='*70}\n")
        print(f"Input:  {self.input_dir}")
        print(f"Output: {self.output_dir}")
        
        # Find all PDFs
        pdf_files = list(self.input_dir.glob('*.pdf'))
        
        if not pdf_files:
            print(f"\n⚠️  No PDF files found in {self.input_dir}")
            return
        
        print(f"\n📊 Found {len(pdf_files)} PDF files\n")
        
        converted = 0
        failed = 0
        
        for pdf_path in pdf_files:
            extracted_text = self.extract_pdf(pdf_path)
            
            if extracted_text:
                self.convert_to_markdown(pdf_path, extracted_text)
                converted += 1
            else:
                failed += 1
        
        # Summary
        print(f"\n{'='*70}")
        print(f"CONVERSION COMPLETE")
        print(f"{'='*70}\n")
        print(f"✅ Converted: {converted} PDFs")
        if failed > 0:
            print(f"❌ Failed: {failed} PDFs")
        print(f"\n💡 RAG-ready files saved to: {self.output_dir}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Extract PDFs and convert to RAG-ready markdown'
    )
    parser.add_argument(
        '--input',
        default='data/processed/final',
        help='Input directory containing PDFs'
    )
    parser.add_argument(
        '--output',
        default='data/final',
        help='Output directory for markdown files'
    )
    
    args = parser.parse_args()
    
    converter = PDFToRAGConverter(args.input, args.output)
    converter.process_all_pdfs()

if __name__ == "__main__":
    main()
