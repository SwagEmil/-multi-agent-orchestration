# Data Preparation Guide - Knowledge Base Documentation

## Important Clarification: RAG vs Fine-Tuning

**You do NOT need to train a model for RAG.**

| Approach | What Happens | Cost | Use Case |
|----------|-------------|------|----------|
| **RAG (Our Approach)** | Documents are embedded and searched at runtime. Uses pre-trained embedding model (no training). | Low - only API calls for retrieval | Perfect for knowledge that changes frequently |
| **Fine-Tuning** | You train/update the LLM itself on your data. Expensive and time-consuming. | High - requires GPU training | When you need model to "internalize" knowledge |

**For your use case:** RAG is the right choice because:
- ✅ No training required
- ✅ Can update documentation anytime (just re-ingest)
- ✅ Transparent citations (you see which docs were used)
- ✅ Lower cost

---

## Phase -1: Data Preparation (Before Building)

### Step 1: Gather Documentation Sources

**What you need:**
- Official AI agent documentation (1000+ pages you mentioned)
- Best practices guides
- Concept explanations
- Code examples
- Troubleshooting guides

**Supported Formats:**
- ✅ PDF files
- ✅ Markdown (.md)
- ✅ Plain text (.txt)
- ✅ Word docs (will add converter if needed)

**Organization Structure:**
```
data/documents/
├── core_concepts/
│   ├── ai_agents_fundamentals.pdf
│   ├── orchestration_patterns.md
│   └── reasoning_frameworks.pdf
├── best_practices/
│   ├── error_handling.md
│   ├── prompt_engineering.pdf
│   └── code_analysis_guide.pdf
├── examples/
│   ├── case_studies.pdf
│   └── implementation_examples.md
└── troubleshooting/
    ├── common_bugs.md
    └── debugging_strategies.pdf
```

---

### Step 2: Validate Documentation Quality

Create a validation script to check:

#### Quality Metrics

**✅ 1. Readability Check**
- Text extraction works properly
- No garbled characters or encoding issues
- PDFs are searchable (not scanned images)

**✅ 2. Content Structure**
- Documents have clear headings
- Sentences are complete
- Paragraphs are coherent

**✅ 3. Relevance Check**
- Content is about AI agents, coding, research, etc.
- Not generic/unrelated material

**✅ 4. De-duplication**
- No duplicate documents
- Minimal redundant content

---

### Step 3: Run Validation Script

Here's the validation tool we'll build:

```python
#!/usr/bin/env python3
"""
Validate documentation quality before ingestion.

Usage:
    python scripts/validate_documentation.py --docs-dir data/documents/
"""

import argparse
from pathlib import Path
from PyPDF2 import PdfReader
import re
from collections import Counter

class DocumentValidator:
    def __init__(self):
        self.issues = []
        self.stats = {
            'total_files': 0,
            'total_pages': 0,
            'readable_files': 0,
            'failed_files': 0,
            'total_characters': 0
        }
    
    def validate_pdf(self, pdf_path):
        """Validate a single PDF file"""
        try:
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            
            # Test text extraction on first page
            first_page_text = reader.pages[0].extract_text()
            
            # Check for issues
            if len(first_page_text.strip()) < 50:
                self.issues.append({
                    'file': pdf_path.name,
                    'issue': 'Very short or empty first page (possible scanned PDF)',
                    'severity': 'high'
                })
                return False
            
            # Check for garbled text (too many special chars)
            special_char_ratio = sum(1 for c in first_page_text if not c.isalnum() and not c.isspace()) / len(first_page_text)
            if special_char_ratio > 0.3:
                self.issues.append({
                    'file': pdf_path.name,
                    'issue': 'High ratio of special characters (possible encoding issue)',
                    'severity': 'medium'
                })
            
            # Extract all text
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text()
            
            self.stats['total_pages'] += num_pages
            self.stats['total_characters'] += len(full_text)
            self.stats['readable_files'] += 1
            
            return True
            
        except Exception as e:
            self.issues.append({
                'file': pdf_path.name,
                'issue': f'Failed to read PDF: {str(e)}',
                'severity': 'high'
            })
            self.stats['failed_files'] += 1
            return False
    
    def validate_markdown(self, md_path):
        """Validate a Markdown file"""
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if len(text.strip()) < 100:
                self.issues.append({
                    'file': md_path.name,
                    'issue': 'File too short (< 100 characters)',
                    'severity': 'medium'
                })
            
            # Check for headings
            headings = re.findall(r'^#+\s+.+$', text, re.MULTILINE)
            if len(headings) == 0:
                self.issues.append({
                    'file': md_path.name,
                    'issue': 'No headings found (may lack structure)',
                    'severity': 'low'
                })
            
            self.stats['total_characters'] += len(text)
            self.stats['readable_files'] += 1
            return True
            
        except UnicodeDecodeError:
            self.issues.append({
                'file': md_path.name,
                'issue': 'Encoding error (not UTF-8)',
                'severity': 'high'
            })
            self.stats['failed_files'] += 1
            return False
    
    def validate_directory(self, directory_path):
        """Validate all documents in directory"""
        print("🔍 Validating documentation...\n")
        
        for file_path in Path(directory_path).rglob('*'):
            if file_path.is_file():
                self.stats['total_files'] += 1
                
                if file_path.suffix == '.pdf':
                    print(f"  Checking: {file_path.name}")
                    self.validate_pdf(file_path)
                elif file_path.suffix in ['.md', '.txt']:
                    print(f"  Checking: {file_path.name}")
                    self.validate_markdown(file_path)
        
        self.print_report()
    
    def print_report(self):
        """Print validation report"""
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)
        
        print(f"\n📊 STATISTICS:")
        print(f"  Total files: {self.stats['total_files']}")
        print(f"  Readable files: {self.stats['readable_files']}")
        print(f"  Failed files: {self.stats['failed_files']}")
        print(f"  Total pages (PDFs): {self.stats['total_pages']}")
        print(f"  Total characters: {self.stats['total_characters']:,}")
        print(f"  Estimated chunks: ~{self.stats['total_characters'] // 500}")
        
        if self.issues:
            print(f"\n⚠️  ISSUES FOUND ({len(self.issues)}):")
            
            # Group by severity
            high = [i for i in self.issues if i['severity'] == 'high']
            medium = [i for i in self.issues if i['severity'] == 'medium']
            low = [i for i in self.issues if i['severity'] == 'low']
            
            if high:
                print(f"\n  🔴 HIGH SEVERITY ({len(high)}):")
                for issue in high:
                    print(f"    - {issue['file']}: {issue['issue']}")
            
            if medium:
                print(f"\n  🟡 MEDIUM SEVERITY ({len(medium)}):")
                for issue in medium:
                    print(f"    - {issue['file']}: {issue['issue']}")
            
            if low:
                print(f"\n  🟢 LOW SEVERITY ({len(low)}):")
                for issue in low:
                    print(f"    - {issue['file']}: {issue['issue']}")
            
            # Recommendations
            print("\n📝 RECOMMENDATIONS:")
            if high:
                print("  - Fix HIGH severity issues before ingestion")
                print("  - Scanned PDFs need OCR processing")
                print("  - Files with encoding errors should be re-saved as UTF-8")
            if medium:
                print("  - Review MEDIUM issues - may affect retrieval quality")
            
        else:
            print("\n✅ No issues found! Documentation looks good.")
        
        print("\n" + "="*60)
        
        # Final verdict
        if self.stats['failed_files'] == 0:
            print("✅ READY FOR INGESTION")
        else:
            print(f"⚠️  FIX {self.stats['failed_files']} FAILED FILES BEFORE PROCEEDING")
        
        return len(self.issues) == 0

def main():
    parser = argparse.ArgumentParser(description="Validate documentation quality")
    parser.add_argument("--docs-dir", required=True, help="Directory containing documentation")
    args = parser.parse_args()
    
    validator = DocumentValidator()
    validator.validate_directory(args.docs_dir)

if __name__ == "__main__":
    main()
```

**Expected Output:**
```
🔍 Validating documentation...

  Checking: ai_agents_fundamentals.pdf
  Checking: orchestration_patterns.md
  Checking: reasoning_frameworks.pdf

============================================================
VALIDATION REPORT
============================================================

📊 STATISTICS:
  Total files: 5
  Readable files: 5
  Failed files: 0
  Total pages (PDFs): 1,247
  Total characters: 2,847,392
  Estimated chunks: ~5,694

✅ No issues found! Documentation looks good.

============================================================
✅ READY FOR INGESTION
```

---

### Step 4: Test Retrieval Quality

After ingestion, test if the system finds relevant docs:

```python
#!/usr/bin/env python3
"""
Test retrieval quality on sample queries.

Usage:
    python scripts/test_retrieval.py
"""

from knowledge_base.vector_store import VectorStore

def test_retrieval():
    vector_store = VectorStore()
    
    # Test queries that agents might ask
    test_queries = [
        "How should code agents handle bugs?",
        "Best practices for orchestrating multiple agents",
        "Error handling strategies for AI agents",
        "How to implement chain-of-thought reasoning",
        "Research agent web search guidelines"
    ]
    
    print("🧪 Testing Retrieval Quality\n")
    print("="*60)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        results = vector_store.search(query, n_results=3)
        
        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', 'Unknown')
            score = result['relevance_score']
            snippet = result['text'][:100] + "..."
            
            print(f"\n   {i}. [{source}] (Score: {score:.2f})")
            print(f"      {snippet}")
        
        print("\n" + "-"*60)

if __name__ == "__main__":
    test_retrieval()
```

**Expected Output:**
```
🧪 Testing Retrieval Quality

============================================================

📝 Query: How should code agents handle bugs?
   Found 3 results:

   1. [best_practices.pdf] (Score: 0.87)
      When code agents identify bugs, they should immediately log the bug details including severity, context, and...

   2. [error_handling.md] (Score: 0.82)
      Bug handling workflow: 1. Detect the issue, 2. Categorize severity, 3. Document code context, 4. Trigger...

   3. [troubleshooting.md] (Score: 0.76)
      Common bug patterns in AI agent systems include race conditions, improper error propagation, and...
```

---

## Your Action Items (Before Building)

### Checklist:

- [ ] **Gather all documentation files** (PDFs, Markdown, etc.)
- [ ] **Organize into directories** (by topic/category)
- [ ] **Run validation script** to check quality
- [ ] **Fix any high-severity issues** (scanned PDFs, encoding errors)
- [ ] **Review medium-severity warnings** (optional fixes)
- [ ] **Place validated docs in** `data/documents/`
- [ ] **Confirm you're ready** for ingestion

---

## What Happens Next (After Validation)

Once your docs are validated:

1. **Ingestion** (one-time setup):
   ```bash
   python scripts/ingest_documentation.py --docs-dir data/documents/
   ```
   - Chunks your docs (~500 chars each)
   - Generates embeddings (using pre-trained model, no training!)
   - Stores in ChromaDB vector database

2. **Testing** (verify retrieval quality):
   ```bash
   python scripts/test_retrieval.py
   ```
   - Ensures agents can find relevant docs

3. **Build the system** (Phase 0-4)
   - Agents automatically query knowledge base before every task

---

## FAQ

**Q: Do I need to label or annotate the documents?**  
A: No! RAG works with raw documentation. Just organize by topic for easier management.

**Q: What if I add more documentation later?**  
A: Just re-run the ingestion script. It will update the vector database.

**Q: How do I know if retrieval is working well?**  
A: The test script shows you what docs agents will see. If results look irrelevant, we can tune chunking size.

**Q: What about proprietary/sensitive information?**  
A: Everything stays local (ChromaDB is embedded). No data leaves your machine.

---

## Timeline Estimate

| Task | Time |
|------|------|
| Gather documentation | 1-2 days (depends on your sources) |
| Run validation | 5 minutes |
| Fix issues (if any) | 30 min - 2 hours |
| Ingest into vector DB | 10-30 minutes (depends on size) |
| Test retrieval quality | 15 minutes |
| **TOTAL** | **2-3 days** |

---

**Ready when you are!** Let me know when you have your documentation gathered, and I'll help you validate and ingest it.
