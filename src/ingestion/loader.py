import os
import fitz  # PyMuPDF

class DocumentLoader:
    """Handles loading documents from various raw sources (inline text, text files, PDFs, etc.)."""
    
    @staticmethod
    def load(sources: list[dict]) -> list[dict]:
        documents = []
        for source in sources:
            source_type = source.get('type')
            if source_type == 'text':
                doc = {
                    'content': source['content'],
                    'source': source.get('name', 'inline_text'),
                    'metadata': source.get('metadata', {})
                }
                documents.append(doc)
            elif source_type == 'file':
                path = source['path']
                if not os.path.exists(path):
                    print(f"Warning: File not found at {path}")
                    continue
                
                # Check for PDF extension
                if path.lower().endswith('.pdf'):
                    try:
                        print(f"[LOAD] Extracting text from PDF: {path}")
                        content_text = ""
                        with fitz.open(path) as doc_pdf:
                            num_pages = len(doc_pdf)
                            for page_num, page in enumerate(doc_pdf):
                                text = page.get_text()
                                if text:
                                    content_text += text + "\n"
                        
                        if not content_text.strip():
                            print(f"[LOAD] Warning: Extracted empty text from PDF: {path}")
                            
                        doc = {
                            'content': content_text,
                            'source': path,
                            'metadata': {
                                **source.get('metadata', {}),
                                'file_type': 'pdf',
                                'pages': num_pages
                            }
                        }
                        documents.append(doc)
                    except Exception as e:
                        print(f"Error loading PDF file {path}: {e}")
                else:
                    # Treat as standard text file
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            doc = {
                                'content': f.read(),
                                'source': path,
                                'metadata': {
                                    **source.get('metadata', {}),
                                    'file_type': 'txt'
                                }
                            }
                            documents.append(doc)
                    except Exception as e:
                        print(f"Error loading text file {path}: {e}")
            else:
                print(f"Warning: Unsupported source type '{source_type}'")
        
        print(f"[LOAD] Loaded {len(documents)} document(s)")
        return documents
