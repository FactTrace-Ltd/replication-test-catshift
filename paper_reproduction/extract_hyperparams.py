#!/usr/bin/env python3
import subprocess
import sys

# Try to import pdfplumber, install if needed
try:
    import pdfplumber
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber", "-q"])
    import pdfplumber

with pdfplumber.open('2026.findings-acl.353.pdf') as pdf:
    # Search through all pages for hyperparameters
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if any(keyword in text.lower() for keyword in ['learning rate', 'epochs', 'batch', 'lora', 'hyperparameter', 'fine.tun', 'training']):
            if any(kw in text.lower() for kw in ['8e-05', '8e-5', '2e-5', 'adam', 'batch size']):
                print(f"\n{'='*60}")
                print(f"PAGE {i+1}")
                print(f"{'='*60}\n")
                print(text)
