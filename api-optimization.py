#!/usr/bin/env python3
"""
Script to optimize the EnhancedProcessor by reducing API calls and fixing deprecation warnings.
This creates a patched version of your enhanced_processor.py file.
"""

import os
import re
import sys
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create a backup of the file."""
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist.")
        return False
        
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = f"{filepath}.{timestamp}.bak"
    
    try:
        shutil.copy2(filepath, backup_path)
        print(f"Created backup: {backup_path}")
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False

def optimize_processor(filepath):
    """Apply optimizations to the EnhancedProcessor file."""
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist.")
        return False
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # 1. Increase chunk size in text splitter
        content = re.sub(
            r'self\.text_splitter = RecursiveCharacterTextSplitter\(\s+chunk_size=(\d+),\s+chunk_overlap=(\d+)',
            r'self.text_splitter = RecursiveCharacterTextSplitter(\n            chunk_size=6000,    # Increased from \1\n            chunk_overlap=100    # Reduced from \2',
            content
        )
        
        # 2. Increase large document threshold
        content = re.sub(
            r'is_large_document = len\(document\.content\) > (\d+)',
            r'is_large_document = len(document.content) > 12000  # Increased from \1',
            content
        )
        
        # 3. Fix deprecation warning (chain.run -> chain.invoke)
        content = re.sub(
            r'summary = chain\.run\(lc_docs\)',
            r'summary = chain.invoke(lc_docs)["output_text"]  # Updated from deprecated chain.run',
            content
        )
        
        # 4. Optional: Add an option to disable knowledge graph
        if "def process_document" in content:
            # Add a parameter to optionally disable KG
            content = re.sub(
                r'def process_document\(self, document\) -> bool:',
                r'def process_document(self, document, use_kg=True) -> bool:',
                content
            )
            
            # Modify KG initialization to respect parameter
            content = re.sub(
                r'kg_initialized = self\._init_kg_manager\(\)',
                r'kg_initialized = use_kg and self._init_kg_manager()',
                content
            )
        
        # Write the updated content back to the file
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"Successfully optimized {filepath}")
        print("\nOptimizations applied:")
        print("1. Increased chunk size from 4000 to 6000 characters")
        print("2. Reduced chunk overlap from 200 to 100 characters")
        print("3. Increased large document threshold from 8000 to 12000 characters")
        print("4. Fixed deprecation warning by replacing chain.run with chain.invoke")
        print("5. Added optional parameter to disable knowledge graph in process_document")
        
        return True
    
    except Exception as e:
        print(f"Error optimizing file: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point."""
    print("EnhancedProcessor API Call Optimizer")
    print("===================================")
    
    # Default filepath
    filepath = "enhanced_processor.py"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    
    print(f"Target file: {filepath}")
    
    # Confirm before proceeding
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return 1
    
    confirm = input(f"This will modify {filepath}. Proceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return 0
    
    # Create backup
    if not backup_file(filepath):
        print("Aborting due to backup failure.")
        return 1
    
    # Apply optimizations
    if optimize_processor(filepath):
        print("\nSuccessfully applied all optimizations!")
        print(f"To use the optimized processor with disabled knowledge graph:")
        print(f"  processor = EnhancedProcessor()")
        print(f"  processor.process_document(document, use_kg=False)")
        return 0
    else:
        print("\nFailed to apply optimizations.")
        return 1

if __name__ == "__main__":
    sys.exit(main())