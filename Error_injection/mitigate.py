#!/usr/bin/env python3
"""
Script to copy all SQL files from Error_injection/{category}/{sub_error}/error_sqls 
to SSS-data/sql folder while preserving original files and names.
"""

import os
import shutil
from pathlib import Path

def copy_sql_files():
    """
    Copy all SQL files from Error_injection structure to SSS-data/sql folder.
    """
    # Source and destination directories
    source_base = Path("Error_injection")
    dest_dir = Path("SSS-data/sql")
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Track copied files and potential conflicts
    copied_files = []
    conflicts = []
    
    # Walk through the Error_injection directory structure
    if not source_base.exists():
        print(f"❌ Source directory '{source_base}' does not exist!")
        return
    
    print(f"🔍 Scanning {source_base} for SQL files...")
    print(f"📁 Destination: {dest_dir}")
    print("-" * 50)
    
    # Iterate through categories
    for category_dir in source_base.iterdir():
        if not category_dir.is_dir():
            continue
            
        print(f"📂 Category: {category_dir.name}")
        
        # Iterate through sub-error types
        for sub_error_dir in category_dir.iterdir():
            if not sub_error_dir.is_dir():
                continue
                
            print(f"  📂 Sub-error: {sub_error_dir.name}")
            
            # Look for error_sqls folder
            error_sqls_dir = sub_error_dir / "error_sqls"
            if not error_sqls_dir.exists() or not error_sqls_dir.is_dir():
                print(f"    ⚠️  No 'error_sqls' folder found")
                continue
            
            # Copy all SQL files from error_sqls folder
            sql_files = list(error_sqls_dir.glob("*.sql"))
            if not sql_files:
                print(f"    ℹ️  No SQL files found in error_sqls")
                continue
                
            print(f"    📄 Found {len(sql_files)} SQL file(s)")
            
            for sql_file in sql_files:
                dest_file = dest_dir / sql_file.name
                
                # Check if file already exists with same name
                if dest_file.exists():
                    conflicts.append({
                        'original': str(sql_file),
                        'destination': str(dest_file),
                        'name': sql_file.name
                    })
                    
                    # Create a unique name by adding category and sub-error prefix
                    unique_name = f"{category_dir.name}_{sub_error_dir.name}_{sql_file.name}"
                    dest_file = dest_dir / unique_name
                    print(f"      ⚠️  Conflict! Renaming to: {unique_name}")
                
                try:
                    # Copy the file
                    shutil.copy2(sql_file, dest_file)
                    copied_files.append({
                        'source': str(sql_file),
                        'destination': str(dest_file),
                        'category': category_dir.name,
                        'sub_error': sub_error_dir.name
                    })
                    print(f"      ✅ Copied: {sql_file.name}")
                    
                except Exception as e:
                    print(f"      ❌ Error copying {sql_file.name}: {e}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 COPY SUMMARY")
    print("=" * 50)
    print(f"✅ Successfully copied: {len(copied_files)} files")
    print(f"⚠️  Name conflicts: {len(conflicts)} files")
    print(f"📁 Destination: {dest_dir.absolute()}")
    
    if conflicts:
        print("\n🔄 Files renamed due to conflicts:")
        for conflict in conflicts:
            print(f"   - {conflict['name']} (from {Path(conflict['original']).parent.parent.name}/{Path(conflict['original']).parent.name})")
    
    if copied_files:
        print(f"\n📋 All copied files are now in: {dest_dir.absolute()}")
        
        # Optional: Print breakdown by category
        categories = {}
        for file_info in copied_files:
            cat = file_info['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(file_info)
        
        print("\n📊 Files by category:")
        for category, files in categories.items():
            print(f"   {category}: {len(files)} files")

def main():
    """Main function"""
    print("🚀 Starting SQL file copy process...")
    print("📋 Task: Copy Error_injection/{category}/{sub_error}/error_sqls/*.sql to SSS-data/sql/")
    print()
    
    try:
        copy_sql_files()
        print("\n🎉 Copy process completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()