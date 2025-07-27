#!/usr/bin/env python3
"""
Script to merge all ground_truth.jsonl files from Error_data/{category}/{sub_error}/ 
into a single JSONL file in SSS-data directory.
"""

import os
import json
from pathlib import Path

def find_ground_truth_files(base_path):
    """
    Find all ground_truth.jsonl files in the directory structure.
    """
    ground_truth_files = []
    
    def search_recursive(current_path, path_parts=[]):
        for item in current_path.iterdir():
            if not item.is_dir():
                if item.name == "ground_truth.jsonl":
                    ground_truth_files.append({
                        'path': item,
                        'path_parts': path_parts,
                        'category': path_parts[-1] if path_parts else 'Unknown',
                        'parent_dir': path_parts[-2] if len(path_parts) >= 2 else 'Unknown'
                    })
                continue
                
            current_parts = path_parts + [item.name]
            search_recursive(item, current_parts)
    
    search_recursive(base_path)
    return ground_truth_files

def merge_ground_truth_files():
    """
    Merge all ground_truth.jsonl files into a single file.
    """
    # Source and destination
    source_base = Path("Error_injection/Error_data")
    dest_dir = Path("SSS-data")
    output_file = dest_dir / "merged_ground_truth.jsonl"
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    total_entries = 0
    processed_files = 0
    failed_files = []
    file_stats = []
    
    if not source_base.exists():
        print(f"❌ Source directory '{source_base}' does not exist!")
        return
    
    print(f"🔍 Scanning {source_base} for ground_truth.jsonl files...")
    print(f"📁 Output file: {output_file}")
    print("-" * 60)
    
    # Find all ground_truth.jsonl files
    ground_truth_files = find_ground_truth_files(source_base)
    
    if not ground_truth_files:
        print("❌ No ground_truth.jsonl files found!")
        return
    
    print(f"✅ Found {len(ground_truth_files)} ground_truth.jsonl files")
    print()
    
    # Open output file for writing
    with open(output_file, 'w', encoding='utf-8') as outfile:
        
        # Process each ground_truth.jsonl file
        for file_info in ground_truth_files:
            file_path = file_info['path']
            category = file_info['category']
            parent_dir = file_info['parent_dir']
            path_str = '/'.join(file_info['path_parts'])
            
            print(f"📄 Processing: {path_str}/ground_truth.jsonl")
            
            try:
                entries_count = 0
                
                # Read and process the JSONL file
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line_num, line in enumerate(infile, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            # Parse JSON line
                            entry = json.loads(line)
                            
                            # # Add metadata about source file
                            # entry['source_category'] = category
                            # entry['source_parent'] = parent_dir
                            # entry['source_path'] = path_str
                            # entry['source_file'] = str(file_path)
                            
                            # Write to output file
                            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
                            entries_count += 1
                            total_entries += 1
                            
                        except json.JSONDecodeError as e:
                            print(f"    ⚠️  Line {line_num}: Invalid JSON - {e}")
                            continue
                
                print(f"    ✅ Processed {entries_count} entries")
                processed_files += 1
                
                file_stats.append({
                    'path': path_str,
                    'category': category,
                    'entries': entries_count
                })
                
            except FileNotFoundError:
                print(f"    ❌ File not found: {file_path}")
                failed_files.append(str(file_path))
            except PermissionError:
                print(f"    ❌ Permission denied: {file_path}")
                failed_files.append(str(file_path))
            except Exception as e:
                print(f"    ❌ Error processing {file_path}: {e}")
                failed_files.append(str(file_path))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 MERGE SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully processed: {processed_files} files")
    print(f"❌ Failed to process: {len(failed_files)} files")
    print(f"📝 Total entries merged: {total_entries}")
    print(f"📁 Output file: {output_file.absolute()}")
    print(f"📏 Output file size: {output_file.stat().st_size:,} bytes" if output_file.exists() else "Output file not created")
    
    if failed_files:
        print(f"\n❌ Failed files:")
        for failed_file in failed_files:
            print(f"   - {failed_file}")
    
    if file_stats:
        print(f"\n📊 Entries by category:")
        category_totals = {}
        for stat in file_stats:
            cat = stat['category']
            if cat not in category_totals:
                category_totals[cat] = 0
            category_totals[cat] += stat['entries']
        
        for category, count in sorted(category_totals.items()):
            print(f"   {category}: {count} entries")
        
        print(f"\n📋 Detailed breakdown:")
        for stat in file_stats:
            print(f"   {stat['path']}: {stat['entries']} entries")

def validate_merged_file():
    """
    Validate the merged JSONL file by checking if all lines are valid JSON.
    """
    output_file = Path("SSS-data/merged_ground_truth.jsonl")
    
    if not output_file.exists():
        print("❌ Merged file does not exist!")
        return False
    
    print(f"\n🔍 Validating merged file: {output_file}")
    
    try:
        valid_lines = 0
        invalid_lines = 0
        
        with open(output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    json.loads(line)
                    valid_lines += 1
                except json.JSONDecodeError as e:
                    print(f"    ❌ Line {line_num}: Invalid JSON - {e}")
                    invalid_lines += 1
        
        print(f"✅ Validation complete:")
        print(f"   Valid lines: {valid_lines}")
        print(f"   Invalid lines: {invalid_lines}")
        
        return invalid_lines == 0
        
    except Exception as e:
        print(f"❌ Error validating file: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Starting ground_truth.jsonl merge process...")
    print("📋 Task: Merge all Error_data/{category}/{sub_error}/ground_truth.jsonl files")
    print()
    
    try:
        merge_ground_truth_files()
        
        # Validate the merged file
        if validate_merged_file():
            print("\n🎉 Merge and validation completed successfully!")
        else:
            print("\n⚠️  Merge completed but validation found issues")
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()