"""Dataset utility functions for ordinal classifier."""

import os
import hashlib
import shutil
import random
from pathlib import Path
from collections import defaultdict


def get_md5_hash(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def is_image_file(file_path):
    """Check if file is an image based on extension."""
    return file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']


def rename_to_md5(data_dir):
    """
    Rename all image files to their MD5 hash and deduplicate.
    - Scans all images in train/ and valid/ folders.
    - If multiple files have the same content (MD5 hash), it keeps one
      and deletes the others.
    - Renames the remaining unique files to {hash}.jpg, normalizing extensions.
    """
    data_dir = Path(data_dir)
    print("🔄 Scanning for duplicates and renaming files to MD5 hashes...")

    # 1. Collect all image files and their hashes
    all_images_by_hash = defaultdict(list)
    for split_dir in ['train', 'valid']:
        split_path = data_dir / split_dir
        if not split_path.exists():
            continue
        
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            print(f"  Scanning {split_dir}/{class_dir.name}...")
            for file_path in class_dir.iterdir():
                if is_image_file(file_path):
                    try:
                        file_hash = get_md5_hash(file_path)
                        all_images_by_hash[file_hash].append(file_path)
                    except Exception as e:
                        print(f"  ! Could not process file {file_path}, skipping. Error: {e}")

    # 2. Deduplicate files with the same hash
    deleted_count = 0
    print("\n🔎 Checking for duplicate files...")
    for file_hash, file_paths in all_images_by_hash.items():
        if len(file_paths) > 1:
            # Keep the first file, delete the rest as duplicates
            files_to_keep = file_paths[:1]
            files_to_delete = file_paths[1:]
            print(f"  Found {len(file_paths)} files with hash {file_hash}. Keeping one, deleting {len(files_to_delete)}.")
            for f in files_to_delete:
                print(f"    - Deleting duplicate: {f.relative_to(data_dir)}")
                try:
                    f.unlink()
                    deleted_count += 1
                except OSError as e:
                    print(f"    ! Error deleting file {f}: {e}")
            # Update the list to only contain the file we're keeping
            all_images_by_hash[file_hash] = files_to_keep

    # 3. Rename remaining unique files to their MD5 hash
    renamed_count = 0
    print("\n🔄 Renaming files to canonical MD5 format...")
    for file_hash, file_paths in all_images_by_hash.items():
        if not file_paths:
            continue
        
        canonical_file = file_paths[0]
        expected_name = f"{file_hash}.jpg"
        
        # If the file is not in the correct format, rename it
        if canonical_file.name != expected_name:
            new_path = canonical_file.parent / expected_name
            
            # This handles a rare case where a file with a different content
            # might already have the name we want to use. We remove the source
            # file in this case, as it's a content duplicate of another file.
            if new_path.exists():
                print(f"  ! Deleting source file {canonical_file.name} as its target name {new_path.name} is already taken by a different file's content.")
                canonical_file.unlink()
                deleted_count += 1
            else:
                try:
                    canonical_file.rename(new_path)
                    renamed_count += 1
                except OSError as e:
                     print(f"  ! Error renaming file {canonical_file.name}: {e}")

    print(f"\n✅ Renamed {renamed_count} files and removed {deleted_count} duplicates.")
    return renamed_count, deleted_count


def count_files_by_class(data_dir):
    """Count files in each class across train/valid splits."""
    data_dir = Path(data_dir)
    counts = defaultdict(lambda: {'train': 0, 'valid': 0, 'total': 0})
    
    for split_dir in ['train', 'valid']:
        split_path = data_dir / split_dir
        if not split_path.exists():
            continue
            
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            file_count = len([f for f in class_dir.iterdir() if is_image_file(f)])
            counts[class_name][split_dir] = file_count
            counts[class_name]['total'] += file_count
    
    return dict(counts)


def check_cross_split_duplicates(data_dir):
    """Check for duplicate files across train/valid splits."""
    data_dir = Path(data_dir)
    train_hashes = {}
    valid_hashes = {}
    duplicates = []
    
    print("🔍 Checking for duplicates across train/valid splits...")
    
    # Collect hashes from train split
    train_path = data_dir / 'train'
    if train_path.exists():
        for class_dir in train_path.iterdir():
            if not class_dir.is_dir():
                continue
            for file_path in class_dir.iterdir():
                if is_image_file(file_path):
                    try:
                        file_hash = get_md5_hash(file_path)
                        train_hashes[file_hash] = file_path
                    except Exception as e:
                        print(f"  ! Could not hash {file_path}: {e}")
    
    # Collect hashes from valid split and check for duplicates
    valid_path = data_dir / 'valid'
    if valid_path.exists():
        for class_dir in valid_path.iterdir():
            if not class_dir.is_dir():
                continue
            for file_path in class_dir.iterdir():
                if is_image_file(file_path):
                    try:
                        file_hash = get_md5_hash(file_path)
                        if file_hash in train_hashes:
                            duplicates.append({
                                'hash': file_hash,
                                'train_file': train_hashes[file_hash],
                                'valid_file': file_path
                            })
                        valid_hashes[file_hash] = file_path
                    except Exception as e:
                        print(f"  ! Could not hash {file_path}: {e}")
    
    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} duplicate files across train/valid splits:")
        for dup in duplicates:
            train_rel = dup['train_file'].relative_to(data_dir)
            valid_rel = dup['valid_file'].relative_to(data_dir)
            print(f"  Hash {dup['hash'][:8]}... appears in both:")
            print(f"    Train: {train_rel}")
            print(f"    Valid: {valid_rel}")
    else:
        print("✅ No duplicate files found across train/valid splits")
    
    return duplicates


def rebalance_dataset(data_dir, target_valid_ratio=0.2, seed=42, check_duplicates=True):
    """Rebalance dataset to achieve target train/validation split."""
    data_dir = Path(data_dir)
    random.seed(seed)
    
    if check_duplicates:
        duplicates = check_cross_split_duplicates(data_dir)
        if duplicates:
            print(f"\n⚠️  Warning: Found {len(duplicates)} duplicate files across splits.")
            print("Consider removing duplicates before rebalancing to avoid data leakage.")
    
    print(f"\n🎯 Rebalancing dataset to {int((1-target_valid_ratio)*100)}/{int(target_valid_ratio*100)} train/valid split...")
    
    # Get current file counts
    current_counts = count_files_by_class(data_dir)
    
    print("\nCurrent distribution:")
    total_files = 0
    for class_name, counts in current_counts.items():
        total = counts['total']
        train_pct = (counts['train'] / total * 100) if total > 0 else 0
        valid_pct = (counts['valid'] / total * 100) if total > 0 else 0
        print(f"  {class_name}: {total} total ({counts['train']} train {train_pct:.1f}%, {counts['valid']} valid {valid_pct:.1f}%)")
        total_files += total
    
    print(f"\nTotal files: {total_files}")
    
    # Collect all files by class
    all_files_by_class = defaultdict(list)
    
    for split_dir in ['train', 'valid']:
        split_path = data_dir / split_dir
        if not split_path.exists():
            continue
            
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            for file_path in class_dir.iterdir():
                if is_image_file(file_path):
                    all_files_by_class[class_name].append(file_path)
    
    # Shuffle and split each class
    moves_made = 0
    
    for class_name, files in all_files_by_class.items():
        if not files:
            continue
            
        # Shuffle files for random split
        random.shuffle(files)
        
        total_files_class = len(files)
        target_valid_count = max(1, int(total_files_class * target_valid_ratio))
        target_train_count = total_files_class - target_valid_count
        
        print(f"\n📊 {class_name}: {total_files_class} files → {target_train_count} train, {target_valid_count} valid")
        
        # Create target directories
        train_dir = data_dir / 'train' / class_name
        valid_dir = data_dir / 'valid' / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        valid_dir.mkdir(parents=True, exist_ok=True)
        
        # Split files
        train_files = files[:target_train_count]
        valid_files = files[target_train_count:]
        
        # Move files to correct directories
        for file_path in train_files:
            target_path = train_dir / file_path.name
            if file_path != target_path:
                shutil.move(str(file_path), str(target_path))
                moves_made += 1
                
        for file_path in valid_files:
            target_path = valid_dir / file_path.name
            if file_path != target_path:
                shutil.move(str(file_path), str(target_path))
                moves_made += 1
    
    print(f"\n✅ Moved {moves_made} files to achieve target split")
    
    # Show final distribution
    final_counts = count_files_by_class(data_dir)
    
    print("\n📈 Final distribution:")
    total_train = 0
    total_valid = 0
    
    for class_name, counts in final_counts.items():
        total = counts['total']
        train_count = counts['train']
        valid_count = counts['valid']
        train_pct = (train_count / total * 100) if total > 0 else 0
        valid_pct = (valid_count / total * 100) if total > 0 else 0
        print(f"  {class_name}: {total} total ({train_count} train {train_pct:.1f}%, {valid_count} valid {valid_pct:.1f}%)")
        total_train += train_count
        total_valid += valid_count
    
    overall_train_pct = (total_train / (total_train + total_valid) * 100) if (total_train + total_valid) > 0 else 0
    overall_valid_pct = (total_valid / (total_train + total_valid) * 100) if (total_train + total_valid) > 0 else 0
    
    print(f"\n🎯 Overall split: {total_train + total_valid} total ({total_train} train {overall_train_pct:.1f}%, {total_valid} valid {overall_valid_pct:.1f}%)")
    
    return final_counts