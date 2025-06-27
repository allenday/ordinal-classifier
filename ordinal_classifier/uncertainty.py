"""Uncertainty analysis tools for ordinal classifier."""

import hashlib
import torch
from pathlib import Path
from fastai.vision.all import PILImage


def calculate_md5(file_path: Path) -> str:
    """Calculates the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def calculate_entropy(probs):
    """Calculates the entropy of a probability distribution."""
    return -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1)


def calculate_ordinal_uncertainty(probs, classifier):
    """
    Calculates uncertainty for ordinal classification using the model's ordinal mapping.
    For ordinal data, uncertainty is high when probability mass is spread across
    non-adjacent classes or when the prediction is between two classes.
    """
    # Get class info from the classifier to ensure correct ordinal mapping
    class_info = classifier.get_class_info()
    vocab = class_info.get('classes')
    ordinal_mapping = class_info.get('ordinal_mapping')

    if not vocab or not ordinal_mapping:
        # Fallback to simple positional uncertainty if mapping is missing
        print("⚠️  Ordinal mapping not found, falling back to simple positional uncertainty.")
        ordinal_positions = torch.arange(len(probs), dtype=torch.float32, device=probs.device)
    else:
        # Create a tensor of ordinal positions corresponding to the vocab order
        # The 'probs' tensor is ordered according to dls.vocab
        positions = [ordinal_mapping.get(class_name, idx) for idx, class_name in enumerate(vocab)]
        ordinal_positions = torch.tensor(positions, dtype=torch.float32, device=probs.device)
    
    # Calculate weighted average position (expected ordinal value)
    expected_position = (probs * ordinal_positions).sum().item()
    
    # Uncertainty is the distance from the nearest integer position
    # This captures how "between classes" the prediction is
    uncertainty = abs(expected_position - round(expected_position))
    
    return uncertainty


def find_uncertain_images(path_img, classifier, model_type):
    """
    Analyzes images, calculates their prediction uncertainty, and renames
    the files to a canonical format: uncertainty_{score}_pred-{class}_{md5}.ext
    This process is idempotent based on image content and will re-process
    all image files in the directory to update scores or fix old filenames.
    """
    path_img = Path(path_img)
    
    print(f"Model type: {model_type}")
    print(f"Model device: {classifier.get_device()}")
    
    # Get all image files
    all_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        all_files.extend(list(path_img.glob(f"*{ext}")))
    
    if not all_files:
        print(f"No images found in {path_img}.")
        return

    # Pre-filter for valid/readable images
    print(f"Verifying {len(all_files)} image files...")
    verified_files = []
    for f in all_files:
        try:
            # Quick check to see if the image can be opened
            _ = PILImage.create(f)
            verified_files.append(f)
        except Exception:
            print(f"Skipping corrupt or unreadable file: {f.name}")

    if not verified_files:
        print("No valid, readable images were found to process.")
        return

    print(f"Processing {len(verified_files)} valid images...")

    # Get predictions and calculate uncertainty
    results = []
    
    for i, img_file in enumerate(verified_files):
        if i > 0 and i % 10 == 0:
            print(f"Processed {i}/{len(verified_files)} images...")
        
        try:
            # Use the fixed predict_single method from the core library
            pred_class, probs = classifier.predict_single(img_file, return_probs=True)
            
            # Calculate uncertainty based on model type
            if model_type == 'ordinal':
                uncertainty = calculate_ordinal_uncertainty(probs, classifier)
            else:
                uncertainty = calculate_entropy(probs).item()

            # Calculate MD5 hash for canonical naming
            md5_hash = calculate_md5(img_file)
            
            results.append({
                'original_file': img_file, 
                'md5': md5_hash,
                'uncertainty': uncertainty,
                'predicted_class': pred_class,
                'max_prob': probs.max().item()
            })
            
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
            continue

    if not results:
        print("No images were successfully processed.")
        return

    # Sort by uncertainty (highest first)
    results.sort(key=lambda x: x['uncertainty'], reverse=True)

    # Rename files with uncertainty scores using MD5 hash
    print("\nRenaming files with canonical format (uncertainty_..._md5.ext)...")
    renamed_count = 0
    deleted_count = 0
    for res in results:
        original_file = res['original_file']
        md5 = res['md5']
        uncertainty = res['uncertainty']
        pred_class = res['predicted_class']
        max_prob = res['max_prob']
        
        # Format the uncertainty score into the filename
        # e.g., "uncertainty_0.4500_pred-3-medium_conf-0.75_d41d8cd98f00.jpg"
        new_name = (
            f"uncertainty_{uncertainty:.4f}_"
            f"pred-{pred_class}_"
            f"conf-{max_prob:.2f}_"
            f"{md5}{original_file.suffix}"
        )
        new_path = path_img / new_name

        # If file already has the correct name and score, do nothing
        if new_path == original_file:
            continue
        
        try:
            if new_path.exists():
                # A file with this canonical name already exists.
                # This means the current file is a content duplicate.
                print(f"Duplicate content: {new_path.name} exists. Deleting source {original_file.name}.")
                original_file.unlink()
                deleted_count += 1
            else:
                # Rename the file to its new canonical name
                original_file.rename(new_path)
                renamed_count += 1
        except OSError as e:
            print(f"Error handling file {original_file.name}: {e}")

    print(f"\nFinished. Renamed {renamed_count} files and removed {deleted_count} duplicates in {path_img}.")
    if results:
        print(f"Uncertainty range: {results[-1]['uncertainty']:.4f} to {results[0]['uncertainty']:.4f}")
    
    # Print top 5 most uncertain images
    print("\nTop 5 most uncertain predictions:")
    for i, res in enumerate(results[:5]):
        new_name = (
            f"uncertainty_{res['uncertainty']:.4f}_"
            f"pred-{res['predicted_class']}_"
            f"conf-{res['max_prob']:.2f}_"
            f"{res['md5']}{res['original_file'].suffix}"
        )
        print(f"{i+1}. {new_name} (from: {res['original_file'].name})")
        print(f"   Uncertainty: {res['uncertainty']:.4f}")
        print(f"   Predicted: {res['predicted_class']}")
        print(f"   Confidence: {res['max_prob']:.2f}")