"""Evaluation and reporting functionality for shot type classifier."""

import os
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from .core import ShotTypeClassifier


class ModelEvaluator:
    """Evaluate model performance and generate reports."""
    
    def __init__(self, classifier: ShotTypeClassifier):
        """Initialize with a trained classifier.
        
        Args:
            classifier: Trained ShotTypeClassifier instance
        """
        self.classifier = classifier
        self.learn = classifier.learn
        
        if self.learn is None:
            raise ValueError("Classifier must have a loaded model")
    
    def evaluate_directory(
        self,
        data_dir: Union[str, Path],
        recursive: bool = True,
        save_results: bool = True,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict:
        """Evaluate model on a directory of labeled images.
        
        Args:
            data_dir: Directory with subdirectories named by shot types
            recursive: Whether to search subdirectories recursively
            save_results: Whether to save evaluation results
            output_dir: Directory to save results (required if save_results=True)
            
        Returns:
            Dictionary with evaluation metrics
        """
        data_dir = Path(data_dir)
        if save_results and output_dir is None:
            raise ValueError("Must specify output_dir when save_results=True to avoid polluting workspace")
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Get model's actual vocabulary
        model_classes = list(self.learn.dls.vocab) if hasattr(self.learn.dls, 'vocab') else []
        
        # Collect all images with their true labels
        true_labels = []
        pred_labels = []
        image_paths = []
        confidences = []
        
        # Get all shot type directories - exclude common non-class directories
        excluded_dirs = {"evaluation_results", "predictions", "heatmaps", "models", ".git", "__pycache__"}
        shot_type_dirs = [d for d in data_dir.iterdir() 
                         if d.is_dir() and d.name not in excluded_dirs]
        
        print(f"Evaluating on {len(shot_type_dirs)} shot type categories...")
        print(f"Model classes: {model_classes}")
        
        for shot_dir in shot_type_dirs:
            # Use directory name directly if it matches model classes
            if shot_dir.name in model_classes:
                true_label = shot_dir.name
            else:
                print(f"Warning: Skipping directory '{shot_dir.name}' - not in model classes")
                continue
                
            # Get image files
            if recursive:
                image_files = list(shot_dir.rglob("*.jpg")) + \
                             list(shot_dir.rglob("*.jpeg")) + \
                             list(shot_dir.rglob("*.png"))
            else:
                image_files = [f for f in shot_dir.iterdir() 
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            print(f"Processing {len(image_files)} images from {shot_dir.name}...")
            
            for img_file in image_files:
                try:
                    # Get prediction
                    pred_class, probs = self.classifier.predict_single(
                        img_file, return_probs=True
                    )
                    
                    true_labels.append(true_label)
                    pred_labels.append(pred_class)
                    image_paths.append(str(img_file.relative_to(data_dir)))
                    confidences.append(float(probs.max()))
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    continue
        
        if len(true_labels) == 0:
            raise ValueError("No valid images found for evaluation")
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted'
        )
        
        # Create detailed results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_samples': len(true_labels),
            'true_labels': true_labels,
            'pred_labels': pred_labels,
            'image_paths': image_paths,
            'confidences': confidences,
            'model_classes': model_classes
        }
        
        # Generate confusion matrix using actual model classes
        cm = confusion_matrix(true_labels, pred_labels, labels=model_classes)
        results['confusion_matrix'] = cm
        
        # Generate classification report using actual model classes
        class_report = classification_report(
            true_labels, pred_labels,
            labels=model_classes,
            output_dict=True
        )
        results['classification_report'] = class_report
        
        if save_results:
            self._save_evaluation_results(results, output_dir)
        
        return results
    
    def _map_dir_name_to_shot_type(self, dir_name: str) -> Optional[str]:
        """Map directory name to model's expected class name.
        
        Args:
            dir_name: Directory name
            
        Returns:
            Model class name or None if not recognized
        """
        # Get model vocabulary
        if hasattr(self.learn, 'dls') and self.learn.dls and hasattr(self.learn.dls, 'vocab'):
            vocab = list(self.learn.dls.vocab)
            # Direct match with model vocabulary
            if dir_name in vocab:
                return dir_name
        
        return None
    
    def _save_evaluation_results(self, results: Dict, output_dir: Path) -> None:
        """Save evaluation results to files.
        
        Args:
            results: Evaluation results dictionary
            output_dir: Output directory
        """
        # Save summary metrics
        summary = {
            'accuracy': results['accuracy'],
            'precision': results['precision'], 
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'num_samples': results['num_samples']
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(output_dir / "evaluation_summary.csv", index=False)
        
        # Save detailed results
        detailed_df = pd.DataFrame({
            'image_path': results['image_paths'],
            'true_label': results['true_labels'],
            'predicted_label': results['pred_labels'],
            'confidence': results['confidences'],
            'correct': [t == p for t, p in zip(results['true_labels'], results['pred_labels'])]
        })
        detailed_df.to_csv(output_dir / "detailed_results.csv", index=False)
        
        # Save classification report
        class_report_df = pd.DataFrame(results['classification_report']).transpose()
        class_report_df.to_csv(output_dir / "classification_report.csv")
        
        # Generate and save confusion matrix plot
        self._plot_confusion_matrix(
            results['confusion_matrix'], 
            output_dir / "confusion_matrix.png",
            results['model_classes']
        )
        
        # Generate and save per-class accuracy plot
        self._plot_per_class_metrics(
            results['classification_report'],
            output_dir / "per_class_metrics.png",
            results['model_classes']
        )
        
        print(f"Evaluation results saved to: {output_dir}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray, output_path: Path, model_classes: List[str]) -> None:
        """Plot and save confusion matrix.
        
        Args:
            cm: Confusion matrix
            output_path: Output file path
            model_classes: List of model class names
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=model_classes,
            yticklabels=model_classes,
            cbar_kws={'label': 'Percentage (%)'}
        )
        
        plt.title('Confusion Matrix (%)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Shot Type', fontsize=12)
        plt.ylabel('True Shot Type', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save raw counts
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=model_classes,
            yticklabels=model_classes,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Shot Type', fontsize=12)
        plt.ylabel('True Shot Type', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path.with_suffix('.counts.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_metrics(self, class_report: Dict, output_path: Path, model_classes: List[str]) -> None:
        """Plot per-class metrics.
        
        Args:
            class_report: Classification report dictionary
            output_path: Output file path
            model_classes: List of model class names
        """
        # Extract per-class metrics
        metrics = ['precision', 'recall', 'f1-score']
        
        data = []
        for class_name in model_classes:
            if class_name in class_report:
                for metric in metrics:
                    data.append({
                        'Shot Type': class_name,
                        'Metric': metric.title(),
                        'Value': class_report[class_name][metric]
                    })
        
        if not data:
            print("Warning: No metrics data available for plotting")
            return
            
        df = pd.DataFrame(data)
        
        # Create grouped bar plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Shot Type', y='Value', hue='Metric')
        plt.title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Shot Type', fontsize=12)
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metric')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self, results: Dict) -> None:
        """Print evaluation summary to console.
        
        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION SUMMARY")
        print("="*50)
        print(f"Total samples: {results['num_samples']}")
        print(f"Overall accuracy: {results['accuracy']:.3f}")
        print(f"Weighted precision: {results['precision']:.3f}")
        print(f"Weighted recall: {results['recall']:.3f}")
        print(f"Weighted F1-score: {results['f1_score']:.3f}")
        
        model_classes = results.get('model_classes', [])
        
        print("\nPer-class metrics:")
        print("-" * 50)
        for class_name in model_classes:
            if class_name in results['classification_report']:
                metrics = results['classification_report'][class_name]
                print(f"{class_name:20s} | P: {metrics['precision']:.3f} | "
                      f"R: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f} | "
                      f"Support: {int(metrics['support'])}")
        
        print("\nConfusion Matrix (% of true class):")
        print("-" * 50)
        cm = results['confusion_matrix']
        
        # Handle the case where there might be zero divisions
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            cm_percent = np.nan_to_num(cm_percent)  # Replace NaN with 0
        
        # Print header
        header = "True\\Pred".ljust(20)
        for class_name in model_classes:
            header += f" {class_name[:8]:>9s}"
        print(header)
        
        # Print rows
        for i, true_class in enumerate(model_classes):
            row = f"{true_class[:19]:20s}"
            for j in range(len(model_classes)):
                row += f" {cm_percent[i, j]:8.1f}%"
            print(row) 