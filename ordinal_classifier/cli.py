"""Command line interface for shot type classifier."""

import click
import pandas as pd
import torch
from pathlib import Path
from typing import Optional

from .core import ShotTypeClassifier
from .heatmaps import HeatmapGenerator
from .evaluation import ModelEvaluator
from .uncertainty import find_uncertain_images
from .dataset_utils import rebalance_dataset, rename_to_md5

# Import architectures for CLI
from fastai.vision.all import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import efficientnet_b0, efficientnet_b3, efficientnet_b5

# Architecture mapping for CLI
ARCHITECTURES = {
    'resnet18': resnet18,
    'resnet34': resnet34, 
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b3': efficientnet_b3,
    'efficientnet_b5': efficientnet_b5,
}

def select_device(device: str) -> str:
    """Select the appropriate device based on availability and user preference.
    
    Args:
        device: User specified device ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        Actual device to use
    """
    if device == 'auto':
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        click.echo("⚠️  CUDA requested but not available, falling back to CPU")
        return 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        click.echo("⚠️  MPS requested but not available, falling back to CPU")
        return 'cpu'
    else:
        return device

@click.group()
@click.version_option(version="2.0.0")
def main():
    """Ordinal Classifier
    
    Detect ordinal image classes using a pre-trained ResNet-50.
    """
    pass


@main.command()
@click.argument('data_path', type=click.Path(exists=True, path_type=Path))
@click.option('--epochs', '-e', default=5, help='Number of training epochs')
@click.option('--fine-tune-epochs', '-f', default=3, help='Number of fine-tuning epochs')
@click.option('--learning-rate', '-lr', default=5e-4, help='Learning rate')
@click.option('--batch-size', '-bs', default=16, help='Batch size')
@click.option('--image-size', '-s', default='375,666', help='Image size as height,width')
@click.option('--model-name', '-n', default='shot-type-classifier', help='Model name for saving')
@click.option('--no-save', is_flag=True, help='Do not save the trained model')
@click.option('--ordinal', is_flag=True, help='Use ordinal-aware label smoothing')
@click.option('--smoothing', default=0.02, help='Label smoothing factor for ordinal training')
@click.option('--arch', default='resnet50', type=click.Choice(list(ARCHITECTURES.keys())), 
              help='Model architecture')
@click.option('--scheduler', default='sgdr', type=click.Choice(['one_cycle', 'flat_cos', 'sgdr']),
              help='Learning rate scheduler')
@click.option('--early-stopping/--no-early-stopping', default=True, 
              help='Enable early stopping to prevent overfitting')
@click.option('--patience', default=3, help='Early stopping patience (epochs to wait)')
@click.option('--min-delta', default=0.001, help='Minimum improvement threshold for early stopping')
@click.option('--monitor', default='valid_acc', type=click.Choice(['valid_acc', 'valid_loss']),
              help='Metric to monitor for early stopping (default: valid_acc)')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cpu', 'cuda', 'mps']),
              help='Device to use for training (auto detects best available)')
def train(
    data_path: Path,
    epochs: int,
    fine_tune_epochs: int,
    learning_rate: float,
    batch_size: int,
    image_size: str,
    model_name: str,
    no_save: bool,
    ordinal: bool,
    smoothing: float,
    arch: str,
    scheduler: str,
    early_stopping: bool,
    patience: int,
    min_delta: float,
    monitor: str,
    device: str
):
    """Train a shot type classifier.
    
    DATA_PATH should contain 'train' and 'valid' subdirectories with images
    organized by shot type.
    
    Use --ordinal for ordinal-aware training that reduces adjacent classification errors.
    """
    # Parse image size
    try:
        height, width = map(int, image_size.split(','))
        img_size = (height, width)
    except ValueError:
        raise click.BadParameter("Image size must be in format 'height,width'")
    
    # Select device
    actual_device = select_device(device)
    click.echo(f"🔧 Using device: {actual_device}")
        
    click.echo(f"Training shot type classifier...")
    click.echo(f"Data path: {data_path}")
    click.echo(f"Architecture: {arch}")
    click.echo(f"Scheduler: {scheduler}")
    click.echo(f"Epochs: {epochs} + {fine_tune_epochs} fine-tuning")
    click.echo(f"Learning rate: {learning_rate}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Image size: {img_size}")
    
    if ordinal:
        click.echo(f"🎯 Using ordinal-aware training (smoothing: {smoothing})")
        if 'ordinal' not in model_name:
            model_name = f"ordinal-{model_name}"
    else:
        click.echo("📊 Using standard classification")
    
    if early_stopping:
        click.echo(f"🛑 Early stopping enabled (monitor: {monitor}, patience: {patience}, min_delta: {min_delta})")
    else:
        click.echo("⚠️  Early stopping disabled")
    
    # Get architecture function
    arch_func = ARCHITECTURES[arch]
    
    # Initialize classifier
    classifier = ShotTypeClassifier()
    
    # Train model
    try:
        if ordinal:
            classifier.train_with_ordinal_smoothing(
                data_path=data_path,
                epochs=epochs,
                learning_rate=learning_rate,
                fine_tune_epochs=fine_tune_epochs,
                image_size=img_size,
                batch_size=batch_size,
                save_model=not no_save,
                model_name=model_name,
                smoothing=smoothing,
                arch=arch_func,
                scheduler=scheduler,
                early_stopping=early_stopping,
                patience=patience,
                min_delta=min_delta,
                monitor=monitor
            )
        else:
            classifier.train(
                data_path=data_path,
                epochs=epochs,
                learning_rate=learning_rate,
                fine_tune_epochs=fine_tune_epochs,
                image_size=img_size,
                batch_size=batch_size,
                save_model=not no_save,
                model_name=model_name,
                arch=arch_func,
                early_stopping=early_stopping,
                patience=patience,
                min_delta=min_delta,
                monitor=monitor
            )
        
        # Move model to selected device after training
        if actual_device != 'cpu':
            classifier.move_to_device(actual_device)
            
        click.echo(f"✅ Training completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Training failed: {e}")
        raise click.ClickException(str(e))


@main.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--model-path', '-m', type=click.Path(exists=True, path_type=Path), 
              help='Path to model file')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='Output directory for predictions')
@click.option('--save-predictions/--no-save', default=False,
              help='Save prediction CSV files (requires --output-dir)')
@click.option('--show-probabilities', is_flag=True,
              help='Show prediction probabilities')
@click.option('--recursive', '-r', is_flag=True,
              help='Search subdirectories recursively for images')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cpu', 'cuda', 'mps']),
              help='Device to use for inference (auto detects best available)')
def predict(
    input_path: Path,
    model_path: Optional[Path],
    output_dir: Optional[Path],
    save_predictions: bool,
    show_probabilities: bool,
    recursive: bool,
    device: str
):
    """Predict shot types for images.
    
    INPUT_PATH can be either a single image file or a directory containing images.
    Use --recursive to search subdirectories for images.
    """
    # Select device
    actual_device = select_device(device)
    click.echo(f"🔧 Using device: {actual_device}")
    
    # Initialize classifier
    classifier = ShotTypeClassifier(model_path)
    
    try:
        # Load model
        classifier.load_model()
        
        # Move model to selected device
        if actual_device != 'cpu':
            classifier.move_to_device(actual_device)
            
        click.echo(f"✅ Model loaded successfully on {actual_device}")
        
    except Exception as e:
        click.echo(f"❌ Failed to load model: {e}")
        raise click.ClickException(str(e))
    
    # Validate save_predictions requires output_dir
    if save_predictions and not output_dir:
        click.echo("❌ Error: --save-predictions requires --output-dir to be specified")
        click.echo("   This prevents polluting your dataset directory with prediction files")
        raise click.ClickException("Must specify --output-dir when using --save-predictions")
    
    # Handle single image vs directory
    if input_path.is_file():
        # Single image prediction
        try:
            if show_probabilities:
                pred_class, probs = classifier.predict_single(
                    input_path, return_probs=True
                )
                click.echo(f"Image: {input_path.name}")
                click.echo(f"Predicted class: {pred_class}")
                click.echo("Probabilities:")
                
                # Get the actual model vocabulary
                if hasattr(classifier.learn.dls, 'vocab'):
                    vocab = list(classifier.learn.dls.vocab)
                else:
                    # Get from class info method
                    class_info = classifier.get_class_info()
                    vocab = class_info.get('classes', [f"class_{i}" for i in range(len(probs))])
                
                # Probabilities are already normalized from predict_single
                for i, shot_type in enumerate(vocab):
                    if i < len(probs):
                        prob = float(probs[i]) * 100
                        click.echo(f"  {shot_type}: {prob:.2f}%")
            else:
                pred_class = classifier.predict_single(input_path)
                click.echo(f"Image: {input_path.name}")
                click.echo(f"Predicted class: {pred_class}")
                
        except Exception as e:
            click.echo(f"❌ Prediction failed: {e}")
            raise click.ClickException(str(e))
            
    else:
        # Directory prediction
        try:
            click.echo(f"Predicting shot types for images in: {input_path}")
            if recursive:
                click.echo("🔍 Searching subdirectories recursively...")
            
            results = classifier.predict_batch(
                image_dir=input_path,
                save_predictions=save_predictions,
                output_dir=output_dir,
                recursive=recursive
            )
            
            click.echo(f"✅ Processed {len(results)} images")
            
            # Show summary
            if len(results) > 0:
                click.echo("\nSummary:")
                for _, row in results.head(10).iterrows():
                    path_to_show = row.get('relative_path', row['filename'])
                    click.echo(f"  {path_to_show}: {row['predicted_class']} "
                             f"({row['confidence']:.1f}%)")
                             
                if len(results) > 10:
                    click.echo(f"  ... and {len(results) - 10} more images")
                    
            if save_predictions and output_dir:
                click.echo(f"📁 Predictions saved to: {output_dir}")
                
        except Exception as e:
            click.echo(f"❌ Batch prediction failed: {e}")
            raise click.ClickException(str(e))


@main.command()
@click.argument('data_path', type=click.Path(exists=True, path_type=Path))
@click.option('--model-path', '-m', type=click.Path(exists=True, path_type=Path),
              help='Path to model file')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='Output directory for evaluation results')
@click.option('--recursive', '-r', is_flag=True,
              help='Search subdirectories recursively for images')
@click.option('--no-save', is_flag=True,
              help='Do not save evaluation results to files')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cpu', 'cuda', 'mps']),
              help='Device to use for evaluation (auto detects best available)')
def evaluate(
    data_path: Path,
    model_path: Optional[Path],
    output_dir: Optional[Path],
    recursive: bool,
    no_save: bool,
    device: str
):
    """Evaluate model performance on labeled data.
    
    DATA_PATH should contain subdirectories named by shot types, each containing
    images of that shot type. Generates confusion matrix and performance metrics.
    """
    # Select device
    actual_device = select_device(device)
    click.echo(f"🔧 Using device: {actual_device}")
    
    # Initialize classifier and load model
    classifier = ShotTypeClassifier(model_path)
    
    try:
        classifier.load_model()
        
        # Move model to selected device
        if actual_device != 'cpu':
            classifier.move_to_device(actual_device)
            
        click.echo(f"✅ Model loaded successfully on {actual_device}")
        
    except Exception as e:
        click.echo(f"❌ Failed to load model: {e}")
        raise click.ClickException(str(e))
    
    # Validate that saving results requires output_dir
    if not no_save and not output_dir:
        click.echo("❌ Error: Saving evaluation results requires --output-dir to be specified")
        click.echo("   This prevents polluting your workspace with result files")
        click.echo("   Use --no-save to skip saving files, or specify --output-dir")
        raise click.ClickException("Must specify --output-dir when saving evaluation results")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(classifier)
    
    try:
        click.echo(f"Evaluating model on data in: {data_path}")
        if recursive:
            click.echo("🔍 Searching subdirectories recursively...")
            
        # Run evaluation
        results = evaluator.evaluate_directory(
            data_dir=data_path,
            recursive=recursive,
            save_results=not no_save,
            output_dir=output_dir
        )
        
        # Print summary to console
        evaluator.print_summary(results)
        
        if not no_save:
            output_path = output_dir or Path("evaluation_results")
            click.echo(f"\n📊 Detailed results and plots saved to: {output_path}")
            
    except Exception as e:
        click.echo(f"❌ Evaluation failed: {e}")
        raise click.ClickException(str(e))


@main.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--model-path', '-m', type=click.Path(exists=True, path_type=Path),
              help='Path to model file')
@click.option('--alpha', '-a', default=0.5, type=float,
              help='Heatmap blending alpha (0.0-1.0)')
@click.option('--only-heatmap', is_flag=True,
              help='Save only heatmap without original image overlay')
@click.option('--save-original/--no-save-original', default=True,
              help='Save original images alongside heatmaps')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cpu', 'cuda', 'mps']),
              help='Device to use for heatmap generation (auto detects best available)')
def heatmap(
    input_path: Path,
    output_dir: Path,
    model_path: Optional[Path],
    alpha: float,
    only_heatmap: bool,
    save_original: bool,
    device: str
):
    """Generate activation heatmaps for images.
    
    INPUT_PATH can be either a single image file or a directory containing images.
    OUTPUT_DIR is where heatmaps will be saved.
    """
    # Validate alpha
    if not 0.0 <= alpha <= 1.0:
        raise click.BadParameter("Alpha must be between 0.0 and 1.0")
    
    # Select device
    actual_device = select_device(device)
    click.echo(f"🔧 Using device: {actual_device}")
        
    # Initialize classifier and load model
    classifier = ShotTypeClassifier(model_path)
    
    try:
        classifier.load_model()
        
        # Move model to selected device
        if actual_device != 'cpu':
            classifier.move_to_device(actual_device)
            
        click.echo(f"✅ Model loaded successfully on {actual_device}")
        
    except Exception as e:
        click.echo(f"❌ Failed to load model: {e}")
        raise click.ClickException(str(e))
    
    # Initialize heatmap generator
    heatmap_gen = HeatmapGenerator(classifier)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle single image vs directory
    if input_path.is_file():
        # Single image heatmap
        try:
            output_path = output_dir / f"{input_path.stem}_heatmap.png"
            
            click.echo(f"Generating heatmap for: {input_path.name}")
            
            pred_class = heatmap_gen.generate_single_heatmap(
                image_path=input_path,
                output_path=output_path,
                alpha=alpha
            )
            
            click.echo(f"✅ Heatmap saved to: {output_path}")
            click.echo(f"Predicted class: {pred_class}")
            
        except Exception as e:
            click.echo(f"❌ Heatmap generation failed: {e}")
            raise click.ClickException(str(e))
            
    else:
        # Directory heatmap generation
        try:
            click.echo(f"Generating heatmaps for images in: {input_path}")
            click.echo(f"Output directory: {output_dir}")
            click.echo(f"Alpha blending: {alpha}")
            
            heatmap_gen.generate_heatmaps(
                image_dir=input_path,
                output_dir=output_dir,
                alpha=alpha,
                only_heatmap=only_heatmap,
                save_original=save_original
            )
            
            click.echo(f"✅ Heatmap generation completed!")
            
        except Exception as e:
            click.echo(f"❌ Heatmap generation failed: {e}")
            raise click.ClickException(str(e))


@main.command()
@click.option('--model-path', '-m', type=click.Path(exists=True, path_type=Path),
              help='Path to model file')
def info(model_path: Optional[Path]):
    """Show information about the classifier and model."""
    click.echo("Ordinal Classifier v2.0.0")
    click.echo("=" * 40)
    
    # Try to load model and show info
    if model_path or Path("models/shot-type-classifier.pkl").exists():
        try:
            classifier = ShotTypeClassifier(model_path)
            classifier.load_model()
            
            # Get class information
            class_info = classifier.get_class_info()
            
            click.echo(f"\nModel information:")
            click.echo(f"  Model file: {model_path or 'models/shot-type-classifier.pkl'}")
            click.echo(f"  Input size: 375 x 666 pixels")
            click.echo(f"  Classes: {class_info['num_classes']}")
            
            if class_info['classes']:
                click.echo(f"\nModel classes:")
                for i, class_name in enumerate(class_info['classes'], 1):
                    ordinal_pos = class_info['ordinal_mapping'].get(class_name, '?')
                    click.echo(f"  {i}. {class_name} (ordinal position: {ordinal_pos})")
                
                if class_info['is_valid_ordinal_sequence']:
                    click.echo(f"  ✅ Valid ordinal sequence: {class_info['ordinal_range'][0]} to {class_info['ordinal_range'][1]}")
                else:
                    click.echo(f"  ⚠️  Warning: Classes may not form valid ordinal sequence")
                    click.echo(f"     Expected: 0, 1, 2, ... prefixes for optimal ordinal training")
                
        except Exception as e:
            click.echo(f"\n⚠️  Could not load model: {e}")
            
    else:
        click.echo("\nNo model found. Train a model first or specify --model-path")
        click.echo("\nThis classifier works with any ordinal classification task where")
        click.echo("folder names have numeric prefixes (e.g., '0-category', '1-another').")
            
    click.echo("\nUsage examples:")
    click.echo("  ordinal-classifier train ./data --ordinal")
    click.echo("  ordinal-classifier predict ./images --recursive")
    click.echo("  ordinal-classifier evaluate ./test --recursive")
    click.echo("  ordinal-classifier heatmap ./images ./heatmaps")


@main.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--epochs', '-e', default=5, help='Number of training epochs')
@click.option('--fine-tune-epochs', '-f', default=3, help='Number of fine-tuning epochs')
@click.option('--learning-rate', '-lr', default=1e-3, help='Learning rate')
@click.option('--batch-size', '-bs', default=16, help='Batch size')
@click.option('--image-size', '-s', default='375,666', help='Image size as height,width')
@click.option('--model-name', '-n', default='ordinal-shot-classifier', help='Model name for saving')
@click.option('--no-save', is_flag=True, help="Don't save the trained model")
@click.option('--loss-type', default='ordinal', type=click.Choice(['ordinal', 'label_smoothing', 'standard']), 
              help='Loss function type')
@click.option('--lambda-ord', default=1.0, help='Weight for ordinal penalty (ordinal loss only)')
@click.option('--smoothing', default=0.1, help='Label smoothing factor (label_smoothing loss only)')
def train_ordinal(data_path, epochs, fine_tune_epochs, learning_rate, batch_size, 
                  image_size, model_name, no_save, loss_type, lambda_ord, smoothing):
    """Train an ordinal regression model for shot type classification."""
    from .ordinal import OrdinalShotTypeClassifier
    
    # Parse image size
    height, width = map(int, image_size.split(','))
    
    try:
        classifier = OrdinalShotTypeClassifier(loss_type=loss_type)
        
        click.echo(f"Training ordinal shot type classifier...")
        click.echo(f"Data path: {data_path}")
        click.echo(f"Epochs: {epochs} (+ {fine_tune_epochs} fine-tuning)")
        click.echo(f"Learning rate: {learning_rate}")
        click.echo(f"Batch size: {batch_size}")
        click.echo(f"Image size: {height}x{width}")
        click.echo(f"Loss type: {loss_type}")
        
        if loss_type == 'ordinal':
            click.echo(f"Lambda ordinal: {lambda_ord}")
        elif loss_type == 'label_smoothing':
            click.echo(f"Smoothing factor: {smoothing}")
        
        classifier.train_ordinal(
            data_path=data_path,
            epochs=epochs,
            learning_rate=learning_rate,
            fine_tune_epochs=fine_tune_epochs,
            image_size=(height, width),
            batch_size=batch_size,
            save_model=not no_save,
            model_name=model_name,
            loss_type=loss_type,
            lambda_ord=lambda_ord
        )
        
        click.echo(f"✅ Training completed!")
        if not no_save:
            click.echo(f"Model saved as: models/{model_name}.pkl")
            
    except Exception as e:
        click.echo(f"❌ Training failed: {e}")
        raise click.Abort()


@main.command()
@click.argument('data_path1', type=click.Path(exists=True))
@click.argument('data_path2', type=click.Path(exists=True), required=False)
@click.option('--epochs', '-e', default=3, help='Number of training epochs for comparison')
@click.option('--batch-size', '-bs', default=16, help='Batch size')
@click.option('--output-dir', '-o', help='Directory to save comparison results')
def compare_approaches(data_path1, data_path2, epochs, batch_size, output_dir):
    """Compare standard vs ordinal classification approaches."""
    from .ordinal import compare_models
    from .evaluation import ModelEvaluator
    
    # Use data_path1 for training, data_path2 for testing (if provided)
    test_path = data_path2 if data_path2 else data_path1
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = Path("comparison_results")
        output_dir.mkdir(exist_ok=True)
    
    click.echo("🔄 Comparing Standard vs Ordinal Classification Approaches")
    click.echo(f"Training data: {data_path1}")
    click.echo(f"Test data: {test_path}")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Results will be saved to: {output_dir}")
    
    try:
        # Train all models
        models = compare_models(data_path1, test_path, epochs)
        
        # Evaluate all models
        click.echo("\n📊 Evaluating models...")
        results = {}
        
        for model_name, classifier in models.items():
            click.echo(f"\nEvaluating {model_name}...")
            
            evaluator = ModelEvaluator(classifier)
            eval_results = evaluator.evaluate_directory(
                test_path, 
                recursive=True,
                output_dir=output_dir / f"{model_name}_evaluation"
            )
            
            results[model_name] = eval_results
            evaluator.print_summary(eval_results)
        
        # Create comparison report
        comparison_data = []
        for model_name, eval_results in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': eval_results['accuracy'],
                'Precision': eval_results['precision'],
                'Recall': eval_results['recall'],
                'F1-Score': eval_results['f1_score'],
                'Num_Samples': eval_results['num_samples']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
        
        click.echo(f"\n📈 COMPARISON SUMMARY")
        click.echo("=" * 50)
        print(comparison_df.to_string(index=False))
        
        click.echo(f"\n✅ Comparison completed! Results saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Comparison failed: {e}")
        raise click.Abort()


@main.command()
@click.argument('image_dir', type=click.Path(exists=True, path_type=Path))
@click.option('--model-path', '-m', type=click.Path(exists=True, path_type=Path),
              help='Path to model file')
@click.option('--model-type', default='auto', type=click.Choice(['auto', 'ordinal', 'classification']),
              help='Model type (auto-detected by default)')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cpu', 'cuda', 'mps']),
              help='Device to use (auto detects best available)')
def find_uncertain(image_dir: Path, model_path: Optional[Path], model_type: str, device: str):
    """Find the most uncertain images in a directory.
    
    Analyzes images and renames them with uncertainty scores in the format:
    uncertainty_{score}_pred-{class}_conf-{confidence}_{md5}.ext
    
    High uncertainty for classification models means high entropy.
    High uncertainty for ordinal models means the prediction is between ordinal classes.
    """
    # Select device
    actual_device = select_device(device)
    click.echo(f"🔧 Using device: {actual_device}")

    # Load model
    classifier = ShotTypeClassifier(model_path)
    
    try:
        classifier.load_model()
        
        # Move model to selected device
        if actual_device != 'cpu':
            classifier.move_to_device(actual_device)
            
        click.echo(f"✅ Model loaded successfully on {actual_device}")
        
    except Exception as e:
        click.echo(f"❌ Failed to load model: {e}")
        raise click.ClickException(str(e))
    
    # Auto-detect model type if not specified
    if model_type == 'auto':
        model_file = model_path or Path("models/shot-type-classifier.pkl")
        detected_type = 'ordinal' if 'ordinal' in model_file.name else 'classification'
        click.echo(f"🔍 Auto-detected model type: {detected_type}")
        model_type = detected_type
    
    try:
        find_uncertain_images(image_dir, classifier, model_type)
        click.echo(f"✅ Uncertainty analysis completed!")
        
    except Exception as e:
        click.echo(f"❌ Uncertainty analysis failed: {e}")
        raise click.ClickException(str(e))


@main.command()
@click.argument('data_dir', type=click.Path(exists=True, path_type=Path))
@click.option('--valid-ratio', default=0.2, type=float,
              help='Target validation ratio (default: 0.2)')
@click.option('--seed', default=42, type=int,
              help='Random seed for reproducible splits (default: 42)')
@click.option('--skip-rename', is_flag=True,
              help='Skip MD5 renaming step')
def rebalance(data_dir: Path, valid_ratio: float, seed: int, skip_rename: bool):
    """Rebalance dataset with MD5 renaming and deduplication.
    
    DATA_DIR should contain 'train' and 'valid' subdirectories with images
    organized by class.
    
    This command:
    1. Renames all image files to their MD5 hash (unless --skip-rename)
    2. Removes duplicate files based on content
    3. Rebalances the dataset to achieve the target train/validation split
    4. Maintains class distribution across train/valid splits
    """
    if not (0 < valid_ratio < 1):
        raise click.BadParameter("Validation ratio must be between 0 and 1")
    
    click.echo(f"🚀 Rebalancing dataset: {data_dir}")
    click.echo(f"Target validation ratio: {valid_ratio}")
    click.echo(f"Random seed: {seed}")
    
    try:
        # Step 1: Rename files to MD5 hashes
        if not skip_rename:
            rename_to_md5(data_dir)
        else:
            click.echo("⏭️  Skipping MD5 renaming")
        
        # Step 2: Rebalance the dataset
        rebalance_dataset(data_dir, valid_ratio, seed)
        
        click.echo("\n🎉 Dataset rebalancing completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Error during rebalancing: {e}")
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main() 