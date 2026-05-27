"""
Parallel checkpoint analysis for ViscoCANN models.
This module provides functions to process multiple checkpoints in parallel
and generate plots for each checkpoint.
"""

import os
import re
import glob
import tensorflow as tf
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import Plots
import copy

# Set multiprocessing start method to 'spawn' for better TensorFlow compatibility
# This is especially important on Windows
if mp.get_start_method(allow_none=True) != 'spawn':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, which is fine
        pass


def extract_epoch_from_checkpoint(checkpoint_path):
    """
    Extract epoch number from checkpoint filename.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        int: Epoch number extracted from filename
    """
    filename = os.path.basename(checkpoint_path)
    match = re.search(r'ckpt-epoch-(\d+)\.ckpt', filename)
    if match:
        return int(match.group(1))
    return 0


def get_checkpoint_paths(checkpoint_dir):
    """
    Get all checkpoint paths from directory, sorted by epoch number.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoint files
        
    Returns:
        list: Sorted list of checkpoint paths (without extensions)
    """
    # Find all .index files (they indicate complete checkpoints)
    index_files = glob.glob(os.path.join(checkpoint_dir, "ckpt-epoch-*.ckpt.index"))
    
    # Remove .index extension to get base checkpoint paths
    checkpoint_paths = [path[:-6] for path in index_files]  # Remove ".index"
    
    # Sort by epoch number
    checkpoint_paths.sort(key=extract_epoch_from_checkpoint)
    
    return checkpoint_paths


def build_custom_objects():
    """Construct custom_objects dict inside each worker to avoid pickling modules."""
    import ContinuumMechanics as CM
    import layers
    import subANNs
    import utils

    return {
        'CM': CM,
        'dirModel': CM.dirModelOrtho,
        'weightModel': CM.weightModelOrtho,
        'PositiveHeNormal': subANNs.PositiveHeNormal,
        'PositiveGlorotNormal': subANNs.PositiveGlorotNormal,
        'PositiveVarianceScaling': subANNs.PositiveVarianceScaling,
        'PsiSigmaLayer': CM.PsiSigmaLayer,
        'GradientLayer': CM.GradientLayer,
        'ScaleLayer': layers.ScaleLayer,
        'stressUpdateLayer': CM.stressUpdateLayer,
        'SparsityRegularizer': utils.SparsityRegularizer
    }


def process_single_checkpoint_worker(args):
    """
    Worker function for multiprocessing that processes a single checkpoint.
    
    Args:
        args (tuple): Contains (checkpoint_path, model_config, custom_objects, pathToData, outputFolder, rateDependent)
                     Note: custom_objects is passed but not used - we reconstruct them in the worker
        
    Returns:
        tuple: (epoch_number, success_status, error_message, output_folder)
    """
    checkpoint_path, model_config, pathToData, outputFolder, rateDependent = args
    
    try:
        custom_objects = build_custom_objects()
        # Extract epoch number for naming
        epoch = extract_epoch_from_checkpoint(checkpoint_path)
        
        # Create a new model instance using from_config
        model_copy = tf.keras.models.Model.from_config(
            model_config, 
            custom_objects=custom_objects
        )
        # model_copy = tf.keras.models.model_from_json(
        #     model_config,
        #     custom_objects=custom_objects
        # )
        
        # Load checkpoint weights
        model_copy.load_weights(checkpoint_path)
        
        # Create epoch-specific output folder
        epoch_output_folder = os.path.join(outputFolder, f"epoch_{epoch:04d}")
        if not os.path.exists(epoch_output_folder):
            os.makedirs(epoch_output_folder)


        trainDs = tf.data.experimental.load(pathToData + "ds_train_defGrad_loocv_50", compression='GZIP')
        valDs = tf.data.experimental.load(pathToData + "ds_valid_defGrad_loocv_50", compression='GZIP')
        
        # Generate plot
        # Plots.plot_vhb_4905(model_copy, pathToData, epoch_output_folder, rateDependent)
        Plots.plot_stress(model_copy, trainDs, valDs, epoch_output_folder)

        
        # Clean up
        del model_copy
        tf.keras.backend.clear_session()
        
        return epoch, True, None, epoch_output_folder
        
    except Exception as e:
        epoch_val = epoch if 'epoch' in locals() else extract_epoch_from_checkpoint(checkpoint_path)
        return epoch_val, False, str(e), None


def parallel_checkpoint_analysis(model_fit, pathToData, outputFolder, rateDependent, checkpoint_dir, max_workers=None):
    """
    Process all checkpoints in parallel using multiprocessing and generate plots.
    
    Args:
        model_fit: Fitted model template
        checkpoint_dir (str): Directory containing checkpoint files
        pathToData (str): Path to data directory  
        outputFolder (str): Base output folder for all plots
        rateDependent (bool): Whether the model is rate-dependent
        custom_objects (dict): Custom objects for model reconstruction
        max_workers (int, optional): Maximum number of parallel workers
        
    Returns:
        dict: Results summary with success/failure status for each epoch
    """
    # Get all checkpoint paths
    if checkpoint_dir is None:
        checkpoint_dir = outputFolder + "\\ckpt"
    checkpoint_paths = get_checkpoint_paths(checkpoint_dir)
    
    if not checkpoint_paths:
        print(f"No checkpoints found in {checkpoint_dir}")
        return {}
    
    print(f"Found {len(checkpoint_paths)} checkpoints to process")
    
    # Create main output directory for checkpoint analysis
    analysis_output_dir = os.path.join(outputFolder, "checkpoint_analysis")
    if not os.path.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir)

    # Determine number of workers (default to CPU count)
    if max_workers is None:
        max_workers = max(1, mp.cpu_count())
    
    print(f"Using {max_workers} parallel workers")
    
    # Get model configuration for serialization to worker processes
    model_config = model_fit.get_config()
    # model_config = model_fit.to_json()
    
    # Prepare arguments for each worker
    worker_args = [
        (checkpoint_path, model_config, pathToData, analysis_output_dir, rateDependent)
        for checkpoint_path in checkpoint_paths
    ]
    
    # Process checkpoints in parallel using multiprocessing
    results = {}
    
    print("Processing checkpoints in parallel...")
    
    with Pool(processes=max_workers) as pool:
        # Use pool.map to process all checkpoints in parallel
        worker_results = pool.map(process_single_checkpoint_worker, worker_args)
    
    # Process results
    for epoch, success, error, output_folder in worker_results:
        results[epoch] = {
            'success': success,
            'error': error,
            'checkpoint_path': next(cp for cp in checkpoint_paths if extract_epoch_from_checkpoint(cp) == epoch),
            'output_folder': output_folder
        }
        
        if not success:
            print(f"  ✗ Failed Epoch {epoch}: {error}")
        else:
            print(f"  ✓ Success Epoch {epoch}")
    
    # Print summary
    successful = sum(1 for r in results.values() if r['success'])
    failed = len(results) - successful
    
    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\nFailed checkpoints:")
        for epoch, result in results.items():
            if not result['success']:
                print(f"  Epoch {epoch}: {result['error']}")
    
    return results


def create_training_progress_video(checkpoint_analysis_dir, output_video_path=None, fps=2, duration_per_frame=0.5):
    """
    Create a video from training progress PDFs showing the evolution of model predictions over epochs.
    
    Args:
        checkpoint_analysis_dir (str): Directory containing epoch subfolders with PDFs
        output_video_path (str, optional): Path for output video. If None, creates in checkpoint_analysis_dir
        fps (int): Frames per second for the output video
        duration_per_frame (float): Duration to display each frame in seconds
        
    Returns:
        str: Path to the created video file
    """
    try:
        import cv2
        import numpy as np
        from pdf2image import convert_from_path
        import tempfile
        import shutil
    except ImportError as e:
        print(f"Required packages not installed: {e}")
        print("Please install: pip install opencv-python pdf2image pillow")
        print("For Windows, you may also need poppler: https://github.com/oschwartz10612/poppler-windows")
        return None
    
    # Find all epoch directories
    epoch_dirs = []
    for item in os.listdir(checkpoint_analysis_dir):
        item_path = os.path.join(checkpoint_analysis_dir, item)
        if os.path.isdir(item_path) and item.startswith('epoch_'):
            epoch_dirs.append(item_path)
    
    if not epoch_dirs:
        print(f"No epoch directories found in {checkpoint_analysis_dir}")
        return None
    
    # Sort by epoch number
    def extract_epoch_from_dirname(dirname):
        match = re.search(r'epoch_(\d+)', os.path.basename(dirname))
        return int(match.group(1)) if match else 0
    
    epoch_dirs.sort(key=extract_epoch_from_dirname)
    
    print(f"Found {len(epoch_dirs)} epoch directories")
    
    # Find the training validation panel PDFs
    pdf_files = []
    for epoch_dir in epoch_dirs:
        pdf_path = os.path.join(epoch_dir, "stretch_stress.pdf")
        if os.path.exists(pdf_path):
            pdf_files.append(pdf_path)
        else:
            print(f"Warning: No stretch_stress.pdf found in {epoch_dir}")
    
    if not pdf_files:
        print("No stretch_stress.pdf files found")
        return None
    
    print(f"Found {len(pdf_files)} PDF files to convert")
    
    # Set up output video path
    if output_video_path is None:
        output_video_path = os.path.join(checkpoint_analysis_dir, "training_progress_video.mp4")
    
    # Create temporary directory for converted images
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Converting PDFs to images...")
        
        # Convert PDFs to images
        image_files = []
        for i, pdf_path in enumerate(pdf_files):
            try:
                # Convert PDF to image (first page only)
                images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=150)
                if images:
                    # Save as temporary image
                    epoch_num = extract_epoch_from_dirname(os.path.dirname(pdf_path))
                    img_path = os.path.join(temp_dir, f"epoch_{epoch_num:04d}.png")
                    images[0].save(img_path, 'PNG')
                    image_files.append(img_path)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Converted {i + 1}/{len(pdf_files)} PDFs")
                        
            except Exception as e:
                print(f"  Error converting {pdf_path}: {e}")
                continue
        
        if not image_files:
            print("No images were successfully converted from PDFs")
            return None
        
        print(f"Successfully converted {len(image_files)} PDFs to images")
        
        # Sort image files by epoch number
        image_files.sort(key=lambda x: extract_epoch_from_dirname(x))
        
        # Read first image to get dimensions
        first_img = cv2.imread(image_files[0])
        if first_img is None:
            print(f"Could not read first image: {image_files[0]}")
            return None
        
        height, width, _ = first_img.shape
        print(f"Video dimensions: {width}x{height}")
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print("Error: Could not open video writer")
            return None
        
        print("Creating video...")
        
        # Add frames to video
        frames_per_image = int(fps * duration_per_frame)
        total_frames_added = 0
        
        for i, img_path in enumerate(image_files):
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Warning: Could not read image {img_path}")
                continue
            
            # Resize image if necessary
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            
            # Add the same frame multiple times for duration
            for _ in range(frames_per_image):
                video_writer.write(img)
                total_frames_added += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Added frames for {i + 1}/{len(image_files)} images")
        
        # Release video writer
        video_writer.release()
        
        # Verify video was created
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
            total_duration = total_frames_added / fps
            print(f"Video created successfully:")
            print(f"  Path: {output_video_path}")
            print(f"  Size: {file_size:.1f} MB")
            print(f"  Duration: {total_duration:.1f} seconds")
            print(f"  Total frames: {total_frames_added}")
            print(f"  Epochs included: {len(image_files)}")
            return output_video_path
        else:
            print("Error: Video file was not created")
            return None


def create_video_from_checkpoint_analysis(outputFolder, fps=2, duration_per_frame=0.5):
    """
    Convenience function to create training progress video from existing checkpoint analysis.
    
    Args:
        outputFolder (str): Base output folder containing checkpoint_analysis subdirectory
        fps (int): Frames per second for the output video
        duration_per_frame (float): Duration to display each frame in seconds
        
    Returns:
        str: Path to the created video file
    """
    checkpoint_analysis_dir = os.path.join(outputFolder, "checkpoint_analysis")
    
    if not os.path.exists(checkpoint_analysis_dir):
        print(f"Checkpoint analysis directory not found: {checkpoint_analysis_dir}")
        print("Please run parallel checkpoint analysis first.")
        return None
    
    return create_training_progress_video(
        checkpoint_analysis_dir, 
        fps=fps, 
        duration_per_frame=duration_per_frame
    )


if __name__ == "__main__":
    # Example usage
    print("This module provides checkpoint analysis functions.")
    print("Use parallel_checkpoint_analysis() to evaluate the material response in parallel for all checkpoints.")
    print("Use create_training_progress_video() to generate training progress videos from checkpoint PDFs generated by parallel_checkpoint_analysis().")