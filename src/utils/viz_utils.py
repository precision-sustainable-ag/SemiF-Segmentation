import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import cv2


def plot_train_val_metrics(metrics_file, metrics=["train_loss", "valid_loss", "train_dataset_iou", "valid_dataset_iou"], output_file=None):
    """
    Plots training and validation metrics (e.g., loss, IoU) from a metrics CSV file after each epoch.
    Loss and IoU are separated into subplots.
    
    Args:
        metrics_file (str): Path to the metrics CSV file.
        metrics (list): List of metric column names to plot (e.g., ["train_loss", "valid_loss"]).
        output_file (str, optional): If specified, saves the plot to this file. Default is None.
    
    Returns:
        None: Displays the plot (and optionally saves it to a file).
    """
    # Load the metrics CSV into a pandas DataFrame
    metrics_df = pd.read_csv(metrics_file)

    # Separate rows for training and validation metrics
    if "train_loss" in metrics:
        train_df = metrics_df.dropna(subset=["train_loss"])
    if "valid_loss" in metrics:
        val_df = metrics_df.dropna(subset=["valid_loss"])

    # Define colors for training (dark) and validation (light) metrics
    colors = {
        "train": {"loss": "#1f77b4", "dataset_iou": "#2ca02c", "per_image_iou": "#9467bd"},  # Dark colors for training
        "valid": {"loss": "#ffa500", "dataset_iou": "#98df8a", "per_image_iou": "#c5b0d5"},  # Light colors for validation
    }

    # Initialize the subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot the loss metrics
    for metric in metrics:
        if "loss" in metric:
            if "train" in metric:
                axs[0].plot(
                    train_df["epoch"], train_df[metric],
                    label=f"Training {metric.split('_')[-1].capitalize()}",
                    marker="o", color=colors["train"]["loss"]
                )
            elif "valid" in metric:
                axs[0].plot(
                    val_df["epoch"], val_df[metric],
                    label=f"Validation {metric.split('_')[-1].capitalize()}",
                    marker="s", color=colors["valid"]["loss"]
                )

    axs[0].set_title("Loss")
    axs[0].set_ylabel("Loss")
    axs[0].legend(loc="upper right")
    axs[0].grid(True)

    # Plot the IoU metrics
    for metric in metrics:
        if "iou" in metric:
            key = "dataset_iou" if "dataset" in metric else "per_image_iou"
            if "train" in metric:
                axs[1].plot(
                    train_df["epoch"], train_df[metric],
                    label=f"Training {key.replace('_', ' ').capitalize()}",
                    marker="o", color=colors["train"].get(key, "#2ca02c")
                )
            elif "valid" in metric:
                axs[1].plot(
                    val_df["epoch"], val_df[metric],
                    label=f"Validation {key.replace('_', ' ').capitalize()}",
                    marker="s", color=colors["valid"].get(key, "#98df8a")
                )

    axs[1].set_title("IoU")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("IoU")
    axs[1].legend(loc="lower right")
    axs[1].grid(True)

    # Force integer x-axis ticks
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))


    # Adjust layout
    plt.tight_layout()

    # Show or save the plot
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    plt.close()


def viz_batch(dataloader, output_dir="output"):
    """
    Saves the images and masks from a single batch as plots to the specified folder.

    Args:
        dataloader (DataLoader): A PyTorch DataLoader object.
        output_dir (str): Path to the folder where plots will be saved.

    Returns:
        None
    """

    for images, masks, stems in dataloader:
        batch_size = images.shape[0]  # Number of images in the batch
        
        for idx in range(batch_size):
            # Convert the image to HWC format (CHW -> HWC) for visualization
        
            image = images[idx].permute(1, 2, 0).numpy()  # CHW -> HWC
            # Clip the image data to the valid range [0, 1]
            image = np.clip(image, 0, 1)
            
            # Convert the mask to 2D format for visualization
            mask = masks[idx].squeeze(0).numpy()  # Remove channel dimension
            
            # Plot the image and mask
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # Plot the image
            axes[0].imshow(image)
            axes[0].set_title(f"Image {stems[idx]}")
            axes[0].axis("off")
            
            # Plot the mask
            axes[1].imshow(mask, cmap="gray")
            axes[1].set_title(f"Mask {stems[idx]}")
            axes[1].axis("off")
            
            plt.tight_layout()
            
            # Save the plot
            batch_sample_dir = Path(output_dir, "batch_sample")
            batch_sample_dir.mkdir(parents=True, exist_ok=True)
            plot_path = Path(output_dir, "batch_sample", f"{stems[idx]}.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)  # Close the figure to save memory
            
            print(f"Saved: {plot_path}")
        
        break  # Process only the first batch

def visualize_predictions(images, masks, pr_masks, output_dir="output", num_samples=5):
    """
    Visualize and save predictions of a segmentation model.
    Parameters:
    images (Tensor): Batch of input images.
    masks (Tensor): Batch of ground truth masks.
    pr_masks (Tensor): Batch of predicted masks.
    output_dir (str, optional): Directory to save the visualizations. Defaults to "output".
    num_samples (int, optional): Number of samples to visualize. Defaults to 5.
    This function creates a side-by-side comparison of the original image, ground truth mask, 
    and predicted mask for the first `num_samples` samples in the batch. The visualizations 
    are saved as PNG files in the specified `output_dir`.
    """
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
        if idx < num_samples:  # Visualize first `num_samples` samples
            plt.figure(figsize=(12, 6))
            # Original Image
            plt.subplot(1, 3, 1)
            plt.imshow(
                image.cpu().numpy().transpose(1, 2, 0)
            )  # Convert CHW to HWC for plotting
            plt.title("Image")
            plt.axis("off")

            # Ground Truth Mask
            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask.cpu().numpy(), cmap="tab20")  # Visualize ground truth mask
            plt.title("Ground truth")
            plt.axis("off")

            # Predicted Mask
            plt.subplot(1, 3, 3)
            plt.imshow(pr_mask.cpu().numpy(), cmap="tab20")  # Visualize predicted mask
            plt.title("Prediction")
            plt.axis("off")
            plt.savefig(Path(output_dir, f"output_{idx}.png"))
        else:
            break

def side_by_side_plot(img_path, mask_gt, mask_pred, output_path: Path, class_colors=None, class_labels=None):
    """
    Plot Image | Ground Truth | Prediction side by side and save.
    """
    image = cv2.imread(str(img_path))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image)
    axs[0].set_title('Image')
    axs[0].axis('off')

    axs[1].imshow(mask_gt, cmap='jet', interpolation='nearest')
    axs[1].set_title('Ground Truth')
    axs[1].axis('off')

    axs[2].imshow(mask_pred, cmap='jet', interpolation='nearest')
    axs[2].set_title('Prediction')
    axs[2].axis('off')

    plt.tight_layout()

    # Optional: Add legend if provided
    if class_colors and class_labels:
        handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=np.array(color)/255, markersize=10)
                   for color in class_colors.values()]
        axs[2].legend(handles, class_labels.values(), bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(output_path)
    plt.close()
