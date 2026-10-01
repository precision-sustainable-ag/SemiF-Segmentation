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

def pad_to_multiple(image: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    """
    Pad a CHW image tensor to make H and W divisible by `multiple`.
    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W)
        multiple (int): The multiple to pad to.
    Returns:
        torch.Tensor: Padded image tensor (C, H_pad, W_pad)
    """
    c, h, w = image.shape

    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h == 0 and pad_w == 0:
        return image  # Already correct size

    # Pad (top=0, bottom=pad_h, left=0, right=pad_w)
    image_np = image.cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))  # CHW -> HWC

    image_np = cv2.copyMakeBorder(
        image_np,
        top=0, bottom=pad_h,
        left=0, right=pad_w,
        borderType=cv2.BORDER_REFLECT_101
    )

    image_np = np.transpose(image_np, (2, 0, 1))  # Back to CHW
    return torch.from_numpy(image_np).to(image.device).float()

def colorize_mask(mask: np.ndarray, color_dict: dict) -> np.ndarray:
        """Convert mask (H, W) to color image (H, W, 3) using color_dict"""
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in color_dict.items():
            color_mask[mask == class_id] = color
        return color_mask

def side_by_side_plot(img_path, mask_gt, mask_pred, output_path: Path, class_colors=None, class_labels=None):
    """
    Plot Image | Ground Truth | Prediction side by side and save.
    """
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Colorize masks if class_colors provided
    if mask_gt is not None:
        if class_colors:
            mask_pred = colorize_mask(mask_pred, class_colors)
            mask_gt = colorize_mask(mask_gt, class_colors) if mask_gt is not None else None
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(image)
        axs[0].set_title(f'{img_path.stem}')
        axs[0].axis('off')
        axs[1].imshow(mask_gt, cmap='jet', interpolation='nearest')
        axs[1].set_title('Ground Truth')
        axs[1].axis('off')
        axs[2].imshow(mask_pred, cmap='jet', interpolation='nearest')
        axs[2].set_title('Prediction')
        axs[2].axis('off')
    else:

        if class_colors:
            mask_pred = colorize_mask(mask_pred, class_colors)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].imshow(image)
        axs[0].set_title(f'{img_path.stem}')
        axs[0].axis('off')
        axs[1].imshow(mask_pred, cmap='jet', interpolation='nearest')
        axs[1].set_title('Prediction')
        axs[1].axis('off')

    plt.tight_layout()

    # Optional: Add legend if provided
    if class_colors and class_labels:
        handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=np.array(color)/255, markersize=10)
                   for color in class_colors.values()]
        if mask_gt is not None:
            axs[2].legend(handles, class_labels.values(), bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axs[1].legend(handles, class_labels.values(), bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def draw_instances(image: np.ndarray, masks: np.ndarray, boxes: np.ndarray, texts=None, alpha: float = 0.45) -> np.ndarray:
    """RGB image with each instance filled in its own color, outlined, and boxed.

    Args:
        image: (H, W, 3) uint8 RGB.
        masks: (N, H, W) bool.
        boxes: (N, 4) xyxy.
        texts: optional N labels drawn at the box corners (e.g. scores).
    """
    out = image.astype(np.float32)
    thickness = max(2, round(max(image.shape[:2]) / 700))
    colors = (plt.cm.tab20(np.arange(len(masks)) % 20)[:, :3] * 255).astype(np.uint8)
    for mask, color in zip(masks, colors):
        out[mask] = (1 - alpha) * out[mask] + alpha * color
    out = out.astype(np.uint8)
    for k, (mask, box, color) in enumerate(zip(masks, boxes, colors)):
        c = tuple(int(v) for v in color)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, c, thickness)
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x0, y0), (x1, y1), c, thickness)
        if texts is not None:
            scale = thickness / 2
            (tw, th), _ = cv2.getTextSize(texts[k], cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            y = max(y0, th + 4)
            cv2.rectangle(out, (x0, y - th - 4), (x0 + tw + 4, y), c, -1)
            cv2.putText(out, texts[k], (x0 + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)
    return out


def instance_plot(img_path, gt_target: dict, prediction: dict, output_path: Path,
                  score_threshold: float = 0.5, mask_threshold: float = 0.5, class_labels: dict | None = None,
                  image_scale: float = 1.0):
    """Image | ground-truth instances | predicted instances (score >= score_threshold, with scores).

    gt_target: {"boxes", "masks" (N, H, W)}, or None to leave out that panel
    (unlabeled images); prediction: a torchvision detector
    output {"boxes", "labels", "scores", "masks" (N, 1, H, W)}. Tensors or arrays.
    image_scale: the scale the image was resized by before prediction; the image
    is then padded/cropped at the bottom right to the predicted masks' size.
    """
    to_np = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)  # noqa: E731
    image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

    keep = to_np(prediction["scores"]) >= score_threshold
    scores, labels = to_np(prediction["scores"])[keep], to_np(prediction["labels"])[keep]
    pred_masks = to_np(prediction["masks"])[keep][:, 0] > mask_threshold
    if image_scale != 1:
        image = cv2.resize(image, None, fx=image_scale, fy=image_scale, interpolation=cv2.INTER_AREA)
    h, w = prediction["masks"].shape[-2:]
    image = cv2.copyMakeBorder(image[:h, :w], 0, max(h - image.shape[0], 0), 0, max(w - image.shape[1], 0),
                               cv2.BORDER_CONSTANT, value=0)
    pred_boxes = to_np(prediction["boxes"])[keep]
    multiclass = class_labels is not None and len(class_labels) > 2
    texts = [f"{class_labels.get(int(l), l)} {s:.2f}" if multiclass else f"{s:.2f}" for l, s in zip(labels, scores)]

    panels = [(image, f"{Path(img_path).stem}")]
    if gt_target is not None:
        gt_masks, gt_boxes = to_np(gt_target["masks"]).astype(bool), to_np(gt_target["boxes"])
        panels.append((draw_instances(image, gt_masks, gt_boxes), f"Ground truth: {len(gt_boxes)} instances"))
    panels.append((draw_instances(image, pred_masks, pred_boxes, texts),
                   f"Prediction: {len(pred_boxes)} instances (score >= {score_threshold})"))
    fig, axs = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6.4))
    for ax, (panel, title) in zip(axs, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()
