## Install Pytorch (recommended by Pytorch website)

`conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`


## MMSegmentation recommendations
`conda install pytorch torchvision -c pytorch`

## Install MMSegmentation
[Link](https://mmsegmentation.readthedocs.io/en/latest/get_started.html#installation)


## Dataset prep

### Annotation Format

The annotations are images of shape (H, W), the value pixel should fall in range [0, num_classes - 1]. You may use 'P' mode of pillow to create your annotation image with color.


## Semifield-Cutout Annotation Format

Cutout masks (*_masks.png) as saved as RGB, can be read using OpenCv, and easily converted to binary 8-bit using numpy. The code below converts an RGB image into grayscale and remaps pixel values to cutout class id.

```Python
mask = cv2.imread("MD_*_mask.png", 0)
mask = np.where(mask == 1, class_id, 0).astype(np.uint8) 
```
