# 1. Problem and The Algorithm

This project addresses task 2.2 which revolves around external-beam radiotherapy for prostate cancer depending on accurate contours of prostate and organs at risk around it. If a contour is too small, tumours may be missed. Drawing these contours by hand on every MRI slice is too slow so this project builds an annotation assistant by proposing a draft segmentation for a clinicians to review and correct. The model is a 2D improved U-Net(Isensee et al., 2018), trained on 2D slices from the HipMRI study to label every pixel as one of six classes: background, body, bone, bladder, rectum or prostate. Segmenting the prostate is difficult because it occupies less than 2% of a pelvic scan and its edges, especially at the base and apex, have low contrast against neighbouring tissue.This model is compared with the standard 2D U-Net on the same data.

The ImprovedUNet2D is an encoder-decoder with skip connections. The encoder has five levels with 16, 32, 64, 128 and 256 feature channels and the first level works at full 258x128 resolution where each level below starts with a stride-2 convolution that halves the image size and doubles the channels down to 16×8. Every level then applies a context block which contains two convolutions with instance normalization, leaky ReLU and the decoder upsamples the feature to full size. At each level, the matching encoder features concatenates and merges with a localization block, a 3x3 convolution followed by a 1x1 convolution. For deep supervision, 1×1 segmentation and full resolution are each resized to full resolution and summed into the final output. This gives the lower decoder levels a direct training signal. The output has 6 channels and each pixel takes the class with the highest score. Both models are trained with cross-entropy plus soft Dice loss.

The baseline UNet2D uses the same five levels and channel width. Each level is a plain double convolution with batch normalisation and ReLU, downsampling uses max-pooling, upsampling uses transposed convolutions, and there is a single output head.

![2D Improved U-Net architecture](images/architecture.png)

# 2. Feasibility Review

Intended users:
The intended user is a radiology oncology staff contouring prostate and organs at risk for radiotherapy planning. The purpose of this prototype is to draft 2D contours for an expert to review and correct. A high global Dice can hide boundary failures at the prostate base/apex and leakage into the rectum or bladder. This prototype tests whether the Improved U-Net's multi-scale design reduces those failures compared with a plain 2D U-Net, or only improves the easy central slices.

Acceptance Criteria:

1. Prostate dice ≥ 0.75 across all test set
2. Boundary quality reported per class should be better than or equal to the baseline
3. Inference ≤ 50 ms per slice; peak training VRAM ≤ 8 GB
4. Organs at risk Dice ≥ 0.8

Model Choice & Course Concepts
The data used will be the HipMRI 2D slices dataset containg training/validation/test and 11,460/660/540, with no patient overlap. The choice of model will be the 2D improved U-net model with residual context modules, instance norm, leaky ReLU, localisation modules and deep supervision will be used. This model will be compared with baseline the standard 2D U-net that will be trained on the same dataset. Course concepts used will be dice plus entropy-loss, residual learning, normalization, and encoders-decoders.

Feasibility Evidence:
The baseline 2D U-Net was trained for 5 epochs on the full training set on an NVIDIA RTX 3070 Laptop GPU (8 GB). Training Loss fell from 0.92 to 0.12 and prostate Dice reached 0.85 by epoch 5 which is is above the 0.75 target. On the other hand, the 2D Improved U-Net reached 0.77 validation prostate Dice after a single epoch.

Technical Risks:

1. Limited GPU memory (8 GB laptop)
2. Image–mask misalignment
3. Class imbalance: prostate is < 2% of pixels
4. Patient leakage between splits

Compute Budget:
Training runs on an NVIDIA RTX 3070 Laptop GPU (8 GB). Rangpur is the backup if the laptop is unavailable or too slow.

Next planned experiment:
Train the baseline 2D U-Net and the 2D Improved U-Net for 50 epochs each with identical splits, loss, optimiser (Adam, lr 5e-4) and seed. Then compare test per-class Dice and IoU, prostate boundary F1, and apex / mid / base Dice, and inspect 3–5 boundary failure cases.

Fallback:
If the Improved U-Net doesn't reach 0.75 prostate Dice, tune the loss weighting and learning rate or crop to the pelvic region of interest.

# 3. Dependencies

| Package                     | Used for                                                      |
| --------------------------- | ------------------------------------------------------------- |
| Python                      | dataset.py, modules.py, train.py, predict.py                  |
| PyTorch                     | models, training, inference                                   |
| CUDA (bundled with PyTorch) | GPU training                                                  |
| NumPy                       | data loading (dataset.py) and plotting only, not in the model |
| nibabel                     | reading NIfTI (.nii.gz) slices                                |
| matplotlib                  | training curves and prediction figures                        |
| tqdm                        | progress bars while loading data                              |

# 4. Dataset and Pre-processing Used

Dataset:
This project uses the 2D slice version of the HipMRI Study prostate dataset. It is provided on the Rangpur cluster at /home/groups/comp3710/HipMRI*Study_open/keras_slices_data. Files are named case*<patient>_week_<week>_slice_<slice>.nii.gz, and each mask has the same name with seg* in place of case*. Patients were scanned in several weeks and each scan contributes to many consecutive slices.

| Split      | Slices | Slice size                     |
| ---------- | ------ | ------------------------------ |
| Train      | 11,460 | 11,400 × 256×128, 60 × 256×144 |
| Validation | 660    | 256×128                        |
| Test       | 540    | 256×128                        |

These numbers were retrieved from a data audit through the use of the following code:

python3 -c "
import glob, nibabel as nib
from collections import Counter
for folder in ['keras_slices_train', 'keras_slices_validate', 'keras_slices_test']:
print(folder, Counter(nib.load(i).shape for i in glob.glob(folder + '/_.nii_')))
"

Preprocessing for both models:

1. Loading. Slices are read with nibabel using the load_data_2D and to_channels functions from Appendix B of the course specification.
2. Fixed size. 11,400 training slices and all validation and test slices are 256x128. The 60 trainign 256*144 are center cropped to 256*128 by removing 8 columns from each side. The mask gets exactly the same crop. Masks are loaded with the same load_data_2D function, and each wide image has a wide mask.
3. Intensity. Each slice is z-score normalized.
4. Labels. Masks are stored as integer class labeled 0-5. These labels represent ("background", "body", "bone", "bladder", "rectum", "prostate"). They are one-hot encoded into 6 channels.
5. Augmentation. A left-right flip applied to both image and mask and a random brightness and contrast change applied to the image only with both a probability of 0.5. Validation and test slices are not augmented.

# 5. Training Setup

Both models are trained by train.py in one run, using the same loaders and settings. The architecture is the only difference.
| Setting | Value |
|---|---|
| Loss | Cross-entropy + soft Dice|
| Optimiser | Adam, learning rate 5×10⁻⁴, weight decay 1×10⁻⁵ |
| Batch size | 16 slices |
| Epochs | 50 per model |
| Random seed | 35 (`torch.manual_seed`) |
| Checkpoint selection | Epoch with the highest validation prostate Dice |
| Hardware | NVIDIA GeForce RTX 3070 Laptop GPU (8 GB) |

Optimiser. The learning rate of 5×10⁻⁴ and weight decay of 1×10⁻⁵ follow the settings used by Isensee et al. (2018) for the original Improved U-Net. The same values were then applied to the baseline model.

Checkpointing. After every epoch the model predicts the whole validation set, and the validation loss and prostate Dice are recorded. Once the validation prostate Dice beats the best one, the weights are saved to output/<model>\_best.pt. After training, the best weights are reloaded and used for all test results, so later epochs that overfit cannot make the reported model worse.

![Training Curves](outputs/training_curves.png)
Training Curves. Both models learn most of what they will learn in the first 5 epochs. For the baseline model, training loss drops from 0.9 to about 0.14 by epoch 5, then falls slowly to about 0.08 by epoch 50. For the improved model, training loss drops from 0.4 to about 0.15 by epoch 5, then falls slowly to about 0.07 by epoch 50.

Target Dice. The validation prostate Dice is above 0.75 from the first epoch and settles around 0.82 to 0.85 for both models. The improved U-Net starts lower with around 0.78 at epoch 1 and catches up around epoch 6.

Training length. Since neither model improved significantly after 10 epochs, 50 epochs were more than needed since 20 epochs would've given the same result.

6.  Results

# Artificial Intelligence Usage Disclosure
