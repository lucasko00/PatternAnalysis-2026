import os
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import HipMRIDataset, HIPMRI_ROOT, NUM_CLASSES
from modules import UNet2D, ImprovedUNet2D
from train import predict_all, dice_iou, boundary_f1, LABELS, device

PROSTATE = 5

def load_model(model_class, path):
    """Build a model and load its trained weights."""
    model = model_class(in_channels=1, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

def overlay(ax, image, mask, title):
    ax.imshow(image, cmap="gray")
    ax.imshow(torch.where(mask > 0, mask.float(), torch.nan), cmap="jet",
              vmin=0, vmax=NUM_CLASSES - 1, alpha=0.4, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def errors(ax, image, pred, truth, label=PROSTATE):
    """MRI slice with prostate errors."""
    fp = (pred == label) & (truth != label)    
    fn = (truth == label) & (pred != label)    
    rgba = torch.zeros(*fp.shape, 4)           
    rgba[fp] = torch.tensor([1.0, 0.0, 0.0, 0.8])
    rgba[fn] = torch.tensor([0.0, 0.4, 1.0, 0.8])
    ax.imshow(image, cmap="gray")
    ax.imshow(rgba.numpy(), interpolation="nearest")
    ax.set_title("Errors: red = FP, blue = FN", fontsize=9)
    ax.axis("off")

 
def slice_dice(pred, truth, label=PROSTATE):
    """Prostate Dice for one slice (1.0 if the label is in neither)."""
    p, t = pred == label, truth == label
    total = p.sum() + t.sum()
    return 1.0 if total == 0 else (2 * (p & t).sum() / total).item()

def main():
    test_set = HipMRIDataset(HIPMRI_ROOT, "test", augment=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
    images = torch.from_numpy(test_set.images)
    truth = torch.from_numpy(test_set.masks).long()
 
    models = {
        "baseline": load_model(UNet2D, os.path.join("outputs", "baseline_best.pt")),
        "improved": load_model(ImprovedUNet2D, os.path.join("outputs", "improved_best.pt")),
    }

    preds = {}
    for name, model in models.items():
        preds[name], _, ms = predict_all(model, test_loader)
        dice, iou = dice_iou(preds[name], truth)
        print(f"\n{name}: {len(test_set)} test slices, {ms:.2f} ms/slice")
        print(f"  {'label':<12}{'Dice':>8}{'IoU':>8}")
        for c in range(NUM_CLASSES):
            print(f"  {LABELS[c]:<12}{dice[c]:>8.4f}{iou[c]:>8.4f}")
        print(f"  prostate boundary F1 (2 px): {boundary_f1(preds[name], truth):.4f}")


    with_prostate = (truth == PROSTATE).flatten(1).any(1).nonzero().flatten()
    step = max(len(with_prostate) // 4, 1)
    examples = with_prostate[::step][:4].tolist()


    fig, axes = plt.subplots(len(examples), 6, figsize=(15, 4.5 * len(examples)), squeeze=False)
    for row, i in enumerate(examples):
        ax = axes[row]
        ax[0].imshow(images[i], cmap="gray")
        ax[0].set_title(f"Input (test slice {i})", fontsize=9)
        ax[0].axis("off")
        overlay(ax[1], images[i], truth[i], "Ground truth")
        for col, name in zip((2, 4), models):
            d = slice_dice(preds[name][i], truth[i])
            overlay(ax[col], images[i], preds[name][i], f"{name}\nprostate Dice {d:.2f}")
            errors(ax[col + 1], images[i], preds[name][i], truth[i])
        print(f"slice {i}: baseline {slice_dice(preds['baseline'][i], truth[i]):.3f}, "
              f"improved {slice_dice(preds['improved'][i], truth[i]):.3f} prostate Dice")
        
    fig.tight_layout()
    path = os.path.join("outputs", "predictions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
if __name__ == "__main__":
    main()