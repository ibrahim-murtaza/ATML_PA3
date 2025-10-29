# %% [markdown]
# # 📋 Task 3: Knowledge Distillation
# 
# In this task, we explore Knowledge Distillation (KD), a model compression technique where a smaller "student" model (VGG-11) learns from a larger "teacher" model (VGG-16) on the CIFAR-100 dataset.
# 
# We will implement and compare:
# 1.  **Independent Student (SI):** Baseline VGG-11 trained normally.
# 2.  **Logit Matching (LM):** Student mimics teacher's softened logits using standard KD loss.
# 3.  **Label Smoothing (LS):** Student trained with label smoothing regularization (not strictly KD).
# 4.  **Decoupled KD (DKD):** Student trained using DKD loss, separating target and non-target knowledge.

# %% [markdown]
# ---

# %% [markdown]
# ## **Part 1: Logit Matching Variants - Implementation**
# 
# This part focuses on setting up the environment, data, models, loss functions, and training loops required for Task 3.1.

# %% [markdown]
# ### **1.1. Environment Setup**
# 
# Import necessary libraries and configure the environment.

# %%
# ============================================
# 📦 Imports and Environment Setup
# ============================================
import json
import math
import os
import time
from pathlib import Path  # Added for path object

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast  # For Mixed Precision
from torch.utils.data import DataLoader
from torchvision.models import (  # Using BatchNorm versions + weights enum
    VGG16_BN_Weights, vgg11_bn, vgg16_bn)

# Ensure tqdm uses the non-notebook version
os.environ["TQDM_NOTEBOOK"] = "0"
from tqdm import tqdm

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# For reproducibility and speed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed(SEED)
    cudnn.benchmark = True  # Speeds up training if input sizes don't vary
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for cuDNN

# Create directories for saving models and results
RESULTS_DIR = Path("./results/task3_1")
MODEL_DIR = RESULTS_DIR / "models"
LOG_DIR = RESULTS_DIR / "logs"
FIGURE_DIR = RESULTS_DIR / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Removed dependency on non-existent 'pytorch_cifar_models'.
print("✅ Using torchvision VGG-16 ImageNet weights; will swap head for CIFAR-100.")
print("✅ Environment setup complete.")# Removed dependency on non-existent 'pytorch_cifar_models'.
print("✅ Using torchvision VGG-16 ImageNet weights; will swap head for CIFAR-100.")
print("✅ Environment setup complete.")

# %% [markdown]
# ### **1.2. Dataset and DataLoaders**
# 
# Define transformations and create DataLoaders for CIFAR-100, using larger batch sizes suitable for the hardware.

# %%
# ============================================
# 💾 Dataset Setup (CIFAR-100)
# ============================================
print('==> Preparing CIFAR-100 data..')

# Standard transformations for CIFAR-100
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# Hyperparameters for DataLoader - Increased Batch Size
BATCH_SIZE = 1024 # Increased for powerful GPU
TEST_BATCH_SIZE = BATCH_SIZE * 2 # Increase test batch size as well
NUM_WORKERS = 8 # Adjust based on your system CPU cores

# Custom dataset that returns indices
class CIFAR100WithIndex(torchvision.datasets.CIFAR100):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

# Download and load datasets
trainset = CIFAR100WithIndex(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(
    testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False)

num_classes = 100
print(f"✅ CIFAR-100 loaded. Train batches: {len(trainloader)}, Test batches: {len(testloader)}")

# ============================================
# 🎲 Negative Sampling for CRD
# ============================================
def get_negative_samples(batch_size, n_data, K, device):
    """Sample K negatives for each sample in the batch.
    First column will be replaced with positive indices in training loop."""
    sample_idx = torch.randint(0, n_data, (batch_size, K + 1))
    return sample_idx.to(device)

print("✅ Negative sampling function ready.")

# %% [markdown]
# ### **1.3. Model Definitions**
# 
# Define VGG-11 (student) and VGG-16 (teacher) architectures. Prioritize loading a pretrained teacher, with fallbacks for loading from a path or training from scratch.

# %%
# ============================================
# 🧐 Model Definitions (VGG)
# ============================================

# --- Student Model (VGG-11) ---
def create_student_model(num_classes=100):
    model = vgg11_bn(weights=None, num_classes=num_classes) # Train from scratch
    print(f"   Student VGG-11_bn created (Params: {sum(p.numel() for p in model.parameters()):,})")
    return model

# --- Teacher Model (VGG-16) ---
def create_teacher_model(num_classes=100, local_path=None):
    # 1) Try local checkpoint first
    if local_path and os.path.exists(local_path):
        model = vgg16_bn(weights=None, num_classes=num_classes)
        checkpoint = torch.load(local_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint.get('net', checkpoint)))
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print("   [OK] Loaded from local checkpoint")
        return model, True
    
    # 2) Load from torch.hub (chenyaofo)
    try:
        print("   Loading VGG-16 from torch.hub (chenyaofo/pytorch-cifar-models)...")
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)
        print("   [OK] Loaded pretrained VGG-16 (CIFAR-100, 74.00% acc)")
        return model, True
    except Exception as e:
        print(f"   [WARNING] torch.hub failed: {e}")
        # Fallback to ImageNet
        model = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        print("   [OK] Loaded ImageNet weights (needs fine-tuning)")
        return model, False

print("✅ VGG model definitions ready.")

# %% [markdown]
# ### **1.4. Loss Function Implementations**
# 
# Implement the loss functions for Basic Logit Matching (KD), Label Smoothing (LS), and Decoupled KD (DKD) from scratch, incorporating the fixes.

# %%
# ============================================
# ኪ Loss Function Implementations
# ============================================


def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """Standard Knowledge Distillation loss (Hinton et al., 2015)
    Combines KL divergence loss on soft targets (weight 1-alpha) and cross-entropy on hard targets (weight alpha).
    Corrected alpha convention: alpha=0.7 means 70% hard loss, 30% soft loss.
    """
    hard_loss = F.cross_entropy(student_logits, labels)

    # Ensure teacher_logits are detached if they come directly from the teacher model during KD training
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits.detach() / T, dim=1),  # Use detached teacher logits
        reduction="batchmean",
    ) * (
        T * T
    )  # Scale by T^2 as per Hinton et al.

    # Weighted combination (Corrected: alpha for hard loss, 1-alpha for soft loss)
    return alpha * hard_loss + (1.0 - alpha) * soft_loss


def label_smoothing_loss(logits, labels, epsilon=0.1, num_classes=100):
    """Cross entropy loss with label smoothing regularization."""
    log_probs = F.log_softmax(logits, dim=1)
    # Create target distribution with smoothing
    targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
    targets = (1 - epsilon) * targets + epsilon / num_classes

    # Compute loss (negative log likelihood for smoothed targets)
    loss = -(targets * log_probs).sum(dim=1).mean()
    return loss


def dkd_loss(student_logits, teacher_logits, target_labels, alpha=1.0, beta=8.0, T=4.0, warmup_factor=1.0):
    """Decoupled Knowledge Distillation loss (Zhao et al., 2022)
    Separates loss into Target Class KD (TCKD) and Non-Target Class KD (NCKD).
    NOTE: This loss REPLACES the standard CrossEntropy loss, as TCKD handles the target class.
    """
    # Add CE anchor on ground-truth labels ---
    ce_loss = F.cross_entropy(student_logits, target_labels)

    # Soften logits with temperature
    p_s = F.softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits.detach() / T, dim=1)  # Use detached teacher logits

    # --- Target Class Knowledge Distillation (TCKD) ---
    # Get probabilities for the true target class
    p_s_target = p_s.gather(1, target_labels.unsqueeze(1)).squeeze()
    p_t_target = p_t.gather(1, target_labels.unsqueeze(1)).squeeze()

    # Get probabilities for the sum of non-target classes
    p_s_non_target = 1.0 - p_s_target
    p_t_non_target = 1.0 - p_t_target

    # Clamp values to avoid log(0)
    eps = 1e-7
    p_s_target_clamp = p_s_target.clamp(min=eps, max=1.0 - eps)
    p_t_target_clamp = p_t_target.clamp(min=eps, max=1.0 - eps)
    p_s_non_target_clamp = p_s_non_target.clamp(min=eps, max=1.0 - eps)
    p_t_non_target_clamp = p_t_non_target.clamp(min=eps, max=1.0 - eps)

    # Binary KL divergence for TCKD (target vs non-target)
    tckd = p_t_target_clamp * torch.log(
        p_t_target_clamp / p_s_target_clamp
    ) + p_t_non_target_clamp * torch.log(p_t_non_target_clamp / p_s_non_target_clamp)
    tckd = tckd.mean()  # Average over batch

    # --- Non-Target Class Knowledge Distillation (NCKD) ---
    # Create a mask to zero out the target class probability
    mask = torch.ones_like(p_s).scatter_(1, target_labels.unsqueeze(1), 0).bool()

    # Select non-target probabilities
    p_s_non_target_dist = p_s[mask].view(p_s.size(0), -1)
    p_t_non_target_dist = p_t[mask].view(p_t.size(0), -1)

    # Re-normalize non-target probabilities to sum to 1
    p_s_non_target_dist = p_s_non_target_dist / p_s_non_target.unsqueeze(1).clamp(
        min=eps
    )  # Add clamp for stability
    p_t_non_target_dist = p_t_non_target_dist / p_t_non_target.unsqueeze(1).clamp(
        min=eps
    )  # Add clamp for stability

    # KL divergence for NCKD (among non-target classes)
    # Use log_softmax for student and softmax for teacher for numerical stability in KLDivLoss
    nckd = F.kl_div(
        torch.log(p_s_non_target_dist.clamp(min=eps)),  # ✅ Just log
        p_t_non_target_dist,  # ✅ Already normalized
        reduction="batchmean",
    )

    # Combine losses and scale by T^2 (consistent with KD scaling)
    # Note: No separate CE loss needed, TCKD handles the target class information
    return ce_loss + warmup_factor * (alpha * tckd + beta * nckd) * (T * T)


print("✅ Loss functions (KD, LS, DKD) defined.")

# %% [markdown]
# ### **1.5. Training and Evaluation Loops**
# 
# Define generic functions for training one epoch and evaluating the model, incorporating mixed precision (AMP) and Top-5 accuracy.

# %%
# ============================================
# 🔁 Training & Evaluation Loops
# ============================================

scaler = torch.amp.GradScaler("cuda")  # For Mixed Precision Training


def accuracy_topk(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[f"top{k}"] = correct_k.mul_(100.0 / batch_size).item()
        return res


def train_epoch(
    model,
    teacher_model,
    dataloader,
    optimizer,
    loss_type="ce",
    epoch=0,
    num_epochs=200,
    use_amp=True,
    **loss_kwargs,
):
    model.train()
    if teacher_model:
        teacher_model.eval()  # Teacher should be in eval mode

    total_loss = 0
    total_top1 = 0
    total_top5 = 0
    num_samples = 0

    # Use tqdm for progress bar with more info
    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
    )

    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):  # Enable AMP
            student_logits = model(inputs)

            # Calculate loss based on the specified type
            if loss_type == "ce":  # Standard Cross Entropy (for SI)
                loss = F.cross_entropy(student_logits, targets)
            elif loss_type == "ls":  # Label Smoothing
                loss = label_smoothing_loss(
                    student_logits,
                    targets,
                    epsilon=loss_kwargs.get("epsilon", 0.1),
                    num_classes=num_classes,
                )
            elif loss_type == "kd":  # Standard Knowledge Distillation
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)
                loss = kd_loss(
                    student_logits,
                    teacher_logits,
                    targets,
                    T=loss_kwargs.get("T", 4.0),
                    alpha=loss_kwargs.get("alpha", 0.7),
                )
            elif loss_type == "dkd":  # Decoupled Knowledge Distillation
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)

                warmup_factor = min(epoch / 20, 1.0)
                loss = dkd_loss(
                    student_logits,
                    teacher_logits,
                    targets,
                    alpha=loss_kwargs.get("dkd_alpha", 1.0),
                    beta=loss_kwargs.get("dkd_beta", 8.0),
                    T=loss_kwargs.get("T", 4.0),
                    warmup_factor=warmup_factor
                )
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

        # Scaled backward pass for AMP
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_size
        accs = accuracy_topk(student_logits, targets, topk=(1, 5))
        total_top1 += accs["top1"] * batch_size
        total_top5 += accs["top5"] * batch_size
        num_samples += batch_size

        # Update progress bar description with running averages
        pbar.set_postfix(
            {
                "Loss": f"{total_loss/num_samples:.4f}",
                "Acc@1": f"{total_top1/num_samples:.2f}%",
                "Acc@5": f"{total_top5/num_samples:.2f}%",
            }
        )

    avg_loss = total_loss / num_samples
    avg_acc_top1 = total_top1 / num_samples
    avg_acc_top5 = total_top5 / num_samples
    return avg_loss, avg_acc_top1, avg_acc_top5


def evaluate(model, dataloader, epoch=None, num_epochs=None, use_amp=True):
    """
    Evaluate model on dataloader.
    Args:
        epoch, num_epochs: Optional, for progress bar display. Pass None to hide epoch info.
    Returns:
        avg_loss, accuracy_top1, accuracy_top5
    """
    model.eval()
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    # Set progress bar description
    if epoch is not None and num_epochs is not None:
        desc = f"Epoch {epoch+1}/{num_epochs} [Eval]"
    else:
        desc = "Evaluating"

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for batch_data in pbar:
            # Handle both regular and indexed datasets
            if len(batch_data) == 3:
                inputs, targets, _ = batch_data  # Ignore indices during evaluation
            else:
                inputs, targets = batch_data
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)

            total_loss += loss.item() * batch_size
            total += batch_size

            # Top-1 accuracy
            _, pred_top1 = outputs.max(1)
            correct_top1 += pred_top1.eq(targets).sum().item()

            # Top-5 accuracy
            _, pred_top5 = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_top5 += (
                pred_top5.eq(targets.view(-1, 1).expand_as(pred_top5)).sum().item()
            )

            # Update progress bar
            avg_loss = total_loss / total
            acc_top1 = 100.0 * correct_top1 / total
            acc_top5 = 100.0 * correct_top5 / total
            pbar.set_postfix(
                {
                    "Loss": f"{avg_loss:.4f}",
                    "Acc@1": f"{acc_top1:.2f}%",
                    "Acc@5": f"{acc_top5:.2f}%",
                }
            )

    return avg_loss, acc_top1, acc_top5


print("✅ Training and evaluation loops defined (with AMP and Top-5).")

# %% [markdown]
# ### **1.6. Training Configurations**
# 
# Define hyperparameters and configurations for each of the four models (SI, LM, LS, DKD).
# 
# **Note:** This cell is intentionally placed *before* the Teacher Loading/Training cell so the hyperparameters are available if the teacher needs training.

# %%
# ============================================
# ⚙️ Training Configurations
# ============================================

NUM_EPOCHS = 200 # Standard for CIFAR
BASE_LR = 0.05 # As per RepDistiller experiments for VGG
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LR_DECAY_EPOCHS = [100, 150] # Decay LR at epochs 100 and 150
LR_DECAY_RATE = 0.1
WARMUP_EPOCHS = 5 # Number of epochs for linear LR warmup
USE_AMP = True # Enable/disable Mixed Precision

configs = {
    'SI': {
        'model_name': 'student_vgg11_SI',
        'loss_type': 'ce',
        'use_teacher': False,
        'loss_kwargs': {},
        'epochs': NUM_EPOCHS
    },
    'LM': {
        'model_name': 'student_vgg11_LM',
        'loss_type': 'kd',
        'use_teacher': True,
        'loss_kwargs': {'T': 4.0, 'alpha': 0.7}, # 70% hard loss, 30% soft loss
        'epochs': NUM_EPOCHS
    },
    'LS': {
        'model_name': 'student_vgg11_LS',
        'loss_type': 'ls',
        'use_teacher': False,
        'loss_kwargs': {'epsilon': 0.1},
        'epochs': NUM_EPOCHS
    },
    'DKD': {
        'model_name': 'student_vgg11_DKD',
        'loss_type': 'dkd',
        'use_teacher': True,
        'loss_kwargs': {'dkd_alpha': 1.0, 'dkd_beta': 8.0, 'T': 4.0},
        'epochs': NUM_EPOCHS
    },
}

# Function to adjust learning rate with warmup
def adjust_learning_rate_with_warmup(optimizer, epoch, warmup_epochs, lr_decay_epochs, lr_decay_rate, base_lr):
    if epoch < warmup_epochs:
        # Linear warmup
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        # Standard decay schedule after warmup
        lr = base_lr
        num_decays = sum(epoch >= decay_epoch for decay_epoch in lr_decay_epochs)
        lr *= (lr_decay_rate ** num_decays)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

print("✅ Training configurations defined (incl. warmup).")

# %% [markdown]
# ### **1.7. Load/Train Teacher Model**
# 
# Load the pretrained VGG-16 teacher model from torch.hub (chenyaofo/pytorch-cifar-models), which provides CIFAR-100 pretrained weights. Falls back to local checkpoint if available, or ImageNet weights if download fails.

# %%
# ============================================
# 👨‍🏫 Teacher Model Loading / Training
# ============================================

TEACHER_MODEL_PATH = MODEL_DIR / 'teacher_vgg16_cifar100.pth'
TEACHER_NEEDS_TRAINING = False

print("Attempting to load teacher model...")
teacher_model, teacher_loaded = create_teacher_model(
    num_classes=num_classes,
    local_path=TEACHER_MODEL_PATH
)

teacher_model = teacher_model.to(device)

if not teacher_loaded:
    TEACHER_NEEDS_TRAINING = True
    print("\n" + "="*60)
    print("🔥 FINE-TUNING TEACHER ON CIFAR-100 (from ImageNet weights)")
    print("   This will take approximately 1.5-2 hours.")
    print("="*60 + "\n")

    teacher_optimizer = optim.SGD(
        teacher_model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, nesterov=True
    )
    teacher_history = {'train_loss': [], 'train_acc_top1': [], 'train_acc_top5': [],
                       'test_loss': [], 'test_acc_top1': [], 'test_acc_top5': []}
    best_teacher_acc = 0.0
    teacher_start_time = time.time()

    for epoch in range(NUM_EPOCHS): # Train teacher for the same duration
        current_lr = adjust_learning_rate_with_warmup(
            teacher_optimizer, epoch, WARMUP_EPOCHS, LR_DECAY_EPOCHS, LR_DECAY_RATE, BASE_LR
        )
        print(f"\n--- Training Teacher: Epoch {epoch+1}/{NUM_EPOCHS}, LR: {current_lr:.5f} ---")

        train_loss, train_acc1, train_acc5 = train_epoch(
            teacher_model, None, trainloader, teacher_optimizer, loss_type='ce',
            epoch=epoch, num_epochs=NUM_EPOCHS, use_amp=True
        )
        test_loss, test_acc1, test_acc5 = evaluate(
            teacher_model, testloader, epoch=epoch, num_epochs=NUM_EPOCHS, use_amp=True
        )

        # Store history
        teacher_history['train_loss'].append(train_loss)
        teacher_history['train_acc_top1'].append(train_acc1)
        teacher_history['train_acc_top5'].append(train_acc5)
        teacher_history['test_loss'].append(test_loss)
        teacher_history['test_acc_top1'].append(test_acc1)
        teacher_history['test_acc_top5'].append(test_acc5)

        # Save best teacher model
        if test_acc1 > best_teacher_acc:
            print(f"   ✨ New best teacher accuracy: {test_acc1:.2f}% (Top-5: {test_acc5:.2f}%). Saving...")
            best_teacher_acc = test_acc1
            torch.save({'model_state_dict': teacher_model.state_dict(), 'accuracy': best_teacher_acc}, TEACHER_MODEL_PATH)

    teacher_training_time = time.time() - teacher_start_time
    print(f"\n🏁 Teacher training finished in {teacher_training_time/60:.2f} minutes.")
    print(f"   Best Teacher Accuracy: {best_teacher_acc:.2f}%")

    # Save teacher history
    with open(LOG_DIR / 'teacher_vgg16_training_log.json', 'w') as f:
        json.dump(teacher_history, f)

# Ensure teacher is in eval mode for distillation
teacher_model.eval()

# Evaluate final teacher performance
print("\nEvaluating teacher model on test set...")
teacher_loss, teacher_acc1, teacher_acc5 = evaluate(
    teacher_model, testloader, 
    epoch=None, num_epochs=None, use_amp=False  # Pass None to hide epoch display
)
print(f"Teacher Model Test Accuracy: Top-1 = {teacher_acc1:.2f}%, Top-5 = {teacher_acc5:.2f}%\n")

# %% [markdown]
# ### **1.8. Run Training for All Models**
# 
# Iterate through the configurations and train each student model.

# %%
# ============================================
# 🔥 Run Student Training
# ============================================

all_model_results = {}

# Check if all models already exist before starting the loop
all_trained = all((MODEL_DIR / f"{config['model_name']}_best.pth").exists()
                  for config in configs.values())

if all_trained:
    print("\n===================== SKIPPING TRAINING =====================")
    print("All required student model checkpoints (.pth files) found in:")
    print(f"  {MODEL_DIR}")
    print("Proceeding directly to results loading and summary.")
    # Load existing results if training is skipped, needed for the summary section
    print("\nLoading existing results for summary...")
    for model_key, config in configs.items():
        results_path = LOG_DIR / f"{config['model_name']}_results.json"
        if results_path.exists():
            print(f"   Loading results for {model_key}...")
            try:
                with open(results_path, 'r') as f:
                    # Load the results, but don't overwrite if already loaded later
                    if model_key not in all_model_results:
                         all_model_results[model_key] = json.load(f)
            except Exception as e:
                print(f"   ⚠️ Error loading results for {model_key}: {e}")
        else:
            print(f"   ⚠️ Results file not found for {model_key} at {results_path}. Summary might be incomplete.")

else:
    print("\nStarting student training process as not all models were found...")
    # --- Original Training Loop Start ---
    for model_key, config in configs.items():
        model_save_path = MODEL_DIR / f"{config['model_name']}_best.pth"
        # Optional: Check if *this specific* model exists, skip if it does
        if model_save_path.exists():
             print(f"\n--- Skipping {model_key}: Checkpoint already exists at {model_save_path} ---")
             # Load results for this skipped model if needed for summary
             results_path = LOG_DIR / f"{config['model_name']}_results.json"
             if results_path.exists() and model_key not in all_model_results:
                 print(f"   Loading existing results for {model_key}...")
                 try:
                    with open(results_path, 'r') as f:
                        all_model_results[model_key] = json.load(f)
                 except Exception as e:
                    print(f"   ⚠️ Error loading results for {model_key}: {e}")
             continue # Skip to the next model configuration

        # --- The rest of your training loop from the previous version goes here ---
        print(f"\n{'='*25} Training Student: {model_key} ({config['model_name']}) {'='*25}")

        # --- Model Initialization ---
        student_model = create_student_model(num_classes=num_classes).to(device)
        current_teacher = teacher_model if config['use_teacher'] else None
        if config['use_teacher'] and teacher_model is None:
            print("   WARNING: Distillation requested but teacher wasn't loaded properly.")

        # --- Optimizer ---
        optimizer = optim.SGD(student_model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)

        # --- Training Loop ---
        best_acc_top1 = 0.0
        best_acc_top5 = 0.0
        model_history = {'train_loss': [], 'train_acc_top1': [], 'train_acc_top5': [],
                         'test_loss': [], 'test_acc_top1': [], 'test_acc_top5': []}
        start_time = time.time()

        for epoch in range(config['epochs']):
            current_lr = adjust_learning_rate_with_warmup(
                optimizer, epoch, WARMUP_EPOCHS, LR_DECAY_EPOCHS, LR_DECAY_RATE, BASE_LR
            )

            train_loss, train_acc1, train_acc5 = train_epoch(
                student_model,
                current_teacher, # Pass the potentially fine-tuned teacher
                trainloader,
                optimizer,
                loss_type=config['loss_type'],
                epoch=epoch,
                num_epochs=config['epochs'],
                use_amp=USE_AMP,
                **config['loss_kwargs']
            )
            test_loss, test_acc1, test_acc5 = evaluate(
                student_model, testloader, epoch=epoch,
                num_epochs=config['epochs'], use_amp=USE_AMP
            )

            # Store history
            model_history['train_loss'].append(train_loss)
            model_history['train_acc_top1'].append(train_acc1)
            model_history['train_acc_top5'].append(train_acc5)
            model_history['test_loss'].append(test_loss)
            model_history['test_acc_top1'].append(test_acc1)
            model_history['test_acc_top5'].append(test_acc5)

            # Save best model checkpoint based on Top-1 accuracy
            if test_acc1 > best_acc_top1:
                print(f"   ✨ New best Top-1 Acc: {test_acc1:.2f}% (Top-5: {test_acc5:.2f}%) (prev Top-1: {best_acc_top1:.2f}%). Saving model...")
                best_acc_top1 = test_acc1
                best_acc_top5 = test_acc5 # Store corresponding Top-5
                state = {
                    'model_state_dict': student_model.state_dict(),
                    'accuracy_top1': best_acc_top1,
                    'accuracy_top5': best_acc_top5,
                    'epoch': epoch,
                    'config': config # Save config for reference
                }
                # model_save_path is defined outside the loop now
                torch.save(state, model_save_path)

        # --- Finalization ---
        training_time = time.time() - start_time
        print(f"\n🏁 Training finished for {model_key} in {training_time/60:.2f} minutes.")
        print(f"   Best Test Accuracy: Top-1={best_acc_top1:.2f}%, Top-5={best_acc_top5:.2f}%")

        # --- Inference Time Measurement ---
        print("   Measuring inference time...")
        student_model.eval()
        # Ensure we load the best weights for inference measurement
        best_model_path = MODEL_DIR / f"{config['model_name']}_best.pth" # Corrected path
        if best_model_path.exists():
            best_checkpoint = torch.load(best_model_path, map_location=device)
            student_model.load_state_dict(best_checkpoint['model_state_dict'])
            print(f"   Loaded best weights from epoch {best_checkpoint.get('epoch', 'N/A')} for inference test.") # Use .get for safety
        else:
            print("   Warning: Best checkpoint not found, measuring inference on final weights.")

        dummy_input = torch.randn(TEST_BATCH_SIZE, 3, 32, 32).to(device) # Use test batch size
        repetitions = 100
        warmup = 10
        total_time = 0
        with torch.no_grad():
            for i in range(repetitions + warmup):
                if device == 'cuda': torch.cuda.synchronize()
                start = time.perf_counter()
                with autocast(enabled=USE_AMP):
                    _ = student_model(dummy_input)
                if device == 'cuda': torch.cuda.synchronize()
                end = time.perf_counter()
                if i >= warmup:
                     total_time += (end - start)
        avg_inference_time_ms = (total_time / repetitions) * 1000
        print(f"   Average Inference Time per batch ({TEST_BATCH_SIZE} images): {avg_inference_time_ms:.2f} ms")

        # Save final results and history
        final_results = {
            'best_test_acc_top1': best_acc_top1,
            'best_test_acc_top5': best_acc_top5,
            'final_train_acc_top1': train_acc1,
            'final_train_acc_top5': train_acc5,
            'final_test_acc_top1': test_acc1,
            'final_test_acc_top5': test_acc5,
            'training_time_minutes': training_time / 60,
            'avg_inference_time_ms_per_batch': avg_inference_time_ms,
            'config': config,
            'history': model_history
        }
        all_model_results[model_key] = final_results

        # Save results to JSON
        results_save_path = LOG_DIR / f"{config['model_name']}_results.json"
        with open(results_save_path, 'w') as f:
            # Convert history items to float for JSON
            serializable_history = {k: [float(item) for item in v] for k, v in model_history.items()}
            final_results['history'] = serializable_history
            json.dump(final_results, f, indent=4)
        print(f"   Results saved to {results_save_path}")
    # --- End of Original Training Loop ---

    print("\n===================== ALL STUDENT TRAINING CHECKS/RUNS COMPLETE =====================")

# %% [markdown]
# ### **1.9. Results Summary & Visualization**
# 
# Load the saved results, display a summary table, and plot the training curves.

# %%
# ============================================
# 📊 Results Summary & Visualization
# ============================================

summary_data = []
loaded_results = {}

# Add Teacher results
summary_data.append({
    'Model': 'Teacher (VGG-16)',
    'Method': 'N/A',
    'Best Test Acc@1 (%)': f"{teacher_acc1:.2f}",
    'Best Test Acc@5 (%)': f"{teacher_acc5:.2f}",
    'Training Time (min)': 'N/A' if 'teacher_training_time' not in globals() else f"{teacher_training_time/60:.2f}",
    'Inference (ms/batch)': 'N/A'
})

# Load student results from JSON files
for model_key, config in configs.items():
    results_path = LOG_DIR / f"{config['model_name']}_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
            loaded_results[model_key] = results
            summary_data.append({
                'Model': f"Student (VGG-11) - {model_key}",
                'Method': model_key,
                'Best Test Acc@1 (%)': f"{results['best_test_acc_top1']:.2f}",
                'Best Test Acc@5 (%)': f"{results['best_test_acc_top5']:.2f}",
                'Training Time (min)': f"{results['training_time_minutes']:.2f}",
                'Inference (ms/batch)': f"{results['avg_inference_time_ms_per_batch']:.2f}"
            })
    else:
        print(f"Warning: Results file not found for {model_key}")

summary_df = pd.DataFrame(summary_data)

print("\n--- Task 3.1 Results Summary ---")
print(summary_df.to_string(index=False))

# Save summary table
summary_df.to_csv(LOG_DIR / 'task3_1_summary.csv', index=False)
print(f"\n✅ Summary table saved to {LOG_DIR / 'task3_1_summary.csv'}")

# --- Plot Training Curves ---
print("\nGenerating training curves plot...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # Changed to 2x1

colors = sns.color_palette("husl", len(loaded_results))

for i, (model_key, results) in enumerate(loaded_results.items()):
    epochs_ran = len(results['history']['train_loss'])
    epoch_axis = range(1, epochs_ran + 1)

    # Plot Loss
    axes[0].plot(epoch_axis, results['history']['train_loss'], label=f'{model_key} Train', color=colors[i], linewidth=2)
    axes[0].plot(epoch_axis, results['history']['test_loss'], label=f'{model_key} Test', linestyle='--', color=colors[i], linewidth=2)

    # Plot Accuracy
    axes[1].plot(epoch_axis, results['history']['test_acc_top1'], label=f'{model_key} Test Acc@1', color=colors[i], linewidth=2)

axes[0].set_title('Training & Test Loss Curves', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].tick_params(axis='both', which='major', labelsize=10)

axes[1].set_title('Test Accuracy (Top-1)', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].tick_params(axis='both', which='major', labelsize=10)

plt.suptitle('Task 3.1: Logit Matching Variants Training Comparison', fontsize=18, fontweight='bold', y=1.00)
plt.tight_layout()
plot_save_path = FIGURE_DIR / 'task3_1_training_curves.png'
plt.savefig(plot_save_path, dpi=300)
print(f"✅ Training curves plot saved to {plot_save_path}")
plt.show()

# %% [markdown]
# ---
# ## **Part 2: State-of-the-Art KD Methods - Implementation**
# 
# In this part, we implement and compare two advanced KD methods against our Part 1 baseline:
# 1.  **Hint-based Distillation (FitNets):** Transfer intermediate layer representations.
# 2.  **Contrastive Representation Distillation (CRD):** Align representations using contrastive learning.
# 
# We will compare these with our Part 1 results (SI and LM).

# %% [markdown]
# ### **2.1. Hint-Based Distillation (FitNets) Implementation**
# 
# First, we implement the Hint loss (FitNets). This requires a 'connector' module (typically a 1x1 conv) to match the feature dimensions between the student and teacher's hint layers.

# %%
# ============================================
# 🎯 Hint Loss & Connector Implementation
# ============================================

class Connector(nn.Module):
    """1x1 convolution to match student and teacher feature dimensions"""
    def __init__(self, student_channels, teacher_channels):
        super(Connector, self).__init__()
        self.connector = nn.Conv2d(student_channels, teacher_channels, 
                                   kernel_size=1, stride=1, padding=0, bias=False)
        # Initialize weights (optional but good practice)
        nn.init.kaiming_normal_(self.connector.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        return self.connector(x)

def hint_loss(student_features_connected, teacher_features):
    """
    FitNets hint loss: MSE between (connected) intermediate layer features.
    Assumes student_features_connected and teacher_features have same dimensions.
    """
    # Detach teacher features to prevent gradients from flowing back to the teacher
    return F.mse_loss(student_features_connected, teacher_features.detach())

print("✅ Hint loss function and Connector module defined.")

# %% [markdown]
# ### **2.2. Modified VGG for Feature Extraction**
# 
# We need to modify our VGG models to return intermediate features (hints) in addition to the final logits. We'll create a wrapper class for this, as specified in the assignment manual and `RepDistiller`.

# %%
# ============================================
# 🔧 Modified VGG Models for Feature Extraction
# ============================================

class VGG_with_hints(nn.Module):
    """Wrapper around VGG to extract intermediate features"""
    def __init__(self, base_model, hint_layer_idx=None, use_penultimate=False):
        super(VGG_with_hints, self).__init__()
        self.base_model = base_model
        self.hint_layer_idx = hint_layer_idx
        self.use_penultimate = use_penultimate
        self.features_module_list = base_model.features
        
        # Check if model has avgpool (torchvision VGG) or not (chenyaofo VGG)
        self.has_avgpool = hasattr(base_model, 'avgpool')
        
    def forward(self, x, return_features=False):
        if not return_features:
            return self.base_model(x)
        
        features = None
        hint_features = None
        
        # Iterate through the features sequential module
        for idx, layer in enumerate(self.features_module_list):
            x = layer(x)
            if idx == self.hint_layer_idx:
                hint_features = x  # Extract hint features at specified index
        
        # Handle pooling based on model type
        if self.has_avgpool:
            x = self.base_model.avgpool(x)
        else:
            # chenyaofo models use AdaptiveAvgPool2d as last layer in features
            pass  # Already pooled in features
        
        # ✅ FIXED: Flatten BEFORE saving as penultimate features
        penultimate_features = torch.flatten(x, 1)  # Now [batch, 512]
        
        # Pass through classifier
        logits = self.base_model.classifier(penultimate_features)
        
        # Return appropriate features based on use case
        if self.use_penultimate:
            features = penultimate_features  # For CRD: use flattened penultimate [batch, 512]
        else:
            features = hint_features  # For HINTS: use intermediate conv features [batch, C, H, W]
            if features is None:
                print(f"Warning: Hint layer index {self.hint_layer_idx} was not found.")
        
        return logits, features

print("✅ VGG with feature extraction wrapper defined.")

# %% [markdown]
# ### **2.3. Contrastive Representation Distillation (CRD) Implementation**
# 
# Next, we implement the CRD loss. This requires an 'embedding' module (a Linear layer) to project features into a common space and a contrastive loss function (InfoNCE).
# 
# **Note:** As outlined in the progress summary, for simplicity and to avoid the complexity of a full memory bank (as in the original paper [11] and `RepDistiller/crd/memory.py`), we will use a simplified **in-batch** contrastive loss. This treats other samples in the same batch as negatives, which is a common and effective simplification.

# %%
# ============================================
# 🎯 CRD Loss Implementation (In-Batch)
# ============================================

class Embed(nn.Module):
    """Embedding module for CRD"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = lambda x: F.normalize(x, p=2, dim=1)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1) # Flatten
        x = self.linear(x)
        x = self.l2norm(x)
        return x

class ContrastMemory(nn.Module):
    """Memory bank for CRD"""
    def __init__(self, feat_dim, n_data, nce_k, nce_t, nce_m):
        super(ContrastMemory, self).__init__()
        self.K = nce_k  # 16384 negatives
        self.T = nce_t  # 0.07 temperature
        self.momentum = nce_m  # 0.5 momentum
        self.Z_s = None
        self.Z_t = None
        
        # Initialize memory banks [50000, 128]
        stdv = 1. / math.sqrt(feat_dim / 3.)
        self.register_buffer('memory_s', torch.rand(n_data, feat_dim).mul_(2*stdv).add_(-stdv))
        self.register_buffer('memory_t', torch.rand(n_data, feat_dim).mul_(2*stdv).add_(-stdv))
    
    def forward(self, feat_s, feat_t, idx, sample_idx):
        bs = feat_s.size(0)
        feat_dim = self.memory_s.size(1)
        n_data = self.memory_s.size(0)
        
        # Check inputs for NaN
        if torch.isnan(feat_s).any():
            print(f"[ERROR] feat_s contains NaN!")
        if torch.isnan(feat_t).any():
            print(f"[ERROR] feat_t contains NaN!")
        if torch.isnan(self.memory_s).any():
            print(f"[ERROR] memory_s contains NaN!")
        if torch.isnan(self.memory_t).any():
            print(f"[ERROR] memory_t contains NaN!")
        
        # Retrieve features from memory bank
        weight_s = torch.index_select(self.memory_s, 0, sample_idx.view(-1)).detach()
        weight_s = weight_s.view(bs, self.K+1, feat_dim)
        out_t = torch.bmm(weight_s, feat_t.view(bs, feat_dim, 1))
        
        # Check before exp
        if torch.isnan(out_t).any():
            print(f"[ERROR] out_t contains NaN BEFORE exp! Range: [{out_t.min():.2f}, {out_t.max():.2f}]")
        
        out_t = torch.exp(torch.clamp(out_t / self.T, max=50)).squeeze().contiguous()
        
        # Check after exp
        if torch.isnan(out_t).any() or torch.isinf(out_t).any():
            print(f"[ERROR] out_t contains NaN/Inf AFTER exp!")
        
        weight_t = torch.index_select(self.memory_t, 0, sample_idx.view(-1)).detach()
        weight_t = weight_t.view(bs, self.K+1, feat_dim)
        out_s = torch.bmm(weight_t, feat_s.view(bs, feat_dim, 1))
        
        if torch.isnan(out_s).any():
            print(f"[ERROR] out_s contains NaN BEFORE exp! Range: [{out_s.min():.2f}, {out_s.max():.2f}]")
            
        out_s = torch.exp(torch.clamp(out_s / self.T, max=50)).squeeze().contiguous()
        
        if torch.isnan(out_s).any() or torch.isinf(out_s).any():
            print(f"[ERROR] out_s contains NaN/Inf AFTER exp!")
        
        # Set normalization constants
        if self.Z_t is None:
            self.Z_t = (out_t.mean() * n_data).detach().item()
            print(f"[CRD] Initialized Z_t = {self.Z_t:.2f}")
        if self.Z_s is None:
            self.Z_s = (out_s.mean() * n_data).detach().item()
            print(f"[CRD] Initialized Z_s = {self.Z_s:.2f}")
        
        # Safety check: prevent division by zero or very small numbers
        if self.Z_t < 1e-4 or self.Z_s < 1e-4:
            print(f"[WARNING] Z values too small! Z_t={self.Z_t:.6f}, Z_s={self.Z_s:.6f}")
            self.Z_t = max(self.Z_t, 1.0)
            self.Z_s = max(self.Z_s, 1.0)
        
        out_t = torch.div(out_t, self.Z_t)
        out_s = torch.div(out_s, self.Z_s)
        
        # Check for NaN after normalization
        if torch.isnan(out_t).any() or torch.isnan(out_s).any():
            print(f"[ERROR] NaN detected after normalization! Z_t={self.Z_t:.6f}, Z_s={self.Z_s:.6f}")
        
        # Update memory with momentum
        with torch.no_grad():
            pos_mem_t = torch.index_select(self.memory_t, 0, idx.view(-1))
            pos_mem_t.mul_(self.momentum)
            pos_mem_t.add_(torch.mul(feat_t, 1-self.momentum))
            pos_mem_t = F.normalize(pos_mem_t, p=2, dim=1)
            self.memory_t.index_copy_(0, idx, pos_mem_t)
            
            pos_mem_s = torch.index_select(self.memory_s, 0, idx.view(-1))
            pos_mem_s.mul_(self.momentum)
            pos_mem_s.add_(torch.mul(feat_s, 1-self.momentum))
            pos_mem_s = F.normalize(pos_mem_s, p=2, dim=1)
            self.memory_s.index_copy_(0, idx, pos_mem_s)
        
        return out_s, out_t

class ContrastLoss(nn.Module):
    """Contrastive loss from CRD paper"""
    def __init__(self, n_data, eps=1e-7):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data
        self.eps = eps
    
    def forward(self, x):
        bs = x.size(0)
        N = x.size(1) - 1
        M = float(self.n_data)
        
        # loss for positive pair
        pos_pair = x.select(1, 0)
        log_pos = torch.div(pos_pair, pos_pair.add(N / M + self.eps)).log_()
        
        # loss for negative pair
        neg_pair = x.narrow(1, 1, N)
        log_neg = torch.div(neg_pair.clone().fill_(N / M), neg_pair.add(N / M + self.eps)).log_()
        
        loss = -(log_pos.sum() + log_neg.sum()) / bs
        
        return loss

class CRDLoss(nn.Module):
    """Full CRD with memory bank"""
    def __init__(self, s_dim, t_dim, feat_dim=128, n_data=50000, nce_k=16384, nce_t=0.07, nce_m=0.5):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(s_dim, feat_dim)
        self.embed_t = Embed(t_dim, feat_dim)
        self.contrast = ContrastMemory(feat_dim, n_data, nce_k, nce_t, nce_m)
        self.criterion_s = ContrastLoss(n_data)
        self.criterion_t = ContrastLoss(n_data)
    
    def forward(self, feat_s, feat_t, idx, sample_idx):
        feat_s = self.embed_s(feat_s)
        feat_t = self.embed_t(feat_t)
        out_s, out_t = self.contrast(feat_s, feat_t, idx, sample_idx)
        loss_s = self.criterion_s(out_s)
        loss_t = self.criterion_t(out_t)
        return loss_s + loss_t

print("✅ CRD loss (in-batch) implementation defined.")

# %% [markdown]
# ### **2.4. Part 2 Training Configurations**
# 
# We define the configurations for HINTS and CRD. We first need to find the correct hint layers and their feature dimensions by doing a dummy forward pass.

# %%
# ============================================
# ⚙️ Part 2 Training Configurations
# ============================================

# Create directories for Part 2 results
RESULTS_DIR_P2 = Path("./results/task3_2")
MODEL_DIR_P2 = RESULTS_DIR_P2 / "models"
LOG_DIR_P2 = RESULTS_DIR_P2 / "logs"
FIGURE_DIR_P2 = RESULTS_DIR_P2 / "figures"
MODEL_DIR_P2.mkdir(parents=True, exist_ok=True)
LOG_DIR_P2.mkdir(parents=True, exist_ok=True)
FIGURE_DIR_P2.mkdir(parents=True, exist_ok=True)

# --- Define Hint Layers (Indices in `model.features` Sequential) ---
# VGG-16_bn features: 41 layers. [23] is ReLU after block 3's last conv (512 channels)
TEACHER_HINT_LAYER = 23 
# VGG-11_bn features: 23 layers. [15] is ReLU after block 3's last conv (512 channels)
STUDENT_HINT_LAYER = 15

# --- Get Feature Dimensions --- 
# (Ensuring teacher_model is loaded from Part 1.7)
if 'teacher_model' not in globals():
    print("WARNING: 'teacher_model' not found. Running Part 1.7 logic again.")
    # This logic is copied from Part 1.7 to ensure teacher_model is available
    TEACHER_MODEL_PATH = MODEL_DIR / 'teacher_vgg16_cifar100.pth'
    teacher_model, teacher_loaded = create_teacher_model(
        num_classes=num_classes,
        local_path=TEACHER_MODEL_PATH
    )
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    # Evaluate to get baseline accuracies if not already present
    if 'teacher_acc1' not in globals():
        _, teacher_acc1, teacher_acc5 = evaluate(teacher_model, testloader, use_amp=False)
        print(f"Teacher loaded with Acc@1: {teacher_acc1:.2f}%")

student_temp = create_student_model(num_classes).to(device)

teacher_wrapper = VGG_with_hints(teacher_model, TEACHER_HINT_LAYER).to(device)
student_wrapper_temp = VGG_with_hints(student_temp, STUDENT_HINT_LAYER).to(device)

teacher_channels_hint = -1
student_channels_hint = -1
s_dim_crd = -1
t_dim_crd = -1

# Get channel dimensions by passing dummy input
# Get channel dimensions by passing dummy input
with torch.no_grad():
    dummy = torch.randn(2, 3, 32, 32).to(device)
    
    # ✅ NEW: Get HINTS features (intermediate conv)
    teacher_wrapper_hints = VGG_with_hints(teacher_model, TEACHER_HINT_LAYER, use_penultimate=False)
    student_wrapper_hints = VGG_with_hints(student_temp, STUDENT_HINT_LAYER, use_penultimate=False)
    _, t_feat_hint = teacher_wrapper_hints(dummy, return_features=True)
    _, s_feat_hint = student_wrapper_hints(dummy, return_features=True)
    
    # For HINTS (Conv layer channels)
    teacher_channels_hint = t_feat_hint.shape[1]
    student_channels_hint = s_feat_hint.shape[1]
    
    # ✅ NEW: Get CRD features (penultimate)
    teacher_wrapper_crd = VGG_with_hints(teacher_model, TEACHER_HINT_LAYER, use_penultimate=True).to(device)
    student_wrapper_crd = VGG_with_hints(student_temp, STUDENT_HINT_LAYER, use_penultimate=True).to(device)
    _, t_feat_crd = teacher_wrapper_crd(dummy, return_features=True)
    _, s_feat_crd = student_wrapper_crd(dummy, return_features=True)
    
    # For CRD (Flattened penultimate feature dimensions)
    s_dim_crd = s_feat_crd.numel() // s_feat_crd.shape[0]
    t_dim_crd = t_feat_crd.numel() // t_feat_crd.shape[0]

    # ✅ ADD DEBUG PRINT
    print(f"\n📊 CRD Feature Dimensions:")
    print(f"   Student penultimate feature shape: {s_feat_crd.shape} -> dim = {s_dim_crd}")
    print(f"   Teacher penultimate feature shape: {t_feat_crd.shape} -> dim = {t_dim_crd}")

del student_temp, teacher_wrapper_hints, student_wrapper_hints, student_wrapper_crd

# Use the same base hyperparameters from Part 1
configs_p2 = {
    'HINTS': {
        'model_name': 'student_vgg11_HINTS',
        'loss_type': 'hints',
        'epochs': NUM_EPOCHS,
        'lr': BASE_LR,
        'momentum': MOMENTUM,
        'weight_decay': WEIGHT_DECAY,
        'lr_decay_epochs': LR_DECAY_EPOCHS,
        'lr_decay_rate': LR_DECAY_RATE,
        'loss_kwargs': {
            'T': 4.0,           # For the KD (soft logit) part of the loss
            'alpha_kd': 0.9,    # Weight for KD soft loss (from RepDistiller script)
            'gamma_ce': 1.0,    # Weight for Hard CE loss (from RepDistiller script)
            'beta_hint': 1.0,   # Weight for hint loss (from RepDistiller script)
            'teacher_wrapper': teacher_wrapper,
            'teacher_hint_layer': TEACHER_HINT_LAYER,
            'student_hint_layer': STUDENT_HINT_LAYER,
            'connector_channels': (student_channels_hint, teacher_channels_hint)
        }
    },
    "CRD": {
        "model_name": "student_vgg11_CRD",
        "loss_type": "crd",
        "lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4,
        "epochs": 200, "lr_decay_epochs": [100, 150], "lr_decay_rate": 0.1,
        "loss_kwargs": {
            "beta_crd": 0.8,
            "s_dim": s_dim_crd,   # ← use measured student penultimate dim
			"t_dim": t_dim_crd,   # ← use measured teacher penultimate dim
			"feat_dim": 128,
            "n_data": len(trainset),
            "nce_k": 16384,
            "nce_t": 0.07,
            "nce_m": 0.5,
            "teacher_wrapper": teacher_wrapper_crd,       # ✅ penultimate vectors for CRD
            "teacher_hint_layer": TEACHER_HINT_LAYER,     # kept for consistency
            "student_hint_layer": STUDENT_HINT_LAYER,     # kept for consistency
        },
    },
}

print(f"✅ Part 2 configurations ready.")
print(f"   Hint Layer (S): {STUDENT_HINT_LAYER} (Channels: {student_channels_hint}, Flattened Dim: {s_dim_crd})")
print(f"   Hint Layer (T): {TEACHER_HINT_LAYER} (Channels: {teacher_channels_hint}, Flattened Dim: {t_dim_crd})")

# %% [markdown]
# ### **2.5. Upgraded Training & Evaluation Loops**
# 
# We now create an **upgraded training function** (`train_epoch_master`) that can handle all loss types from Part 1 (CE, LS, KD, DKD) and Part 2 (HINTS, CRD). This keeps the code clean and allows the main training loop to be generic.
# 
# We also copy the `evaluate` function from Part 1 for clarity, renaming it `evaluate_master`.

# %%
# ============================================
# 🔁 Master Training & Evaluation Loops
# ============================================

# Re-define scaler just in case
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

def train_epoch_master(
    model, 
    teacher_model, 
    dataloader, 
    optimizer, 
    loss_type="ce", 
    epoch=0, 
    num_epochs=200, 
    use_amp=True, 
    **loss_kwargs
):
    """
    Master training function that handles all loss types from Part 1 and Part 2.
    
    Args:
        model: Student model
        teacher_model: Teacher model (can be None for CE and LS)
        dataloader: Training dataloader
        optimizer: Optimizer
        loss_type: Type of loss ('ce', 'ls', 'kd', 'dkd', 'hints', 'crd')
        epoch: Current epoch number
        num_epochs: Total number of epochs
        use_amp: Whether to use automatic mixed precision
        **loss_kwargs: Additional arguments for specific loss functions
    
    Returns:
        avg_loss, avg_acc_top1, avg_acc_top5
    """
    model.train()
    if teacher_model: 
        teacher_model.eval()

    # --- Get modules specific to Part 2 losses --- 
    connector = loss_kwargs.get("connector", None)
    crd_criterion = loss_kwargs.get("crd_criterion", None)
    if connector: 
        connector.train()
    if crd_criterion: 
        crd_criterion.train()
    
    teacher_wrapper = loss_kwargs.get("teacher_wrapper", None)
    if teacher_wrapper: 
        teacher_wrapper.eval()
    
    student_hint_layer = loss_kwargs.get("student_hint_layer", None)
    
    # Wrap student model ONLY if we need features (Hints or CRD)
    is_feature_loss = loss_type in ('hints', 'crd')
    if is_feature_loss:
        # ✅ FIXED: Use penultimate features for CRD, intermediate for HINTS
        use_penultimate = (loss_type == 'crd')
        student_wrapper = VGG_with_hints(model, student_hint_layer, use_penultimate=use_penultimate)
        student_wrapper.train()
    
    total_loss = 0
    total_top1 = 0
    total_top5 = 0
    num_samples = 0

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch+1}/{num_epochs} [Train {loss_type.upper()}]",
    )

    for batch_idx, batch_data in pbar:
        # Handle both regular batches (2 elements) and indexed batches (3 elements)
        if len(batch_data) == 3:
            inputs, targets, idx = batch_data
            idx = idx.to(device)
        else:
            inputs, targets = batch_data
            idx = None  # Will create dummy indices if needed for CRD
        
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # If CRD and no idx, create placeholder
        if loss_type == "crd" and idx is None:
            idx = torch.arange(batch_size).to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            student_logits = None
            student_features = None
            teacher_logits = None
            teacher_features = None
            loss = 0

            # --- Forward Passes ---
            if is_feature_loss:
                student_logits, student_features = student_wrapper(inputs, return_features=True)
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)
                    _, teacher_features = teacher_wrapper(inputs, return_features=True)
            else:
                # Standard forward pass for Part 1 losses
                student_logits = model(inputs)
                if loss_type in ('kd', 'dkd'):
                    with torch.no_grad():
                        teacher_logits = teacher_model(inputs)

            # --- Calculate Loss ---
            if loss_type == "ce":
                loss = F.cross_entropy(student_logits, targets)
            
            elif loss_type == "ls":
                loss = label_smoothing_loss(student_logits, targets, **loss_kwargs)
            
            elif loss_type == "kd":
                loss = kd_loss(student_logits, teacher_logits, targets, **loss_kwargs)
            
            elif loss_type == "dkd":
                warmup_factor = min((epoch+1) / 20, 1.0)  # Start epoch 0 -> factor 0.05
                loss_kwargs['warmup_factor'] = warmup_factor
                loss = dkd_loss(student_logits, teacher_logits, targets, **loss_kwargs)
            
            elif loss_type == "hints":
                # Hard CE loss
                loss_cls = F.cross_entropy(student_logits, targets)
                
                # Soft KD loss
                loss_div = F.kl_div(
                    F.log_softmax(student_logits / loss_kwargs.get("T", 4.0), dim=1),
                    F.softmax(teacher_logits.detach() / loss_kwargs.get("T", 4.0), dim=1),
                    reduction="batchmean",
                ) * (loss_kwargs.get("T", 4.0) ** 2)
                
                # Hint loss (feature matching)
                s_feat_connected = connector(student_features)
                
                # Check for NaN before computing hint loss
                if torch.isnan(s_feat_connected).any() or torch.isnan(teacher_features).any():
                    print(f"NaN detected in features at batch {batch_idx}")
                    loss = loss_cls  # Fallback to CE only
                else:
                    loss_hint_val = hint_loss(s_feat_connected, teacher_features)
                    
                    # Check if hint loss is too large
                    if torch.isnan(loss_hint_val) or loss_hint_val > 1e6:
                        print(f"Warning: Hint loss exploded: {loss_hint_val.item()}")
                        loss = loss_cls  # Fallback
                    else:
                        gamma_ce = loss_kwargs.get("gamma_ce", 1.0)
                        alpha_kd = loss_kwargs.get("alpha_kd", 0.9)
                        beta_hint = loss_kwargs.get("beta_hint", 1.0)
                        
                        loss = gamma_ce * loss_cls + alpha_kd * loss_div + beta_hint * loss_hint_val
            
            elif loss_type == "crd":
                # Hard CE loss
                loss_cls = F.cross_entropy(student_logits, targets)
                
                # ✅ FIXED: Add KD loss component (CRD+KD as in RepDistiller)
                T = loss_kwargs.get('T', 4.0)
                alpha_kd = loss_kwargs.get('alpha_kd', 1.0)
                loss_kd = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits.detach() / T, dim=1),
                    reduction="batchmean",
                ) * (T * T)
                
                # Sample negatives (K negatives per sample)
                n_data = loss_kwargs.get('n_data', 50000)
                nce_k = loss_kwargs.get('nce_k', 16384)  # ✅ FIXED: Now 16384 instead of 4096
                sample_idx = get_negative_samples(batch_size, n_data, nce_k, device)
                sample_idx[:, 0] = idx  # First index is the positive sample
                
                # CRD contrastive loss
                loss_crd_val = crd_criterion(student_features, teacher_features, idx, sample_idx)
                
                # Combine all losses
                alpha_ce = loss_kwargs.get("alpha_ce", 1.0)
                beta_crd = loss_kwargs.get("beta_crd", 0.8)
                
                # ✅ FIXED: Now includes KD component
                loss = alpha_ce * loss_cls + alpha_kd * loss_kd + beta_crd * loss_crd_val
            
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

        # Scaled backward pass for AMP
        scaler.scale(loss).backward()
        
        # Unscale gradients and clip to prevent NaN
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        if connector is not None:
            torch.nn.utils.clip_grad_norm_(connector.parameters(), max_norm=10.0)
        if crd_criterion is not None:
            torch.nn.utils.clip_grad_norm_(crd_criterion.parameters(), max_norm=10.0)
        
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_size
        accs = accuracy_topk(student_logits, targets, topk=(1, 5))
        total_top1 += accs["top1"] * batch_size
        total_top5 += accs["top5"] * batch_size
        num_samples += batch_size

        pbar.set_postfix(
            {
                "Loss": f"{total_loss/num_samples:.4f}",
                "Acc@1": f"{total_top1/num_samples:.2f}%",
                "Acc@5": f"{total_top5/num_samples:.2f}%",
            }
        )

    avg_loss = total_loss / num_samples
    avg_acc_top1 = total_top1 / num_samples
    avg_acc_top5 = total_top5 / num_samples
    return avg_loss, avg_acc_top1, avg_acc_top5

# We can reuse the `evaluate` function from Part 1, renamed for clarity
evaluate_master = evaluate

print("✅ Master train/eval loops (for all parts) defined.")

# %% [markdown]
# ### **2.6. Run Training for Part 2 Models**
# 
# Iterate through the `configs_p2` dictionary and train the HINTS and CRD models.

# %%
# ============================================
# 🔥 Run Part 2 Student Training
# ============================================

all_model_results_p2 = {}

# Check if all models are already trained
all_trained_p2 = all(
    (MODEL_DIR_P2 / f"{config['model_name']}_best.pth").exists()
    for config in configs_p2.values()
)

if all_trained_p2:
    print("\n===================== SKIPPING PART 2 TRAINING =====================")
    print(f"All Part 2 student model checkpoints found in: {MODEL_DIR_P2}")
    print("Proceeding to load results...")
    for model_key, config in configs_p2.items():
        results_path = LOG_DIR_P2 / f"{config['model_name']}_results.json"
        if results_path.exists():
            try:
                with open(results_path, "r") as f:
                    all_model_results_p2[model_key] = json.load(f)
            except Exception as e:
                print(f"   ⚠️ Error loading results for {model_key}: {e}")
        else:
            print(f"   ⚠️ Results file not found for {model_key}.")

else:
    print("\nStarting Part 2 student training process...")

    for model_key, config in configs_p2.items():
        model_save_path = MODEL_DIR_P2 / f"{config['model_name']}_best.pth"

        if model_save_path.exists():
            print(f"\n--- Skipping {model_key.upper()}: Checkpoint already exists. ---")
            results_path = LOG_DIR_P2 / f"{config['model_name']}_results.json"
            if results_path.exists():
                try:
                    with open(results_path, "r") as f:
                        all_model_results_p2[model_key] = json.load(f)
                except Exception as e:
                    print(f"   ⚠️ Error loading results for {model_key}: {e}")
            continue

        print(
            f"\n{'='*25} Training Student: {model_key.upper()} ({config['model_name']}) {'='*25}"
        )

        # --- Model & Module Initialization ---
        student_model = create_student_model(num_classes).to(device)
        current_teacher = teacher_model

        params_to_optimize = list(student_model.parameters())
        loss_kwargs = config["loss_kwargs"].copy()  # Copy to avoid modification

        if model_key == "HINTS":
            connector = Connector(*loss_kwargs["connector_channels"]).to(device)
            loss_kwargs["connector"] = connector
            params_to_optimize.extend(list(connector.parameters()))
            print(f"   Training with HINT loss (beta={loss_kwargs['beta_hint']})")

        elif model_key == "CRD":
            crd_criterion = CRDLoss(
                s_dim=loss_kwargs["s_dim"],
                t_dim=loss_kwargs["t_dim"],
                feat_dim=loss_kwargs["feat_dim"],
                n_data=loss_kwargs["n_data"],  # ✅ Add this
                nce_k=loss_kwargs["nce_k"],  # ✅ Add this
                nce_t=loss_kwargs["nce_t"],
                nce_m=loss_kwargs["nce_m"],  # ✅ Add this
            ).to(device)
            loss_kwargs["crd_criterion"] = crd_criterion
            params_to_optimize.extend(list(crd_criterion.parameters()))
            print(f"   Training with CRD loss (beta={loss_kwargs['beta_crd']})")

        # --- Optimizer & Scheduler ---
        optimizer = optim.SGD(
            params_to_optimize,
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            nesterov=True,
        )

        # --- Training Loop ---
        best_acc_top1 = 0.0
        best_acc_top5 = 0.0
        model_history = {
            "train_loss": [],
            "train_acc_top1": [],
            "train_acc_top5": [],
            "test_loss": [],
            "test_acc_top1": [],
            "test_acc_top5": [],
        }
        start_time = time.time()

        for epoch in range(config["epochs"]):
            current_lr = adjust_learning_rate_with_warmup(
                optimizer,
                epoch,
                WARMUP_EPOCHS,
                config["lr_decay_epochs"],
                config["lr_decay_rate"],
                config["lr"],
            )

            train_loss, train_acc1, train_acc5 = train_epoch_master(
                student_model,
                current_teacher,
                trainloader,
                optimizer,
                loss_type=config["loss_type"],
                epoch=epoch,
                num_epochs=config["epochs"],
                use_amp=USE_AMP,
                **loss_kwargs,
            )

            test_loss, test_acc1, test_acc5 = evaluate_master(
                student_model,
                testloader,
                epoch=epoch,
                num_epochs=config["epochs"],
                use_amp=USE_AMP,
            )

            # Store history
            model_history["train_loss"].append(train_loss)
            model_history["train_acc_top1"].append(train_acc1)
            model_history["train_acc_top5"].append(train_acc5)
            model_history["test_loss"].append(test_loss)
            model_history["test_acc_top1"].append(test_acc1)
            model_history["test_acc_top5"].append(test_acc5)

            if test_acc1 > best_acc_top1:
                print(
                    f"   ✨ New best Top-1: {test_acc1:.2f}% (Top-5: {test_acc5:.2f}%). Saving..."
                )
                best_acc_top1 = test_acc1
                best_acc_top5 = test_acc5
                state = {
                    "model_state_dict": student_model.state_dict(),
                    "accuracy_top1": best_acc_top1,
                    "accuracy_top5": best_acc_top5,
                    "epoch": epoch,
                }
                if model_key == "HINTS":
                    state["connector_state_dict"] = connector.state_dict()
                elif model_key == "CRD":
                    state["crd_state_dict"] = crd_criterion.state_dict()

                torch.save(state, model_save_path)

        training_time = time.time() - start_time
        print(
            f"\n🏁 Training finished for {model_key} in {training_time/60:.2f} minutes."
        )
        print(
            f"   Best Test Accuracy: Top-1={best_acc_top1:.2f}%, Top-5={best_acc_top5:.2f}%"
        )

        # --- Inference Time Measurement ---
        print("   Measuring inference time...")
        student_model.load_state_dict(torch.load(model_save_path)["model_state_dict"])
        student_model.eval()

        dummy_input = torch.randn(TEST_BATCH_SIZE, 3, 32, 32).to(device)
        repetitions = 100
        warmup = 10
        total_time_inf = 0
        with torch.no_grad():
            for i in range(repetitions + warmup):
                if device == "cuda":
                    torch.cuda.synchronize()
                start_inf = time.perf_counter()
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    _ = student_model(dummy_input)
                if device == "cuda":
                    torch.cuda.synchronize()
                end_inf = time.perf_counter()
                if i >= warmup:
                    total_time_inf += end_inf - start_inf
        avg_inference_time_ms = (total_time_inf / repetitions) * 1000
        print(
            f"   Average Inference Time per batch ({TEST_BATCH_SIZE} images): {avg_inference_time_ms:.2f} ms"
        )

        # --- Save final results ---
        serializable_config = {k: v for k, v in config.items() if k != "loss_kwargs"}
        serializable_history = {
            k: [float(item) for item in v] for k, v in model_history.items()
        }

        final_results = {
            "best_test_acc_top1": best_acc_top1,
            "best_test_acc_top5": best_acc_top5,
            "final_train_acc_top1": train_acc1,
            "final_train_acc_top5": train_acc5,
            "final_test_acc_top1": test_acc1,
            "final_test_acc_top5": test_acc5,
            "training_time_minutes": training_time / 60,
            "avg_inference_time_ms_per_batch": avg_inference_time_ms,
            "config": serializable_config,
            "history": serializable_history,
        }
        all_model_results_p2[model_key] = final_results

        results_save_path = LOG_DIR_P2 / f"{config['model_name']}_results.json"
        with open(results_save_path, "w") as f:
            json.dump(final_results, f, indent=4)
        print(f"   Results saved to {results_save_path}")

print("\n===================== PART 2 TRAINING COMPLETE =====================")

# %% [markdown]
# ### **2.7. Part 2 Results & Combined Summary**
# 
# Load Part 2 results (HINTS, CRD) and combine with Part 1 results for final comparison table and visualization. Only new Part 2 results are printed below; Part 1 results shown earlier.

# %%
# ============================================
# 📊 Combined Results Summary & Visualization
# ============================================

summary_data_combined = []
loaded_results_combined = {}

# Add Teacher
summary_data_combined.append({
    'Model': 'Teacher (VGG-16)',
    'Method': 'N/A',
    'Best Test Acc@1 (%)': f"{teacher_acc1:.2f}",
    'Best Test Acc@5 (%)': f"{teacher_acc5:.2f}",
    'Training Time (min)': 'N/A',
    'Inference (ms/batch)': 'N/A'
})

# Load Part 1 results (SI, LM, LS, DKD) from LOG_DIR
for model_key in ['SI', 'LM', 'LS', 'DKD']:
    config_p1 = configs.get(model_key)
    if not config_p1: continue
    results_path = LOG_DIR / f"{config_p1['model_name']}_results.json"
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
            loaded_results_combined[model_key] = results
            summary_data_combined.append({
                'Model': f"Student (VGG-11)",
                'Method': model_key,
                'Best Test Acc@1 (%)': f"{results['best_test_acc_top1']:.2f}",
                'Best Test Acc@5 (%)': f"{results['best_test_acc_top5']:.2f}",
                'Training Time (min)': f"{results['training_time_minutes']:.2f}",
                'Inference (ms/batch)': f"{results.get('avg_inference_time_ms_per_batch', 'N/A')}"
            })
    else:
         print(f"Warning: Part 1 results file not found for {model_key}")

# Load Part 2 results (HINTS, CRD) from LOG_DIR_P2
for model_key in ['HINTS', 'CRD']:
    config_p2 = configs_p2.get(model_key)
    if not config_p2: continue
    results_path = LOG_DIR_P2 / f"{config_p2['model_name']}_results.json"
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
            loaded_results_combined[model_key] = results
            summary_data_combined.append({
                'Model': f"Student (VGG-11)",
                'Method': model_key,
                'Best Test Acc@1 (%)': f"{results['best_test_acc_top1']:.2f}",
                'Best Test Acc@5 (%)': f"{results['best_test_acc_top5']:.2f}",
                'Training Time (min)': f"{results['training_time_minutes']:.2f}",
                'Inference (ms/batch)': f"{results.get('avg_inference_time_ms_per_batch', 'N/A')}"
            })
    else:
        print(f"Warning: Part 2 results file not found for {model_key}")

summary_df_combined = pd.DataFrame(summary_data_combined)

# Print only Part 2 results (avoid duplicating Part 1)
print("\n--- Task 3.2 Results Summary (Part 2 New Methods) ---")
summary_p2_only = summary_df_combined[summary_df_combined['Method'].isin(['HINTS', 'CRD'])]
if not summary_p2_only.empty:
    print(summary_p2_only.to_string(index=False))
else:
    print("No Part 2 results found yet. Train HINTS and CRD models first.")

print("\n--- All Methods Ranking by Top-1 Accuracy ---")
# Show ranking without printing full table again
students_only = summary_df_combined[summary_df_combined['Method'] != 'N/A'].copy()
students_only['Acc'] = students_only['Best Test Acc@1 (%)'].astype(float)
students_ranked = students_only.sort_values('Acc', ascending=False)
print(students_ranked[['Method', 'Best Test Acc@1 (%)']].to_string(index=False))

# ✅ FIX: Save combined summary table to Part 2 directory
combined_csv_path = LOG_DIR_P2 / 'task3_1_and_3_2_combined_summary.csv'
summary_df_combined.to_csv(combined_csv_path, index=False)
print(f"\n✅ Combined summary table saved to {combined_csv_path}")

# --- Plot Combined Training Curves ---
print("\nGenerating combined training curves plot...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(14, 12), dpi=150)

colors = sns.color_palette("husl", len(loaded_results_combined))

# Sort by best accuracy for cleaner legend
sorted_results = sorted(loaded_results_combined.items(), 
                        key=lambda item: item[1]['best_test_acc_top1'], 
                        reverse=True)

method_colors = {
    'CRD': '#e41a1c', 
    'HINTS': '#377eb8',
    'DKD': '#4daf4a',
    'LM': '#984ea3',
    'LS': '#ff7f00',
    'SI': '#a65628'
}

for i, (model_key, results) in enumerate(sorted_results):
    epochs_ran = len(results['history']['train_loss'])
    epoch_axis = range(1, epochs_ran + 1)
    color = method_colors.get(model_key, colors[i])
    
    # Plot Loss
    axes[0].plot(epoch_axis, results['history']['train_loss'], label=f'{model_key} Train', color=color, linewidth=2, alpha=0.8)
    axes[0].plot(epoch_axis, results['history']['test_loss'], label=f'{model_key} Test', linestyle='--', color=color, linewidth=2)

    # Plot Accuracy
    axes[1].plot(epoch_axis, results['history']['test_acc_top1'], label=f'{model_key} Test Acc@1', color=color, linewidth=2.5)

axes[0].set_title('Training & Test Loss Curves (All Methods)', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].legend(fontsize=10, loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].tick_params(axis='both', which='major', labelsize=10)
axes[0].set_ylim(bottom=0, top=10)

axes[1].set_title('Test Accuracy (Top-1) (All Methods)', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].legend(fontsize=10, loc='lower right')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].tick_params(axis='both', which='major', labelsize=10)
axes[1].set_ylim(bottom=0, top=75)

plt.suptitle('Task 3.1 & 3.2: KD Methods Comparison', fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()

# ✅ FIX: Save plot to Part 2 figures directory
plot_save_path = FIGURE_DIR_P2 / 'task3_1_and_3_2_combined_curves.png'
plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
print(f"✅ Combined training curves plot saved to {plot_save_path}")
plt.show()


