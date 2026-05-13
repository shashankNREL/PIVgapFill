"""
PIV Gap-Fill CVAE with 3-Phase Loss Scaling Schedule
=====================================================
Implements the strategy described in loss_scaling_plan.md:

  Phase 1 (0 – 20% of epochs):   beta = beta_warm (1e-4), gamma_base = 0
  Phase 2 (20% – 60% of epochs): beta cyclical annealing 0 → max_beta (n_cycles), gamma_base = 0
  Phase 3 (60% – 100% of epochs):beta = max_beta (frozen), gamma_base linear 0 → max_gamma

User-tunable parameters:
  --max_beta      (default 0.08; equivalent to old 5.0 with mean-KLD when latent_size=64)
  --max_gamma     (default 1e-6)
  --lambda_free   free-bits floor per latent dim in nats (default 0.5)
  --skip_noise_std  std of Gaussian noise injected into skip connections during training (default 0.05)
  --n_cycles      number of beta annealing cycles in Phase 2 (default 4)

KLD notes
---------
KLD is now computed as sum over latent dims, then mean over batch (previously it was
mean over both dims and batch, which divided by latent_size and masked the collapse).
This increases the raw KLD value by ~latent_size, so max_beta should be adjusted
accordingly (e.g. latent_size=64 with old max_beta=5 → new max_beta ≈ 0.08).

Raw magnitudes of BCE and VortMSE are printed every epoch to aid calibration.
"""

import os
import math
import datetime
import numpy as np
from numpy import linalg as LA
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import time
import argparse
from partialconv.models import partialconv2d
from sklearn.preprocessing import RobustScaler
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib
import resource
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class VariableHandler:
    def __init__(self, device=torch.device("cpu"), dtype=torch.float):
        self.device = device
        self.dtype = dtype

    def tovar(self, input):
        return Variable(torch.as_tensor(input, dtype=self.dtype,
                                        device=self.device))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CVAE(nn.Module):
    def __init__(self, latent_size, num_labels, ImgSizeX, ImgSizeY,
                 leaky_relu_slope=0.2, dropout_rate=0.2, skip_noise_std=0.05, vh=None):
        super().__init__()
        self.vh = vh if vh is not None else VariableHandler()

        self.latent_size = latent_size
        self.encoder = Encoder(latent_size, num_labels, ImgSizeX, ImgSizeY,
                               leaky_relu_slope, dropout_rate).to(device=self.vh.device, dtype=self.vh.dtype)
        self.decoder = Decoder(latent_size, num_labels,
                               leaky_relu_slope, dropout_rate,
                               skip_noise_std=skip_noise_std).to(device=self.vh.device, dtype=self.vh.dtype)

    def forward(self, x, c, mask):
        batch_size = x.size(0)
        means, log_var, skips = self.encoder(x, c, mask)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size], device=self.vh.device, dtype=self.vh.dtype)
        z = eps * std + means
        recon_x = self.decoder(z, c, skips)
        return recon_x, means, log_var, z

    def inference(self, x, c, mask):
        x = self.vh.tovar(x)
        mask = self.vh.tovar(mask)
        c = self.vh.tovar(c)
        means, _, skips = self.encoder(x, c, mask)
        decodedX = self.decoder(means, c, skips)
        return decodedX.cpu().detach().numpy()

    def load(self, fname):
        self.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
        self.eval()


class Encoder(nn.Module):
    def __init__(self, latent_size, nLabels, ImgSizeX, ImgSizeY, leaky_relu_slope=0.2, dropout_rate=0.2):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout_rate = dropout_rate

        self.EnCodeConv1 = partialconv2d.PartialConv2d(3, 16, 3, stride=2, padding=1, multi_channel=True, return_mask=True)
        self.EnCodeBN1 = nn.BatchNorm2d(16)

        self.EnCodeConv2 = partialconv2d.PartialConv2d(16, 32, 3, stride=1, padding=1, multi_channel=True, return_mask=True)
        self.EnCodeBN2 = nn.BatchNorm2d(32)

        self.EnCodeConv3 = partialconv2d.PartialConv2d(32, 64, 3, stride=2, padding=1, multi_channel=True, return_mask=True)
        self.EnCodeBN3 = nn.BatchNorm2d(64)

        self.EnCodeConv4 = partialconv2d.PartialConv2d(64, 96, 3, stride=1, padding=1, multi_channel=True, return_mask=True)
        self.EnCodeBN4 = nn.BatchNorm2d(96)

        self.EnCodeConv5 = partialconv2d.PartialConv2d(96, 128, 3, stride=2, padding=1, multi_channel=True, return_mask=True)
        self.EnCodeBN5 = nn.BatchNorm2d(128)

        self.EnCodeConv6 = partialconv2d.PartialConv2d(128, 192, 3, stride=2, padding=1, multi_channel=True, return_mask=True)
        self.EnCodeBN6 = nn.BatchNorm2d(192)

        self.EnCodeConv7 = partialconv2d.PartialConv2d(192, 256, 3, stride=1, padding=1, multi_channel=True, return_mask=True)
        self.EnCodeBN7 = nn.BatchNorm2d(256)

        self.inject_label = nn.Linear(256*5*3 + nLabels, latent_size)
        self.linear_means = nn.Linear(latent_size, latent_size)
        self.linear_log_var = nn.Linear(latent_size, latent_size)

    def forward(self, x, c, mask):
        x1, m1 = self.EnCodeConv1(x, mask)
        x1 = F.leaky_relu(self.EnCodeBN1(x1), negative_slope=self.leaky_relu_slope)

        x2, m2 = self.EnCodeConv2(x1, m1)
        x2 = F.leaky_relu(self.EnCodeBN2(x2), negative_slope=self.leaky_relu_slope)

        x3, m3 = self.EnCodeConv3(x2, m2)
        x3 = F.leaky_relu(self.EnCodeBN3(x3), negative_slope=self.leaky_relu_slope)

        x4, m4 = self.EnCodeConv4(x3, m3)
        x4 = F.leaky_relu(self.EnCodeBN4(x4), negative_slope=self.leaky_relu_slope)

        x5, m5 = self.EnCodeConv5(x4, m4)
        x5 = F.leaky_relu(self.EnCodeBN5(x5), negative_slope=self.leaky_relu_slope)

        x6, m6 = self.EnCodeConv6(x5, m5)
        x6 = F.leaky_relu(self.EnCodeBN6(x6), negative_slope=self.leaky_relu_slope)

        x7, m7 = self.EnCodeConv7(x6, m6)
        x7 = F.leaky_relu(self.EnCodeBN7(x7), negative_slope=self.leaky_relu_slope)

        x_flat = x7.view(x7.size(0), -1)
        x_c = torch.cat((x_flat, c), dim=1)

        hidden = F.leaky_relu(self.inject_label(x_c), negative_slope=self.leaky_relu_slope)
        hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)

        means = self.linear_means(hidden)
        log_vars = self.linear_log_var(hidden)

        skips = {
            'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
            'x5': x5, 'x6': x6, 'x7': x7
        }

        return means, log_vars, skips


class Decoder(nn.Module):
    def __init__(self, latent_size, num_labels, leaky_relu_slope=0.2, dropout_rate=0.2,
                 skip_noise_std=0.05):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout_rate = dropout_rate
        # Fix 3: small Gaussian noise injected into skip connections during training
        # to break the encoder short-circuit and force the latent space to be used.
        self.skip_noise_std = skip_noise_std
        self.LD1 = nn.Linear(latent_size + num_labels, 3840)

        self.dec_conv7 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.dec_bn7 = nn.BatchNorm2d(256)

        self.dec_conv6 = nn.Conv2d(448, 192, 3, stride=1, padding=1)
        self.dec_bn6 = nn.BatchNorm2d(192)

        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_conv5 = nn.Conv2d(320, 128, 3, stride=1, padding=1)
        self.dec_bn5 = nn.BatchNorm2d(128)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_conv4 = nn.Conv2d(224, 96, 3, stride=1, padding=1)
        self.dec_bn4 = nn.BatchNorm2d(96)

        self.dec_conv3 = nn.Conv2d(160, 64, 3, stride=1, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(64)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_conv2 = nn.Conv2d(96, 32, 3, stride=1, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(32)

        self.dec_conv1 = nn.Conv2d(48, 16, 3, stride=1, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(16)

        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_conv0 = nn.Conv2d(16, 3, 3, stride=1, padding=1)
        self.dec_bn0 = nn.BatchNorm2d(3)

    def _noisy_skip(self, s):
        """Add small Gaussian noise to a skip-connection tensor during training."""
        if self.training and self.skip_noise_std > 0.0:
            return s + torch.randn_like(s) * self.skip_noise_std
        return s

    def forward(self, z, c, skips):
        z_c = torch.cat((z, c), dim=-1)
        x = F.leaky_relu(self.LD1(z_c), negative_slope=self.leaky_relu_slope)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = x.view(-1, 256, 5, 3)

        x = torch.cat((x, self._noisy_skip(skips['x7'])), dim=1)
        x = F.leaky_relu(self.dec_bn7(self.dec_conv7(x)), negative_slope=self.leaky_relu_slope)

        x = torch.cat((x, self._noisy_skip(skips['x6'])), dim=1)
        x = F.leaky_relu(self.dec_bn6(self.dec_conv6(x)), negative_slope=self.leaky_relu_slope)

        x = self.up5(x)
        x = torch.cat((x, self._noisy_skip(skips['x5'])), dim=1)
        x = F.leaky_relu(self.dec_bn5(self.dec_conv5(x)), negative_slope=self.leaky_relu_slope)

        x = self.up4(x)
        x = torch.cat((x, self._noisy_skip(skips['x4'])), dim=1)
        x = F.leaky_relu(self.dec_bn4(self.dec_conv4(x)), negative_slope=self.leaky_relu_slope)

        x = torch.cat((x, self._noisy_skip(skips['x3'])), dim=1)
        x = F.leaky_relu(self.dec_bn3(self.dec_conv3(x)), negative_slope=self.leaky_relu_slope)

        x = self.up2(x)
        x = torch.cat((x, self._noisy_skip(skips['x2'])), dim=1)
        x = F.leaky_relu(self.dec_bn2(self.dec_conv2(x)), negative_slope=self.leaky_relu_slope)

        x = torch.cat((x, self._noisy_skip(skips['x1'])), dim=1)
        x = F.leaky_relu(self.dec_bn1(self.dec_conv1(x)), negative_slope=self.leaky_relu_slope)

        x = self.up0(x)
        x = torch.sigmoid(self.dec_bn0(self.dec_conv0(x)))

        return x


# ---------------------------------------------------------------------------
# Vorticity helper (unchanged)
# ---------------------------------------------------------------------------

def compute_vorticity_torch(img, tensor_X, tensor_Y):
    """
    Calculate vorticity in PyTorch using central differences.
    img: (N, 3, NX, NY) where channel 1=v, 0=u.
    """
    dv_dx = torch.gradient(img[:, 1, :, :], spacing=(tensor_X,), dim=1)[0]
    du_dy = torch.gradient(img[:, 0, :, :], spacing=(tensor_Y,), dim=2)[0]
    return dv_dx - du_dy


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def loss_fn(recon_x, x, mean, log_var, mask, batch_clean, tensor_X, tensor_Y,
            beta=0.001, gamma_base=0.0, lambda_free=0.5):
    BCE = torch.nn.functional.mse_loss(recon_x * mask, x * mask, reduction='mean')

    # Fix 2: sum over latent dims then mean over batch, preventing the collapse
    # from being hidden by averaging over the latent dimension.
    # Fix 1: free-bits floor — each latent dim must contribute at least lambda_free
    # nats to the KLD penalty. Dimensions already above the floor pass gradients
    # normally; collapsed dims (KLD < lambda_free) receive zero gradient until
    # they rise above the threshold.
    # mean, log_var: (B, latent_size)
    KLD_per_dim = -0.5 * (1.0 + log_var - mean.pow(2) - log_var.exp())  # (B, latent_size)
    KLD_per_dim = torch.clamp(KLD_per_dim, min=lambda_free)              # free-bits floor
    KLD = KLD_per_dim.sum(dim=1).mean()                                  # scalar

    coverage = mask[:, 0].mean()
    effective_gamma = gamma_base * (1.0 - coverage)

    v_recon = compute_vorticity_torch(recon_x, tensor_X, tensor_Y)
    v_clean = compute_vorticity_torch(batch_clean, tensor_X, tensor_Y)
    v_mask = mask[:, 0] * mask[:, 1]
    VortMSE = torch.mean(((v_recon - v_clean) * v_mask)**2)

    TotalLoss = BCE + beta * KLD + effective_gamma * VortMSE
    return TotalLoss, BCE, KLD, VortMSE, effective_gamma


# ---------------------------------------------------------------------------
# 3-Phase scheduling helpers
# ---------------------------------------------------------------------------

def _sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))


def compute_phased_coefficients(epoch, total_epochs, max_beta, max_gamma,
                                beta_warm=1e-4, sigmoid_k=10.0,
                                phase1_frac=0.20, phase2_frac=0.60,
                                n_cycles=4):
    """
    Return (current_beta, current_gamma_base) for the given epoch according
    to the 3-phase schedule described in loss_scaling_plan.md.

    Phase 1  [0,         phase1_frac*N):  beta = beta_warm, gamma = 0
    Phase 2  [phase1_frac*N, phase2_frac*N): beta cyclical annealing → max_beta, gamma = 0
    Phase 3  [phase2_frac*N, N):           beta = max_beta (frozen), gamma linear → max_gamma

    Parameters
    ----------
    epoch         : current epoch (0-indexed)
    total_epochs  : total number of training epochs
    max_beta      : ceiling value for beta (user input)
    max_gamma     : ceiling value for gamma_base (user input)
    beta_warm     : warm-start beta value used during Phase 1 (default 1e-4)
    sigmoid_k     : (unused, kept for backward-compat) steepness parameter
    phase1_frac   : fraction of total epochs defining end of Phase 1 (default 0.20)
    phase2_frac   : fraction of total epochs defining end of Phase 2 (default 0.60)
    n_cycles      : number of annealing cycles within Phase 2 (default 4).
                    Each cycle linearly ramps beta from 0 to max_beta in its first half,
                    then holds max_beta for its second half — allowing the network to
                    repeatedly escape posterior collapse before the final freeze.

    Returns
    -------
    current_beta       : float
    current_gamma_base : float
    phase_label        : str  (for logging)
    """
    phase1_end = int(phase1_frac * total_epochs)
    phase2_end = int(phase2_frac * total_epochs)

    if epoch < phase1_end:
        # ---- Phase 1: reconstruction only ----
        current_beta = beta_warm
        current_gamma_base = 0.0
        phase_label = "P1-Recon"

    elif epoch < phase2_end:
        # ---- Phase 2: cyclical beta annealing, gamma still 0 ----
        # Fix 4: divide Phase 2 into n_cycles equal cycles.  Within each cycle
        # beta linearly ramps from 0 to max_beta in the first half, then holds
        # max_beta for the second half.  This repeatedly restores the KLD
        # gradient signal and prevents persistent posterior collapse.
        t = (epoch - phase1_end) / max(1, phase2_end - phase1_end)  # [0, 1)
        # Use integer cycle index to avoid float modulo precision issues.
        cycle_idx = int(t * n_cycles)
        cycle_t = t * n_cycles - cycle_idx  # fractional position in current cycle [0, 1)
        if cycle_t < 0.5:
            current_beta = max_beta * (cycle_t / 0.5)   # linear ramp 0 → max_beta
        else:
            current_beta = max_beta                       # hold at ceiling
        current_gamma_base = 0.0
        phase_label = "P2-KLD"

    else:
        # ---- Phase 3: beta frozen, linear gamma ramp ----
        # Use (phase3_len - 1) as denominator so gamma reaches max_gamma
        # exactly on the final epoch (epoch = total_epochs - 1).
        phase3_len = total_epochs - phase2_end
        t = (epoch - phase2_end) / max(1, phase3_len - 1)
        current_beta = max_beta
        current_gamma_base = max_gamma * min(t, 1.0)
        phase_label = "P3-Vort"

    return current_beta, current_gamma_base, phase_label


# ---------------------------------------------------------------------------
# Data loading (identical to original)
# ---------------------------------------------------------------------------

def load_and_preprocess_data(file_path, n_samples, train_holes, n_valid):
    ImgFile = os.path.abspath(file_path)
    Data = np.load(ImgFile)
    Images = Data['ImgScale'][0:n_samples].copy()
    Mask = Data['Mask'][0:n_samples, 0:3].copy()

    Coverage = 1.0 - np.sum(Mask[:n_samples, 0], axis=(1, 2)) / (Mask.shape[2] * Mask.shape[3])
    ValidRows = np.argsort(-Coverage)[:n_valid]

    if train_holes == 1:
        CleanData = np.load(os.path.abspath(file_path.replace("Holes", "Clean")))
        CleanImg = CleanData['ImgScale'][0:n_samples]
        X = CleanData['X']
        Y = CleanData['Y']
        Max = CleanData['Max'][0:n_samples]
        Min = CleanData['Min'][0:n_samples]
    else:
        CleanImg = Images
        X = np.zeros_like(Images[:, 0, :, :])
        Y = np.zeros_like(Images[:, 0, :, :])
        Max = np.ones((n_samples, 3, 1, 1))
        Min = np.zeros((n_samples, 3, 1, 1))

    dtImg = 1.0 / float(n_samples)
    rawLabels = np.zeros((n_samples, 10))
    rawLabels[:, 0] = np.arange(1, n_samples + 1) * dtImg
    rawLabels[:, 1] = Min[:n_samples, 0, 0, 0] if 'Min' in locals() else 0
    rawLabels[:, 2] = Max[:n_samples, 0, 0, 0] if 'Max' in locals() else 0
    rawLabels[:, 3] = Min[:n_samples, 1, 0, 0] if 'Min' in locals() else 0
    rawLabels[:, 4] = Max[:n_samples, 1, 0, 0] if 'Max' in locals() else 0
    rawLabels[:, 5] = Min[:n_samples, 2, 0, 0] if 'Min' in locals() else 0
    rawLabels[:, 6] = Max[:n_samples, 2, 0, 0] if 'Max' in locals() else 0
    rawLabels[:, 7] = np.sum(CleanImg[:n_samples, 0], axis=(1, 2))
    rawLabels[:, 8] = np.sum(CleanImg[:n_samples, 1], axis=(1, 2))
    rawLabels[:, 9] = np.sum(CleanImg[:n_samples, 2], axis=(1, 2))

    transform = RobustScaler()
    clabels = transform.fit_transform(rawLabels)

    ValidationSet = Images[ValidRows]
    ValidationLabels = clabels[ValidRows]
    ValidationCoverage = Coverage[ValidRows]
    ValidMask = Mask[ValidRows]
    CleanValidation = CleanImg[ValidRows]

    return {
        'Images': Images, 'clabels': clabels, 'Mask': Mask, 'CleanImg': CleanImg,
        'ValidationSet': ValidationSet, 'ValidationLabels': ValidationLabels,
        'ValidationCoverage': ValidationCoverage, 'ValidMask': ValidMask,
        'CleanValidation': CleanValidation, 'transform': transform,
        'X': X, 'Y': Y, 'Max': Max, 'Min': Min, 'ValidRows': ValidRows
    }


# ---------------------------------------------------------------------------
# Error metric (unchanged)
# ---------------------------------------------------------------------------

def CalculateErrorImages(fname, ImageInput, ImagePredict, nImages, CleanImage=None, Mask=None, Coverage=None):
    if CleanImage is not None:
        magPred = np.sqrt(np.sum(np.square(ImagePredict[:nImages, :3]), axis=1))
        magClean = np.sqrt(np.sum(np.square(CleanImage[:nImages, :3]), axis=1))

        NormMag = np.abs(magClean - magPred) / np.maximum(magClean, 1e-5)
        MaxMagClean = np.max(magClean, axis=(1, 2))
        SumNormMag = np.sum(NormMag, axis=(1, 2))

        MaxError = np.max(SumNormMag / np.maximum(MaxMagClean, 1e-5))
        return MaxError
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Training loop with 3-phase loss scheduling
# ---------------------------------------------------------------------------

def train_cvae_phased(config, data_dict, device, outdir, is_ray_tune=False):
    """
    Training function with 3-phase loss coefficient scheduling.

    config keys (new / changed vs. original):
      max_beta          – ceiling for beta  (default 5.0)
      max_gamma         – ceiling for gamma_base (default 1e-6)
      beta_warm         – warm-start beta in Phase 1 (default 1e-4)
      phase1_frac       – fraction of epochs for Phase 1 (default 0.20)
      phase2_frac       – fraction of epochs marking end of Phase 2 (default 0.60)
      sigmoid_k         – (unused) kept for backward-compat (default 10.0)
      lambda_free       – free-bits floor per latent dim (default 0.5)
      skip_noise_std    – std of skip-connection noise in decoder (default 0.05)
      n_cycles          – number of beta annealing cycles in Phase 2 (default 4)
    """
    Images = data_dict['Images']
    clabels = data_dict['clabels']
    Mask = data_dict['Mask']
    CleanImg = data_dict['CleanImg']

    ValidationSet = data_dict['ValidationSet']
    ValidationLabels = data_dict['ValidationLabels']
    ValidationCoverage = data_dict['ValidationCoverage']
    ValidMask = data_dict['ValidMask']
    CleanValidation = data_dict['CleanValidation']

    vh = VariableHandler(device=device, dtype=torch.float)

    vae = CVAE(
        latent_size=config['latent_size'],
        num_labels=10,
        ImgSizeX=80,
        ImgSizeY=48,
        leaky_relu_slope=config['leaky_relu_slope'],
        dropout_rate=config['dropout_rate'],
        skip_noise_std=config.get('skip_noise_std', 0.05),
        vh=vh
    ).to(device=device)

    optimizer = optim.Adam(vae.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    tensor_Images = vh.tovar(Images)
    tensor_clabels = vh.tovar(clabels)
    tensor_Mask = vh.tovar(Mask)
    tensor_CleanImg = vh.tovar(CleanImg)

    if 'X' in data_dict and np.any(data_dict['X']):
        tensor_X = torch.as_tensor(data_dict['X'] * 1e-3, device=device, dtype=torch.float)
        tensor_Y = torch.as_tensor(data_dict['Y'] * 1e-3, device=device, dtype=torch.float)
    else:
        tensor_X = torch.linspace(0, 0.080, 80, device=device, dtype=torch.float)
        tensor_Y = torch.linspace(0, 0.048, 48, device=device, dtype=torch.float)

    outputFile = os.path.join(outdir, config['output_file'])
    outValidFile = os.path.join(outdir, "OutputImages_Validation.pdf")

    # Scheduling hyper-parameters
    max_beta = config['max_beta']
    max_gamma = config['max_gamma']
    beta_warm = config.get('beta_warm', 1e-4)
    phase1_frac = config.get('phase1_frac', 0.20)
    phase2_frac = config.get('phase2_frac', 0.60)
    sigmoid_k = config.get('sigmoid_k', 10.0)
    n_cycles = config.get('n_cycles', 4)
    lambda_free = config.get('lambda_free', 0.5)
    total_epochs = config['epochs']

    for epoch in range(total_epochs):
        vae.train()

        current_beta, current_gamma, phase_label = compute_phased_coefficients(
            epoch, total_epochs, max_beta, max_gamma,
            beta_warm=beta_warm, sigmoid_k=sigmoid_k,
            phase1_frac=phase1_frac, phase2_frac=phase2_frac,
            n_cycles=n_cycles
        )

        permutation = torch.randperm(Images.shape[0])
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_kld = 0.0
        epoch_vort = 0.0
        epoch_eff_gamma = 0.0
        n_batches = 0

        for i in range(0, Images.shape[0], config['batch_size']):
            indices = permutation[i:i + config['batch_size']]

            batch_x = tensor_Images[indices]
            batch_c = tensor_clabels[indices]
            batch_mask = tensor_Mask[indices]
            batch_clean = tensor_CleanImg[indices]

            recon_x, mean, log_var, z = vae(batch_x, batch_c, batch_mask)

            loss, bce, kld, vort, eff_gamma = loss_fn(
                recon_x, batch_x, mean, log_var,
                mask=batch_mask[:, 0:3],
                batch_clean=batch_clean,
                tensor_X=tensor_X, tensor_Y=tensor_Y,
                beta=current_beta, gamma_base=current_gamma,
                lambda_free=lambda_free
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bce += bce.item()
            epoch_kld += kld.item()
            epoch_vort += vort.item()
            epoch_eff_gamma += float(eff_gamma)
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        avg_bce = epoch_bce / n_batches
        avg_kld = epoch_kld / n_batches
        avg_vort = epoch_vort / n_batches
        avg_eff_gamma = epoch_eff_gamma / n_batches

        if not is_ray_tune:
            # Log raw loss component magnitudes every epoch to aid calibration
            print(
                f"Epoch [{epoch + 1:>5d}/{total_epochs}] [{phase_label}] "
                f"Loss: {avg_loss:.4e} | "
                f"BCE: {avg_bce:.4e} | "
                f"β·KLD: {current_beta * avg_kld:.4e} (β={current_beta:.4e}, KLD={avg_kld:.4e}) | "
                f"γ·Vort: {avg_eff_gamma * avg_vort:.4e} (γ_eff={avg_eff_gamma:.2e}, VortMSE={avg_vort:.4e}) | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

            if (epoch + 1) % 50 == 0:
                torch.save(vae.state_dict(), outputFile)
                vae.eval()
                PredictImage = vae.inference(ValidationSet, ValidationLabels, ValidMask)
                ErrMag = CalculateErrorImages(
                    outValidFile, ValidationSet, PredictImage, config['n_valid'],
                    CleanValidation, ValidMask, ValidationCoverage
                )
                print(
                    f"  >> Checkpoint: Val Err: {ErrMag:.4f}"
                )
                vae.train()
        else:
            vae.eval()
            PredictImage = vae.inference(ValidationSet, ValidationLabels, ValidMask)
            ErrMag = CalculateErrorImages(
                outValidFile, ValidationSet, PredictImage, config['n_valid'],
                CleanValidation, ValidMask, ValidationCoverage
            )

            from ray import tune as _tune
            _tune.report({
                "training_loss": avg_loss,
                "val_error": ErrMag,
                "bce": avg_bce,
                "kld": avg_kld,
                "vort_mse": avg_vort,
            })

    return vae


# ---------------------------------------------------------------------------
# Evaluation (unchanged from original)
# ---------------------------------------------------------------------------

def evaluate_cvae(vae, data_dict, args, outdir):
    Images = data_dict['Images']
    clabels = data_dict['clabels']
    Mask = data_dict['Mask']
    CleanImg = data_dict['CleanImg']
    X = data_dict['X']
    Y = data_dict['Y']
    Max = data_dict['Max']
    Min = data_dict['Min']

    PredictImg = np.zeros_like(CleanImg)
    cleanVort = np.zeros((CleanImg.shape[0], CleanImg.shape[2], CleanImg.shape[3]))
    PredictVort = np.zeros((CleanImg.shape[0], CleanImg.shape[2], CleanImg.shape[3]))
    VortError = np.zeros(CleanImg.shape[0])
    UxError = np.zeros(CleanImg.shape[0])
    UyError = np.zeros(CleanImg.shape[0])
    UzError = np.zeros(CleanImg.shape[0])

    start = time.time()
    for i in range(0, Images.shape[0], args.batch_size):
        bs = min(args.batch_size, Images.shape[0] - i)
        indices = slice(i, i + bs)
        PredictImg[indices] = vae.inference(Images[indices], clabels[indices], Mask[indices])

        CleanImg[indices] = CleanImg[indices] * (Max[indices] - Min[indices]) + Min[indices]
        PredictImg[indices] = PredictImg[indices] * (Max[indices] - Min[indices]) + Min[indices]

        for k in range(bs):
            idx = i + k
            cleanVort[idx] = np.gradient(CleanImg[idx, 1], X*1e-3, axis=0) - np.gradient(CleanImg[idx, 0], Y*1e-3, axis=1)
            PredictVort[idx] = np.gradient(PredictImg[idx, 1], X*1e-3, axis=0) - np.gradient(PredictImg[idx, 0], Y*1e-3, axis=1)

            VortError[idx] = LA.norm(cleanVort[idx] - PredictVort[idx], 'fro') / LA.norm(cleanVort[idx], 'fro')
            UxError[idx] = LA.norm(CleanImg[idx, 0] - PredictImg[idx, 0], 'fro') / LA.norm(CleanImg[idx, 0], 'fro')
            UyError[idx] = LA.norm(CleanImg[idx, 1] - PredictImg[idx, 1], 'fro') / LA.norm(CleanImg[idx, 1], 'fro')
            UzError[idx] = LA.norm(CleanImg[idx, 2] - PredictImg[idx, 2], 'fro') / LA.norm(CleanImg[idx, 2], 'fro')

    print(f"Maximum error in Ux {np.max(UxError):.4f}, Uy {np.max(UyError):.4f}, Uz {np.max(UzError):.4f}")
    print(f"Maximum error in vorticity {np.max(VortError):.4f}")

    end = time.time()
    print(f"Evaluation Time taken in hours {(end-start)/3600.0:.4f}")

    output_file = os.path.join(outdir, 'AllImagesResults_phased_loss.npz')
    np.savez(output_file, PredictImg=PredictImg, cleanVort=cleanVort, PredictVort=PredictVort,
             X=X, Y=Y, VortError=VortError, UxError=UxError, UyError=UyError, UzError=UzError)


# ---------------------------------------------------------------------------
# Ray Tune integration (adapted for new config keys)
# ---------------------------------------------------------------------------

def tune_hyperparameters_phased(data_dict, num_samples=10, max_num_epochs=100, outdir="./ray_results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "latent_size": tune.choice([64, 128]),
        "batch_size": tune.choice([32, 64]),
        "dropout_rate": tune.uniform(0.1, 0.3),
        "leaky_relu_slope": tune.uniform(0.1, 0.3),
        # Phase scheduling
        "max_beta": tune.loguniform(0.5, 10.0),
        "max_gamma": tune.loguniform(1e-7, 1e-5),
        "beta_warm": 1e-4,
        "phase1_frac": tune.uniform(0.10, 0.30),
        "phase2_frac": tune.uniform(0.50, 0.70),
        "sigmoid_k": tune.uniform(5.0, 15.0),
        "epochs": max_num_epochs,
        "n_valid": 5,
        "output_file": "tune_tmp_phased.pkl"
    }

    asha = ASHAScheduler(
        metric="val_error",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                train_cvae_phased, data_dict=data_dict,
                device=device, outdir=outdir, is_ray_tune=True
            ),
            resources={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0}
        ),
        tune_config=tune.TuneConfig(scheduler=asha, num_samples=num_samples),
        param_space=search_space,
    )
    results = tuner.fit()

    best_result = results.get_best_result("val_error", "min")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation error: {}".format(best_result.metrics.get("val_error")))
    return best_result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CVAE PIV gap-fill with 3-phase loss scaling (see loss_scaling_plan.md)'
    )
    parser.add_argument('-f', '--file', dest='file', type=str, default='ResizePIV.npz')
    parser.add_argument('-b', '--batchSize', dest='batch_size', type=int, default=64)
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=1000)
    parser.add_argument('-ns', '--nSamples', dest='nSamples', type=int, default=8000)
    parser.add_argument('-l', '--nl', dest='latent_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default=1e-4)
    parser.add_argument('-v', '--nValidation', dest='nValid', type=int, default=5)
    parser.add_argument('-H', '--withHoles', dest='TrainHoles', type=int, default=1)
    parser.add_argument('-o', '--output', dest='oFile', type=str, default='CVAEoutput_phased.pkl')
    # 3-phase loss scaling parameters
    parser.add_argument('--max_beta', type=float, default=0.08,
                        help='Maximum beta (KLD weight). With the new sum-KLD formulation the raw '
                             'KLD value is ~latent_size× larger than with the old mean-KLD, so '
                             'max_beta should be divided by latent_size relative to the old value '
                             '(e.g. old max_beta=5, latent_size=64 → new max_beta≈0.08). (default: 0.08)')
    parser.add_argument('--max_gamma', type=float, default=1e-6,
                        help='Maximum gamma_base (vorticity loss weight) (default: 1e-6)')
    parser.add_argument('--beta_warm', type=float, default=1e-4,
                        help='Warm-start beta used in Phase 1 (default: 1e-4)')
    parser.add_argument('--phase1_frac', type=float, default=0.20,
                        help='Fraction of epochs for Phase 1 – reconstruction only (default: 0.20)')
    parser.add_argument('--phase2_frac', type=float, default=0.60,
                        help='Fraction of epochs marking end of Phase 2 – KLD ramp (default: 0.60)')
    parser.add_argument('--sigmoid_k', type=float, default=10.0,
                        help='(unused) Kept for backward compatibility (default: 10.0)')
    parser.add_argument('--lambda_free', type=float, default=0.5,
                        help='Free-bits floor per latent dim in nats; prevents posterior collapse '
                             'by blocking gradients when a dim KLD is below this threshold (default: 0.5)')
    parser.add_argument('--skip_noise_std', type=float, default=0.05,
                        help='Std of Gaussian noise added to each encoder skip connection during '
                             'training; forces the decoder to rely on the latent code (default: 0.05)')
    parser.add_argument('--n_cycles', type=int, default=4,
                        help='Number of beta annealing cycles in Phase 2; each cycle ramps beta '
                             'from 0 to max_beta and holds, letting the network escape collapse '
                             'repeatedly (default: 4)')
    # Ray Tune
    parser.add_argument('--tune', action='store_true', help='Run Ray Tune hyperparameter search')
    parser.add_argument('--tune_samples', type=int, default=10, help='Number of Ray Tune trials')
    args = parser.parse_args()

    outdir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%m-%d_%H-%M') + '_phased')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading and preprocessing data...")
    data_dict = load_and_preprocess_data(args.file, args.nSamples, args.TrainHoles, args.nValid)

    scalerFile = os.path.join(outdir, "LabelScaler.pkl")
    joblib.dump(data_dict['transform'], scalerFile)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.tune:
        print(f"Starting Ray Tune. Trials: {args.tune_samples}, max_epochs: {args.epochs}")
        tune_hyperparameters_phased(data_dict, num_samples=args.tune_samples,
                                    max_num_epochs=args.epochs, outdir=outdir)
        print("Done tuning.")
    else:
        config = {
            'latent_size': args.latent_size,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'leaky_relu_slope': 0.2,
            'dropout_rate': 0.2,
            # 3-phase scheduling
            'max_beta': args.max_beta,
            'max_gamma': args.max_gamma,
            'beta_warm': args.beta_warm,
            'phase1_frac': args.phase1_frac,
            'phase2_frac': args.phase2_frac,
            'sigmoid_k': args.sigmoid_k,
            # KLD fixes
            'lambda_free': args.lambda_free,
            'skip_noise_std': args.skip_noise_std,
            'n_cycles': args.n_cycles,
            'n_valid': args.nValid,
            'output_file': args.oFile,
        }

        print("3-phase loss scheduling configuration:")
        print(f"  Phase 1 (0 – {args.phase1_frac*100:.0f}%): beta_warm={args.beta_warm:.1e}, gamma=0")
        print(f"  Phase 2 ({args.phase1_frac*100:.0f}% – {args.phase2_frac*100:.0f}%): "
              f"beta cyclical 0→{args.max_beta} ({args.n_cycles} cycles), gamma=0")
        print(f"  Phase 3 ({args.phase2_frac*100:.0f}% – 100%): beta={args.max_beta} (frozen), "
              f"gamma linear 0→{args.max_gamma:.1e}")
        print(f"  KLD fixes: lambda_free={args.lambda_free}, skip_noise_std={args.skip_noise_std}")
        print("Starting training...")

        start = time.time()
        vae = train_cvae_phased(config, data_dict, device, outdir, is_ray_tune=False)
        end = time.time()
        print(f"Training Time taken in hours {(end-start)/3600.0:.4f}")

        print("Starting evaluation...")
        evaluate_cvae(vae, data_dict, args, outdir)
