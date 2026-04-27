import os
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

class VariableHandler:
    def __init__(self, device=torch.device("cpu"), dtype=torch.float):
        self.device = device
        self.dtype = dtype

    def tovar(self, input):
        return Variable(torch.as_tensor(input, dtype=self.dtype,
                                        device=self.device))


class CVAE(nn.Module):
    def __init__(self, latent_size, num_labels, ImgSizeX, ImgSizeY, 
                 leaky_relu_slope=0.2, dropout_rate=0.2, vh=None):
        super().__init__()
        self.vh = vh if vh is not None else VariableHandler()

        self.latent_size = latent_size
        self.encoder = Encoder(latent_size, num_labels, ImgSizeX, ImgSizeY,
                               leaky_relu_slope, dropout_rate).to(device=self.vh.device, dtype=self.vh.dtype)
        self.decoder = Decoder(latent_size, num_labels,
                               leaky_relu_slope, dropout_rate).to(device=self.vh.device, dtype=self.vh.dtype)

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
    def __init__(self, latent_size, num_labels, leaky_relu_slope=0.2, dropout_rate=0.2):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout_rate = dropout_rate
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

    def forward(self, z, c, skips):
        z_c = torch.cat((z, c), dim=-1)
        x = F.leaky_relu(self.LD1(z_c), negative_slope=self.leaky_relu_slope)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = x.view(-1, 256, 5, 3)

        x = torch.cat((x, skips['x7']), dim=1)
        x = F.leaky_relu(self.dec_bn7(self.dec_conv7(x)), negative_slope=self.leaky_relu_slope)

        x = torch.cat((x, skips['x6']), dim=1)
        x = F.leaky_relu(self.dec_bn6(self.dec_conv6(x)), negative_slope=self.leaky_relu_slope)

        x = self.up5(x)
        x = torch.cat((x, skips['x5']), dim=1)
        x = F.leaky_relu(self.dec_bn5(self.dec_conv5(x)), negative_slope=self.leaky_relu_slope)

        x = self.up4(x)
        x = torch.cat((x, skips['x4']), dim=1)
        x = F.leaky_relu(self.dec_bn4(self.dec_conv4(x)), negative_slope=self.leaky_relu_slope)

        x = torch.cat((x, skips['x3']), dim=1)
        x = F.leaky_relu(self.dec_bn3(self.dec_conv3(x)), negative_slope=self.leaky_relu_slope)

        x = self.up2(x)
        x = torch.cat((x, skips['x2']), dim=1)
        x = F.leaky_relu(self.dec_bn2(self.dec_conv2(x)), negative_slope=self.leaky_relu_slope)

        x = torch.cat((x, skips['x1']), dim=1)
        x = F.leaky_relu(self.dec_bn1(self.dec_conv1(x)), negative_slope=self.leaky_relu_slope)

        x = self.up0(x)
        x = torch.sigmoid(self.dec_bn0(self.dec_conv0(x)))

        return x


def compute_vorticity_torch(img, tensor_X, tensor_Y):
    """
    Calculate vorticity in PyTorch using central differences.
    Matches the logic: omega = dv/dx - du/dy
    img: (N, 3, NX, NY) where channel 1=v, 0=u. 
    tensor_X corresponds to dim 2 (NX), tensor_Y to dim 3 (NY).
    """
    # dv_dx = partial v / partial x
    dv_dx = torch.gradient(img[:, 1, :, :], spacing=(tensor_X,), dim=1)[0]
    # du_dy = partial u / partial y
    du_dy = torch.gradient(img[:, 0, :, :], spacing=(tensor_Y,), dim=2)[0]
    return dv_dx - du_dy


def loss_fn(recon_x, x, mean, log_var, mask, batch_clean, tensor_X, tensor_Y, beta=0.001, gamma_base=0.0):
    # Velocity MSE (on gappy pixels)
    BCE = torch.nn.functional.mse_loss(recon_x * mask, x * mask, reduction='mean')
    
    # Beta-VAE KL Divergence
    KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    
    # Coverage Adaptive Gamma
    # Mask is 1 where we have data, 0 where there are holes. 
    # Coverage = number of 1s / total number of pixels in one channel
    coverage = mask[:, 0].mean() # mean over all batches and spatial dims
    effective_gamma = gamma_base * (1.0 - coverage)
    
    # Vorticity MSE on unmasked regions for physics consistency
    v_recon = compute_vorticity_torch(recon_x, tensor_X, tensor_Y)
    v_clean = compute_vorticity_torch(batch_clean, tensor_X, tensor_Y)
    v_mask = mask[:, 0] * mask[:, 1]
    VortMSE = torch.mean(((v_recon - v_clean) * v_mask)**2)
    
    TotalLoss = BCE + beta * KLD + effective_gamma * VortMSE
    return TotalLoss, BCE, KLD, VortMSE, effective_gamma


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


def load_and_preprocess_data(file_path, n_samples, train_holes, n_valid):
    """
    Loads data and creates variables needed for training.
    """
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


def train_cvae(config, data_dict, device, outdir, is_ray_tune=False):
    """
    Main training function. Designed to be run standard or via Ray Tune.
    Expects data_dict loaded by load_and_preprocess_data.
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
        vh=vh
    ).to(device=device)

    optimizer = optim.Adam(vae.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    tensor_Images = vh.tovar(Images)
    tensor_clabels = vh.tovar(clabels)
    tensor_Mask = vh.tovar(Mask)
    tensor_CleanImg = vh.tovar(CleanImg)

    # Prepare coordinates for Vorticity calculation
    if 'X' in data_dict and np.any(data_dict['X']):
        tensor_X = torch.as_tensor(data_dict['X'] * 1e-3, device=device, dtype=torch.float)
        tensor_Y = torch.as_tensor(data_dict['Y'] * 1e-3, device=device, dtype=torch.float)
    else:
        tensor_X = torch.linspace(0, 0.080, 80, device=device, dtype=torch.float)
        tensor_Y = torch.linspace(0, 0.048, 48, device=device, dtype=torch.float)

    outputFile = os.path.join(outdir, config['output_file'])
    outValidFile = os.path.join(outdir, "OutputImages_Validation.pdf")

    for epoch in range(config['epochs']):
        vae.train()
        
        # Annealing schedules
        beta_anneal_epochs = config.get('beta_anneal_epochs', max(1, config['epochs'] * 0.5))
        gamma_anneal_epochs = config.get('gamma_anneal_epochs', max(1, config['epochs'] * 0.5))
        
        current_beta = min(config['max_beta'], config['max_beta'] * (epoch / beta_anneal_epochs)) if beta_anneal_epochs > 0 else config['max_beta']
        current_gamma = min(config['max_gamma'], config['max_gamma'] * (epoch / gamma_anneal_epochs)) if gamma_anneal_epochs > 0 else config['max_gamma']
        
        permutation = torch.randperm(Images.shape[0])
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, Images.shape[0], config['batch_size']):
            indices = permutation[i:i+config['batch_size']]

            batch_x = tensor_Images[indices]
            batch_c = tensor_clabels[indices]
            batch_mask = tensor_Mask[indices]
            batch_clean = tensor_CleanImg[indices]
            
            recon_x, mean, log_var, z = vae(batch_x, batch_c, batch_mask)
            
            loss, bce, kld, vort, eff_gamma = loss_fn(recon_x, batch_x, mean, log_var, 
                                                      mask=batch_mask[:, 0:3], 
                                                      batch_clean=batch_clean,
                                                      tensor_X=tensor_X, tensor_Y=tensor_Y,
                                                      beta=current_beta, gamma_base=current_gamma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if not is_ray_tune:
            if (epoch+1) % 50 == 0:
                torch.save(vae.state_dict(), outputFile)
                vae.eval()
                PredictImage = vae.inference(ValidationSet, ValidationLabels, ValidMask)
                ErrMag = CalculateErrorImages(outValidFile, ValidationSet, PredictImage, config['n_valid'], CleanValidation, ValidMask, ValidationCoverage)
                print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {avg_loss:.4e}, Val Err: {ErrMag:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}, Beta: {current_beta:.4f}, eff_Gamma: {float(eff_gamma):.1e}")
        else:
            # Ray tune evaluation
            vae.eval()
            PredictImage = vae.inference(ValidationSet, ValidationLabels, ValidMask)
            ErrMag = CalculateErrorImages(outValidFile, ValidationSet, PredictImage, config['n_valid'], CleanValidation, ValidMask, ValidationCoverage)
            
            from ray import tune
            tune.report({
                "training_loss": avg_loss,
                "val_error": ErrMag
            })

    return vae


def evaluate_cvae(vae, data_dict, args, outdir):
    """
    Evaluates completely on CPU (numpy arrays) and calculates statistics.
    """
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

    output_file = os.path.join(outdir, 'AllImagesResults_D2F3_unet.npz')
    np.savez(output_file, PredictImg=PredictImg, cleanVort=cleanVort, PredictVort=PredictVort, 
             X=X, Y=Y, VortError=VortError, UxError=UxError, UyError=UyError, UzError=UzError)


def tune_hyperparameters(data_dict, num_samples=10, max_num_epochs=100, outdir="./ray_results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define search space
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "latent_size": tune.choice([64, 128]),
        "batch_size": tune.choice([32, 64]),
        "dropout_rate": tune.uniform(0.1, 0.3),
        "leaky_relu_slope": tune.uniform(0.1, 0.3),
        "max_beta": tune.loguniform(1e-4, 1.0),
        "beta_anneal_epochs": tune.randint(50, max_num_epochs),
        "max_gamma": tune.loguniform(1e-9, 1e-4),
        "gamma_anneal_epochs": tune.randint(50, max_num_epochs),
        "epochs": max_num_epochs,
        "n_valid": 5,
        "output_file": "tune_tmp.pkl"
    }

    scheduler = ASHAScheduler(
        metric="val_error",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_cvae, data_dict=data_dict, device=device, outdir=outdir, is_ray_tune=True),
            resources={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("val_error", "min")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation error: {}".format(best_result.metrics.get("val_error")))
    return best_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CVAE to fill gaps in PIV data using U-Net architecture / RayTune')
    parser.add_argument('-f', '--file', dest='file', type=str, default='ResizePIV.npz')
    parser.add_argument('-b', '--batchSize', dest='batch_size', type=int, default=64)
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=1000)
    parser.add_argument('-ns', '--nSamples', dest='nSamples', type=int, default=8000)
    parser.add_argument('-l', '--nl', dest='latent_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default=1e-4)
    parser.add_argument('-v', '--nValidation', dest='nValid', type=int, default=5)
    parser.add_argument('-H', '--withHoles', dest='TrainHoles', type=int, default=1)
    parser.add_argument('-o', '--output', dest='oFile', type=str, default='CVAEoutput_unet.pkl')
    parser.add_argument('--tune', action='store_true', help='Run Ray Tune hyperparameter search')
    parser.add_argument('--tune_samples', type=int, default=10, help='Number of Ray Tune trials')
    args = parser.parse_args()

    outdir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%m-%d_%H-%M') + '_unet')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading and preprocessing data...")
    data_dict = load_and_preprocess_data(args.file, args.nSamples, args.TrainHoles, args.nValid)
    
    scalerFile = os.path.join(outdir, "LabelScaler.pkl")
    joblib.dump(data_dict['transform'], scalerFile)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.tune:
        print(f"Starting Ray Tune. Trials: {args.tune_samples}, max_epochs: {args.epochs}")
        best_result = tune_hyperparameters(data_dict, num_samples=args.tune_samples, max_num_epochs=args.epochs, outdir=outdir)
        print("Done tuning.")
    else:
        print("Starting standard training...")
        config = {
            'latent_size': args.latent_size,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'leaky_relu_slope': 0.2, # default
            'dropout_rate': 0.2,     # default
            'max_beta': 20.0,        # default matching d2f1_agy_unet
            'beta_anneal_epochs': 100,
            'max_gamma': 1e-7,       # default matching d2f1_agy_unet
            'gamma_anneal_epochs': 100,
            'n_valid': args.nValid,
            'output_file': args.oFile
        }
        
        start = time.time()
        vae = train_cvae(config, data_dict, device, outdir, is_ray_tune=False)
        end = time.time()
        print(f"Training Time taken in hours {(end-start)/3600.0:.4f}")

        print("Starting evaluation...")
        evaluate_cvae(vae, data_dict, args, outdir)
