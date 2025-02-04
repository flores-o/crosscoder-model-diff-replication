
from utils import *

from torch import nn
import pprint
import torch.nn.functional as F
from typing import Optional, Union
from huggingface_hub import hf_hub_download
import einops

from typing import NamedTuple

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
SAVE_DIR = Path("/workspace/crosscoder-model-diff-replication/checkpoints")

class LossOutput(NamedTuple):
    # loss: torch.Tensor
    l2_loss: torch.Tensor
    l1_loss: torch.Tensor
    l0_loss: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_A: torch.Tensor
    explained_variance_B: torch.Tensor

class CrossCoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = self.cfg["dict_size"]
        d_in = self.cfg["d_in"]
        self.dtype = DTYPES[self.cfg["enc_dtype"]]
        torch.manual_seed(self.cfg["seed"])
        # hardcoding n_models to 2
        self.W_enc = nn.Parameter(
            torch.empty(2, d_in, d_hidden, dtype=self.dtype)
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    d_hidden, 2, d_in, dtype=self.dtype
                )
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    d_hidden, 2, d_in, dtype=self.dtype
                )
            )
        )
        # Make norm of W_dec 0.1 for each column, separate per layer
        self.W_dec.data = (
            self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * self.cfg["dec_init_norm"]
        )
        # Initialise W_enc to be the transpose of W_dec
        self.W_enc.data = einops.rearrange(
            self.W_dec.data.clone(),
            "d_hidden n_models d_model -> n_models d_model d_hidden",
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=self.dtype))
        self.b_dec = nn.Parameter(
            torch.zeros((2, d_in), dtype=self.dtype)
        )
        self.d_hidden = d_hidden

        self.to(self.cfg["device"])
        self.save_dir = None
        self.save_version = 0

    def encode(self, x, apply_relu=True):
        # x: [batch, n_models, d_model]
        x_enc = einops.einsum(
            x,
            self.W_enc,
            "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
        )
        if apply_relu:
            acts = F.relu(x_enc + self.b_enc)
        else:
            acts = x_enc + self.b_enc
        return acts

    def decode(self, acts):
        # acts: [batch, d_hidden]
        acts_dec = einops.einsum(
            acts,
            self.W_dec,
            "batch d_hidden, d_hidden n_models d_model -> batch n_models d_model",
        )
        return acts_dec + self.b_dec

    def forward(self, x):
        # x: [batch, n_models, d_model]
        acts = self.encode(x)
        return self.decode(acts)

    def get_losses(self, x):
        # x: [batch, n_models, d_model]
        x = x.to(self.dtype)
        acts = self.encode(x)
        # acts: [batch, d_hidden]
        x_reconstruct = self.decode(acts)
        diff = x_reconstruct.float() - x.float()
        squared_diff = diff.pow(2)
        l2_per_batch = einops.reduce(squared_diff, 'batch n_models d_model -> batch', 'sum')
        l2_loss = l2_per_batch.mean()

        total_variance = einops.reduce((x - x.mean(0)).pow(2), 'batch n_models d_model -> batch', 'sum')
        explained_variance = 1 - l2_per_batch / total_variance

        per_token_l2_loss_A = (x_reconstruct[:, 0, :] - x[:, 0, :]).pow(2).sum(dim=-1).squeeze()
        total_variance_A = (x[:, 0, :] - x[:, 0, :].mean(0)).pow(2).sum(-1).squeeze()
        explained_variance_A = 1 - per_token_l2_loss_A / total_variance_A

        per_token_l2_loss_B = (x_reconstruct[:, 1, :] - x[:, 1, :]).pow(2).sum(dim=-1).squeeze()
        total_variance_B = (x[:, 1, :] - x[:, 1, :].mean(0)).pow(2).sum(-1).squeeze()
        explained_variance_B = 1 - per_token_l2_loss_B / total_variance_B

        decoder_norms = self.W_dec.norm(dim=-1)
        # decoder_norms: [d_hidden, n_models]
        total_decoder_norm = einops.reduce(decoder_norms, 'd_hidden n_models -> d_hidden', 'sum')
        l1_loss = (acts * total_decoder_norm[None, :]).sum(-1).mean(0)

        l0_loss = (acts>0).float().sum(-1).mean()

        return LossOutput(l2_loss=l2_loss, l1_loss=l1_loss, l0_loss=l0_loss, explained_variance=explained_variance, explained_variance_A=explained_variance_A, explained_variance_B=explained_variance_B)

    def create_save_dir(self):
        base_dir = Path("/workspace/crosscoder-model-diff-replication/checkpoints")
        version_list = [
            int(file.name.split("_")[1])
            for file in list(SAVE_DIR.iterdir())
            if "version" in str(file)
        ]
        if len(version_list):
            version = 1 + max(version_list)
        else:
            version = 0
        self.save_dir = base_dir / f"version_{version}"
        self.save_dir.mkdir(parents=True)

    def save(self):
        if self.save_dir is None:
            self.create_save_dir()
        weight_path = self.save_dir / f"{self.save_version}.pt"
        cfg_path = self.save_dir / f"{self.save_version}_cfg.json"

        torch.save(self.state_dict(), weight_path)
        with open(cfg_path, "w") as f:
            json.dump(self.cfg, f)

        print(f"Saved as version {self.save_version} in {self.save_dir}")
        self.save_version += 1

    @classmethod
    def load_from_hf(
        cls,
        repo_id: str = "ckkissane/crosscoder-gemma-2-2b-model-diff",
        path: str = "blocks.14.hook_resid_pre",
        device: Optional[Union[str, torch.device]] = None
    ) -> "CrossCoder":
        """
        Load CrossCoder weights and config from HuggingFace.
        
        Args:
            repo_id: HuggingFace repository ID
            path: Path within the repo to the weights/config
            model: The transformer model instance needed for initialization
            device: Device to load the model to (defaults to cfg device if not specified)
            
        Returns:
            Initialized CrossCoder instance
        """

        # Download config and weights
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{path}/cfg.json"
        )
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{path}/cc_weights.pt"
        )

        # Load config
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        # Override device if specified
        if device is not None:
            cfg["device"] = str(device)

        # Initialize CrossCoder with config
        instance = cls(cfg)

        # Load weights
        state_dict = torch.load(weights_path, map_location=cfg["device"])
        instance.load_state_dict(state_dict)

        return instance

    @classmethod
    def load(cls, version_dir, checkpoint_version):
        save_dir = Path("/workspace/crosscoder-model-diff-replication/checkpoints") / str(version_dir)
        cfg_path = save_dir / f"{str(checkpoint_version)}_cfg.json"
        weight_path = save_dir / f"{str(checkpoint_version)}.pt"

        cfg = json.load(open(cfg_path, "r"))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(weight_path))
        return self


class MatryoshkaCrossCoder(nn.Module):
    """
    A CrossCoder variant that implements Matryoshka training style.
    The dictionary is partitioned into groups as specified in cfg["group_sizes"].
    In the forward pass, partial reconstructions are computed and accumulated,
    and the final loss is the average MSE over all partial reconstructions.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.group_sizes = cfg["group_sizes"]
        total_dict_size = sum(self.group_sizes)
        # Create group indices for slicing: e.g., [0, group_sizes[0], group_sizes[0]+group_sizes[1], ...]
        self.group_indices = [0] + list(torch.cumsum(torch.tensor(self.group_sizes), dim=0))
 
        d_in = cfg["d_in"]
        self.dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
 
        # Define encoder and decoder with total dictionary size = sum(group_sizes)
        self.W_enc = nn.Parameter(
            torch.empty(2, d_in, total_dict_size, dtype=self.dtype)
        )
        self.W_dec = nn.Parameter(
            torch.empty(total_dict_size, 2, d_in, dtype=self.dtype)
        )
        self.b_enc = nn.Parameter(torch.zeros(total_dict_size, dtype=self.dtype))
        self.b_dec = nn.Parameter(torch.zeros((2, d_in), dtype=self.dtype))
 
        with torch.no_grad():
            # Initialize W_dec with a scaled normal distribution
            self.W_dec[:] = torch.randn_like(self.W_dec) * cfg.get("dec_init_norm", 0.08)
            self.W_dec[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True) * cfg.get("dec_init_norm", 0.08)
            # Set W_enc as the transpose (using einops rearrange)
            self.W_enc[:] = einops.rearrange(
                self.W_dec.clone(),
                "dict_size n_models d_in -> n_models d_in dict_size",
            )
 
        self.d_hidden = total_dict_size
        self.to(cfg["device"])
        self.save_dir = None
        self.save_version = 0
 
    def encode(self, x, apply_relu=True):
        """
        Encode input x.
        x: [batch, 2, d_in]
        Returns: encoded activations of shape [batch, total_dict_size]
        """
        x_enc = einops.einsum(
            x,
            self.W_enc,
            "batch n_models d_in, n_models d_in d_hidden -> batch d_hidden",
        )
        if apply_relu:
            acts = F.relu(x_enc + self.b_enc)
        else:
            acts = x_enc + self.b_enc
        return acts
 
    def forward(self, x):
        """
        Performs incremental decoding in a Matryoshka manner.
        Returns a list of partial reconstructions (each of shape [batch, 2, d_in]).
        Each successive reconstruction accumulates one more group.
        """
        acts = self.encode(x)  # [batch, total_dict_size]
        partial_recons = []
        # Initialize current reconstruction to zeros (optionally, you might add a bias here)
        current_recon = torch.zeros_like(x, dtype=acts.dtype, device=acts.device)
 
        for i in range(len(self.group_sizes)):
            start_idx = self.group_indices[i]
            end_idx = self.group_indices[i+1]
 
            # Extract the latent activations for the current group
            acts_slice = acts[:, start_idx:end_idx]  # [batch, group_size]
            # Extract the corresponding slice of decoder weights
            W_dec_slice = self.W_dec[start_idx:end_idx, :, :]  # [group_size, 2, d_in]
 
            # Compute partial reconstruction for this group
            recon_slice = einops.einsum(
                acts_slice, W_dec_slice,
                "batch g_size, g_size n_models d_in -> batch n_models d_in",
            )
            current_recon = current_recon + recon_slice
            partial_recons.append(current_recon.clone())
 
        return partial_recons
 
    def get_losses(self, x):
        x = x.to(self.dtype)
        partial_recons = self.forward(x)  # List of [batch, 2, d_in]
        loss_accum = 0.0
        for pr in partial_recons:
            loss_accum += F.mse_loss(pr, x)
        l2_loss = loss_accum / len(partial_recons)

        # Compute L1 loss similar to CrossCoder: multiply activations by the sum of decoder norms
        acts = self.encode(x)
        decoder_norms = self.W_dec.norm(dim=-1)  # shape: [total_dict_size, n_models]
        total_decoder_norm = einops.reduce(decoder_norms, 'd_hidden n_models -> d_hidden', 'sum')
        l1_loss = (acts * total_decoder_norm[None, :]).sum(-1).mean(0)

        # Compute L0 loss as the average number of nonzero activations per token
        l0_loss = (acts > 0).float().sum(-1).mean()

        # Optionally, compute explained variance (set to zero here)
        explained_variance = torch.tensor(0.0, device=x.device)
        explained_variance_A = torch.tensor(0.0, device=x.device)
        explained_variance_B = torch.tensor(0.0, device=x.device)

        return LossOutput(
            l2_loss=l2_loss,
            l1_loss=l1_loss,
            l0_loss=l0_loss,
            explained_variance=explained_variance,
            explained_variance_A=explained_variance_A,
            explained_variance_B=explained_variance_B,
        )

 
    def save(self):
        if self.save_dir is None:
            self.create_save_dir()
        weight_path = self.save_dir / f"{self.save_version}.pt"
        cfg_path = self.save_dir / f"{self.save_version}_cfg.json"
 
        torch.save(self.state_dict(), weight_path)
        with open(cfg_path, "w") as f:
            json.dump(self.cfg, f)
 
        print(f"Saved as version {self.save_version} in {self.save_dir}")
        self.save_version += 1
 
    def create_save_dir(self):
        from pathlib import Path
        base_dir = Path("/workspace/crosscoder-model-diff-replication/checkpoints")
        version_list = [
            int(file.name.split("_")[1])
            for file in list(base_dir.iterdir())
            if "version" in str(file)
        ]
        if len(version_list):
            version = 1 + max(version_list)
        else:
            version = 0
        self.save_dir = base_dir / f"version_{version}"
        self.save_dir.mkdir(parents=True)
