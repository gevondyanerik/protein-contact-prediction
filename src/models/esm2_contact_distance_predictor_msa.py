import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import math

class TransformerFusion(nn.Module):
    """
    A transformer-style fusion module that uses multihead self-attention to compute fusion weights.
    """
    def __init__(self, input_dim, d_model=128, num_heads=8):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.proj_out = nn.Linear(d_model, input_dim)
    
    def forward(self, x):
        proj_x = self.proj_in(x)
        attn_output, _ = self.mha(proj_x, proj_x, proj_x)
        fusion_weights = torch.sigmoid(self.proj_out(attn_output))
        return fusion_weights

class ESM2ContactDistancePredictorMSA(nn.Module):
    """
    A multitask model based on ESM2 that predicts:
      - A contact map (via a bilinear head and sigmoid)
      - A distance map (via a bilinear head and ReLU)
    
    If `use_msa` is True, the forward expects a tuple:
        (ref_tokens, msa_tokens)
    where:
        - ref_tokens: tokens for the reference sequence (shape: [B, L+2])
        - msa_tokens: tokens for the MSA (shape: [B, num_seqs, L+2])
    
    The MSA embeddings are averaged and then fused with the reference embeddings via concatenation
    (followed by a learned projection). If `feature_fusion` is False the fusion is done by simple averaging.
    """
    def __init__(self, esm_model_name='esm2_t6_8M_UR50D', msa_model_name='esm_msa1b_t12_100M_UR50S',
                 num_layers_to_freeze=None, feature_fusion=False, fusion_scale=1.0,
                 fusion_d_model=128, fusion_num_heads=8, use_msa=True):
        super().__init__()
        self.use_msa = use_msa
        self.feature_fusion = feature_fusion
        self.fusion_scale = fusion_scale
        self.fusion_d_model = fusion_d_model
        self.fusion_num_heads = fusion_num_heads

        # Load the reference (ESM-2) model.
        self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
        self.esm_model.train()
        if num_layers_to_freeze is not None:
            if hasattr(self.esm_model, "encoder") and hasattr(self.esm_model.encoder, "layers"):
                for i, layer in enumerate(self.esm_model.encoder.layers):
                    if i < num_layers_to_freeze:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        for param in layer.parameters():
                            param.requires_grad = True
            else:
                if num_layers_to_freeze > 0:
                    for param in self.esm_model.parameters():
                        param.requires_grad = False

        # Load the MSA model if requested.
        if self.use_msa:
            self.msa_model, self.msa_alphabet = esm.pretrained.load_model_and_alphabet(msa_model_name)
            self.msa_model.train()

        # Define heads.
        ref_dim = self.esm_model.embed_dim
        self.contact_head = nn.Bilinear(ref_dim, ref_dim, 1)
        self.distance_head = nn.Bilinear(ref_dim, ref_dim, 1)

        # Fusion: if feature_fusion is enabled, fuse embeddings via concatenation and projection.
        if self.feature_fusion:
            self.fusion_method = "concat"
            self.proj = nn.Linear(ref_dim * 2, ref_dim)
        else:
            self.fusion_method = None

    def forward(self, tokens):
        """
        Args:
            tokens: either a single tensor of shape (B, L+2) (if use_msa is False) or a tuple
                    (ref_tokens, msa_tokens) if use_msa is True.
        Returns:
            Tuple (contact_probs, distance_preds)
        """
        if self.use_msa and isinstance(tokens, tuple):
            ref_tokens, msa_tokens = tokens
            ref_out = self.esm_model(ref_tokens, repr_layers=[self.esm_model.num_layers])
            ref_emb = ref_out["representations"][self.esm_model.num_layers]  # (B, L+2, d)
            msa_out = self.msa_model(msa_tokens.view(-1, msa_tokens.shape[-1]), repr_layers=[self.msa_model.num_layers])
            msa_emb = msa_out["representations"][self.msa_model.num_layers]  # (B*num_seqs, L+2, d)
            msa_emb = msa_emb.view(msa_tokens.shape[0], msa_tokens.shape[1], msa_tokens.shape[2], -1)
            msa_emb = msa_emb.mean(dim=1)  # (B, L+2, d)
            if self.fusion_method == "concat":
                fused_emb = torch.cat([ref_emb, msa_emb], dim=-1)
                fused_emb = self.proj(fused_emb)
            else:
                fused_emb = (ref_emb + msa_emb) / 2
        else:
            ref_out = self.esm_model(tokens, repr_layers=[self.esm_model.num_layers])
            fused_emb = ref_out["representations"][self.esm_model.num_layers]
        
        B, Lp2, d = fused_emb.shape
        x_i = fused_emb.unsqueeze(2).expand(B, Lp2, Lp2, d)
        x_j = fused_emb.unsqueeze(1).expand(B, Lp2, Lp2, d)
        x_i_flat = x_i.reshape(B * Lp2 * Lp2, d)
        x_j_flat = x_j.reshape(B * Lp2 * Lp2, d)
        
        contact_logits = self.contact_head(x_i_flat, x_j_flat).view(B, Lp2, Lp2)
        distance_logits = self.distance_head(x_i_flat, x_j_flat).view(B, Lp2, Lp2)
        distance_preds = F.relu(distance_logits)
        
        contact_probs = torch.sigmoid(contact_logits)
        return contact_probs, distance_preds