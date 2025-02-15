import torch
import torch.nn as nn
import torch.nn.functional as F
import esm

class ESM2ContactPredictorMSA(nn.Module):
    """
    A contact prediction model that fuses embeddings from:
      1. A pretrained ESM-2 model (reference sequence branch)
      2. A pretrained MSA Transformer (MSA branch)
    
    The fusion is done by concatenating the embeddings from both branches and projecting
    back to the reference dimension before computing pairwise contact scores.
    
    It assumes that for each sample you supply:
      - ref_tokens: tensor for the reference sequence (shape: (B, L+2))
      - msa_tokens: tensor for the MSA (shape: (B, num_seqs, L+2))
    """
    def __init__(self, esm_model_name='esm2_t6_8M_UR50D', msa_model_name='esm_msa1b_t12_100M_UR50S', 
                 num_layers_to_freeze=None, fusion_method='concat'):
        super().__init__()
        # Load the reference (ESM-2) model and its alphabet.
        self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
        self.esm_model.train()
        # Optionally freeze layers of the reference model.
        if num_layers_to_freeze is not None:
            if hasattr(self.esm_model, "encoder") and hasattr(self.esm_model.encoder, "layers"):
                for i, layer in enumerate(self.esm_model.encoder.layers):
                    if i < num_layers_to_freeze:
                        for param in layer.parameters():
                            param.requires_grad = False
            else:
                if num_layers_to_freeze > 0:
                    for param in self.esm_model.parameters():
                        param.requires_grad = False
        
        # Load the MSA Transformer model and its alphabet.
        self.msa_model, self.msa_alphabet = esm.pretrained.load_model_and_alphabet(msa_model_name)
        self.msa_model.train()
        # For simplicity, we won’t freeze the MSA model here (you can add similar freezing if desired).
        
        self.fusion_method = fusion_method
        ref_dim = self.esm_model.embed_dim
        msa_dim = self.msa_model.embed_dim
        if fusion_method == 'concat':
            fused_dim = ref_dim + msa_dim
            self.proj = nn.Linear(fused_dim, ref_dim)
        else:
            # Default to simple averaging.
            self.proj = None  # not used
        
        self.contact_head = nn.Bilinear(ref_dim, ref_dim, 1)
    
    def forward(self, ref_tokens, msa_tokens):
        """
        Args:
            ref_tokens (Tensor): Reference sequence tokens of shape (B, L+2).
            msa_tokens (Tensor): MSA tokens of shape (B, num_seqs, L+2).
        Returns:
            contact_probs (Tensor): Contact probability matrix of shape (B, L+2, L+2).
        """
        # Get reference embeddings.
        ref_out = self.esm_model(ref_tokens, repr_layers=[self.esm_model.num_layers])
        ref_emb = ref_out["representations"][self.esm_model.num_layers]  # (B, L+2, d_ref)
        
        # Process the MSA branch.
        B, num_seqs, Lp2 = msa_tokens.shape
        msa_tokens_flat = msa_tokens.view(B * num_seqs, Lp2)
        msa_out = self.msa_model(msa_tokens_flat, repr_layers=[self.msa_model.num_layers])
        msa_emb_flat = msa_out["representations"][self.msa_model.num_layers]  # (B*num_seqs, L+2, d_msa)
        msa_emb = msa_emb_flat.view(B, num_seqs, Lp2, -1).mean(dim=1)  # average over MSA sequences -> (B, L+2, d_msa)
        
        # Fuse the reference and MSA embeddings.
        if self.fusion_method == 'concat':
            fused = torch.cat([ref_emb, msa_emb], dim=-1)  # (B, L+2, d_ref+d_msa)
            fused = self.proj(fused)  # (B, L+2, d_ref)
        else:
            fused = (ref_emb + msa_emb) / 2  # simple average
        
        # Compute pairwise contact logits.
        B, Lp2, d = fused.shape
        x_i = fused.unsqueeze(2).expand(B, Lp2, Lp2, d)
        x_j = fused.unsqueeze(1).expand(B, Lp2, Lp2, d)
        x_i_flat = x_i.reshape(B * Lp2 * Lp2, d)
        x_j_flat = x_j.reshape(B * Lp2 * Lp2, d)
        logits_flat = self.contact_head(x_i_flat, x_j_flat)
        logits = logits_flat.view(B, Lp2, Lp2)
        contact_probs = torch.sigmoid(logits)
        return contact_probs
