import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import math

class TransformerFusion(nn.Module):
    """
    A transformer-style fusion module that uses multihead self-attention to compute
    fusion weights from an input tensor.
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

class ESM2ContactDistanceAnglePredictorMSA(nn.Module):
    """
    A multitask model that predicts three outputs from a protein:
      - Contact map (via a bilinear contact head and sigmoid)
      - Distance map (via a distance head and ReLU)
      - Angle map (via an angle head and tanh scaled by π)
    
    This variant integrates MSA information. When the configuration flag `use_msa` is true,
    the model’s forward expects a tuple (ref_tokens, msa_tokens) where:
      - ref_tokens: tokens for the reference sequence (shape: [B, L+2])
      - msa_tokens: tokens for the MSA (shape: [B, num_seqs, L+2])
    
    The MSA branch is processed by a separate pretrained MSA Transformer and its embeddings are
    fused (here by concatenation and projection) with the reference embeddings.
    """
    def __init__(self, 
                 esm_model_name='esm2_t6_8M_UR50D', 
                 msa_model_name='esm_msa1b_t12_100M_UR50S',
                 num_layers_to_freeze=None, 
                 feature_fusion=False, 
                 fusion_scale=1.0, 
                 fusion_source="both",
                 fusion_d_model=128, 
                 fusion_num_heads=8, 
                 use_msa=True):
        super().__init__()
        self.use_msa = use_msa
        # Load the reference model and its alphabet.
        self.ref_model, self.ref_alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
        self.ref_model.train()
        # **New:** Expose the reference alphabet as a top-level attribute.
        self.alphabet = self.ref_alphabet

        if num_layers_to_freeze is not None:
            if hasattr(self.ref_model, "encoder") and hasattr(self.ref_model.encoder, "layers"):
                for i, layer in enumerate(self.ref_model.encoder.layers):
                    if i < num_layers_to_freeze:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        for param in layer.parameters():
                            param.requires_grad = True
            else:
                if num_layers_to_freeze > 0:
                    for param in self.ref_model.parameters():
                        param.requires_grad = False

        # Load the MSA model if requested.
        if self.use_msa:
            self.msa_model, self.msa_alphabet = esm.pretrained.load_model_and_alphabet(msa_model_name)
            self.msa_model.train()

        self.feature_fusion = feature_fusion
        self.fusion_scale = fusion_scale
        self.fusion_source = fusion_source  # "distance", "angle", or "both"
        self.fusion_d_model = fusion_d_model
        self.fusion_num_heads = fusion_num_heads
        if self.feature_fusion:
            if self.fusion_source == "both":
                self.transformer_fusion_distance = None
                self.transformer_fusion_angle = None
            else:
                self.transformer_fusion = None

        ref_dim = self.ref_model.embed_dim
        # Heads
        self.contact_head = nn.Bilinear(ref_dim, ref_dim, 1)
        self.distance_head = nn.Bilinear(ref_dim, ref_dim, 1)
        self.angle_head = nn.Bilinear(ref_dim, ref_dim, 1)
        # For fusion via concatenation, define a projection layer.
        if self.feature_fusion:
            self.fusion_method = "concat"
            self.proj = nn.Linear(ref_dim * 2, ref_dim)
        else:
            self.fusion_method = None

    def forward(self, tokens):
        if self.use_msa and isinstance(tokens, tuple):
            ref_tokens, msa_tokens = tokens
            ref_out = self.ref_model(ref_tokens, repr_layers=[self.ref_model.num_layers])
            ref_emb = ref_out["representations"][self.ref_model.num_layers]  # (B, L+2, d)
            msa_out = self.msa_model(msa_tokens.view(-1, msa_tokens.shape[-1]), repr_layers=[self.msa_model.num_layers])
            msa_emb = msa_out["representations"][self.msa_model.num_layers]  # (B*num_seqs, L+2, d)
            msa_emb = msa_emb.view(msa_tokens.shape[0], msa_tokens.shape[1], msa_tokens.shape[2], -1)
            msa_emb = msa_emb.mean(dim=1)  # average over MSA sequences → (B, L+2, d)
            if self.fusion_method == "concat":
                fused_emb = torch.cat([ref_emb, msa_emb], dim=-1)
                fused_emb = self.proj(fused_emb)
            else:
                fused_emb = (ref_emb + msa_emb) / 2
        else:
            ref_out = self.ref_model(tokens, repr_layers=[self.ref_model.num_layers])
            fused_emb = ref_out["representations"][self.ref_model.num_layers]
        
        B, Lp2, d = fused_emb.shape
        x_i = fused_emb.unsqueeze(2).expand(B, Lp2, Lp2, d)
        x_j = fused_emb.unsqueeze(1).expand(B, Lp2, Lp2, d)
        x_i_flat = x_i.reshape(B * Lp2 * Lp2, d)
        x_j_flat = x_j.reshape(B * Lp2 * Lp2, d)
        
        contact_logits = self.contact_head(x_i_flat, x_j_flat).view(B, Lp2, Lp2)
        distance_logits = self.distance_head(x_i_flat, x_j_flat).view(B, Lp2, Lp2)
        distance_preds = F.relu(distance_logits)
        angle_logits = self.angle_head(x_i_flat, x_j_flat).view(B, Lp2, Lp2)
        angle_preds = math.pi * torch.tanh(angle_logits)
        
        if self.feature_fusion:
            if self.fusion_source == "distance":
                fusion_input = distance_logits
                if self.transformer_fusion is None:
                    self.transformer_fusion = TransformerFusion(input_dim=fusion_input.shape[-1],
                                                                d_model=self.fusion_d_model,
                                                                num_heads=self.fusion_num_heads).to(tokens[0].device if self.use_msa else tokens.device)
                fusion_weights = self.transformer_fusion(fusion_input)
            elif self.fusion_source == "angle":
                fusion_input = angle_logits
                if self.transformer_fusion is None:
                    self.transformer_fusion = TransformerFusion(input_dim=fusion_input.shape[-1],
                                                                d_model=self.fusion_d_model,
                                                                num_heads=self.fusion_num_heads).to(tokens[0].device if self.use_msa else tokens.device)
                fusion_weights = self.transformer_fusion(fusion_input)
            elif self.fusion_source == "both":
                if self.transformer_fusion_distance is None:
                    self.transformer_fusion_distance = TransformerFusion(input_dim=distance_logits.shape[-1],
                                                                         d_model=self.fusion_d_model,
                                                                         num_heads=self.fusion_num_heads).to(tokens[0].device if self.use_msa else tokens.device)
                if self.transformer_fusion_angle is None:
                    self.transformer_fusion_angle = TransformerFusion(input_dim=angle_logits.shape[-1],
                                                                      d_model=self.fusion_d_model,
                                                                      num_heads=self.fusion_num_heads).to(tokens[0].device if self.use_msa else tokens.device)
                fusion_weights_distance = self.transformer_fusion_distance(distance_logits)
                fusion_weights_angle = self.transformer_fusion_angle(angle_logits)
                fusion_weights = 0.5 * (fusion_weights_distance + fusion_weights_angle)
            else:
                fusion_input = distance_logits
                if self.transformer_fusion is None:
                    self.transformer_fusion = TransformerFusion(input_dim=fusion_input.shape[-1],
                                                                d_model=self.fusion_d_model,
                                                                num_heads=self.fusion_num_heads).to(tokens[0].device if self.use_msa else tokens.device)
                fusion_weights = self.transformer_fusion(fusion_input)
            fused_logits = contact_logits + self.fusion_scale * fusion_weights * contact_logits
            contact_probs = torch.sigmoid(fused_logits)
        else:
            contact_probs = torch.sigmoid(contact_logits)
        
        return contact_probs, distance_preds, angle_preds