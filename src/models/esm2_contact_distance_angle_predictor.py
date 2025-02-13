import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import math

class TransformerFusion(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=4):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.proj_out = nn.Linear(d_model, input_dim)
    
    def forward(self, x):
        proj_x = self.proj_in(x)
        attn_output, _ = self.mha(proj_x, proj_x, proj_x)
        fusion_weights = torch.sigmoid(self.proj_out(attn_output))
        return fusion_weights

class ESM2ContactDistanceAnglePredictor(nn.Module):
    """
    A multitask model based on ESM2 that predicts three outputs:
      - Contact map (B, L+2, L+2) via a bilinear head and sigmoid.
      - Distance map (B, L+2, L+2) via a bilinear head and ReLU.
      - Angle map (B, L+2, L+2) via a bilinear head and tanh (scaled by π).
    
    Optionally applies transformer-based fusion. The fusion source can be "distance", "angle", or "both".
    """
    def __init__(self, esm_model_name='esm2_t6_8M_UR50D', num_layers_to_freeze=None,
                 feature_fusion=False, fusion_scale=0.5, fusion_source="distance",
                 fusion_d_model=64, fusion_num_heads=4):
        super().__init__()
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
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
        embedding_dim = self.esm_model.embed_dim
        self.contact_head = nn.Bilinear(embedding_dim, embedding_dim, 1)
        self.distance_head = nn.Bilinear(embedding_dim, embedding_dim, 1)
        self.angle_head = nn.Bilinear(embedding_dim, embedding_dim, 1)
        
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

    def forward(self, tokens):
        result = self.esm_model(tokens, repr_layers=[self.esm_model.num_layers])
        x = result["representations"][self.esm_model.num_layers]  # (B, L+2, d)
        B, seq_len, d = x.shape
        x_i = x.unsqueeze(2).expand(B, seq_len, seq_len, d)
        x_j = x.unsqueeze(1).expand(B, seq_len, seq_len, d)
        x_i_flat = x_i.reshape(B * seq_len * seq_len, d)
        x_j_flat = x_j.reshape(B * seq_len * seq_len, d)
        
        contact_logits = self.contact_head(x_i_flat, x_j_flat).view(B, seq_len, seq_len)
        distance_logits = self.distance_head(x_i_flat, x_j_flat).view(B, seq_len, seq_len)
        distance_preds = F.relu(distance_logits)
        angle_logits = self.angle_head(x_i_flat, x_j_flat).view(B, seq_len, seq_len)
        angle_preds = math.pi * torch.tanh(angle_logits)
        
        if self.feature_fusion:
            if self.fusion_source == "distance":
                fusion_input = distance_logits
                current_input_dim = fusion_input.shape[-1]
                if self.transformer_fusion is None or self.transformer_fusion.proj_in.in_features != current_input_dim:
                    self.transformer_fusion = TransformerFusion(input_dim=current_input_dim,
                                                                d_model=self.fusion_d_model,
                                                                num_heads=self.fusion_num_heads).to(tokens.device)
                fusion_weights = self.transformer_fusion(fusion_input)
            elif self.fusion_source == "angle":
                fusion_input = angle_logits
                current_input_dim = fusion_input.shape[-1]
                if self.transformer_fusion is None or self.transformer_fusion.proj_in.in_features != current_input_dim:
                    self.transformer_fusion = TransformerFusion(input_dim=current_input_dim,
                                                                d_model=self.fusion_d_model,
                                                                num_heads=self.fusion_num_heads).to(tokens.device)
                fusion_weights = self.transformer_fusion(fusion_input)
            elif self.fusion_source == "both":
                current_input_dim = distance_logits.shape[-1]
                if self.transformer_fusion_distance is None or self.transformer_fusion_distance.proj_in.in_features != current_input_dim:
                    self.transformer_fusion_distance = TransformerFusion(input_dim=current_input_dim,
                                                                         d_model=self.fusion_d_model,
                                                                         num_heads=self.fusion_num_heads).to(tokens.device)
                if self.transformer_fusion_angle is None or self.transformer_fusion_angle.proj_in.in_features != current_input_dim:
                    self.transformer_fusion_angle = TransformerFusion(input_dim=current_input_dim,
                                                                      d_model=self.fusion_d_model,
                                                                      num_heads=self.fusion_num_heads).to(tokens.device)
                fusion_weights_distance = self.transformer_fusion_distance(distance_logits)
                fusion_weights_angle = self.transformer_fusion_angle(angle_logits)
                fusion_weights = 0.5 * (fusion_weights_distance + fusion_weights_angle)
            else:
                fusion_input = distance_logits
                current_input_dim = fusion_input.shape[-1]
                if self.transformer_fusion is None or self.transformer_fusion.proj_in.in_features != current_input_dim:
                    self.transformer_fusion = TransformerFusion(input_dim=current_input_dim,
                                                                d_model=self.fusion_d_model,
                                                                num_heads=self.fusion_num_heads).to(tokens.device)
                fusion_weights = self.transformer_fusion(fusion_input)
            fused_logits = contact_logits + self.fusion_scale * fusion_weights * contact_logits
            contact_probs = torch.sigmoid(fused_logits)
        else:
            contact_probs = torch.sigmoid(contact_logits)
        
        return contact_probs, distance_preds, angle_preds