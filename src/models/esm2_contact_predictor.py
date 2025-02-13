import torch
import torch.nn as nn
import torch.nn.functional as F
import esm

class ESM2ContactPredictor(nn.Module):
    """
    A model based on ESM2 that predicts a contact map from a protein sequence.
    Input tokens are of shape (B, L+2) (including special tokens).
    Output is a contact probability matrix of shape (B, L+2, L+2).
    """
    def __init__(self, esm_model_name='esm2_t6_8M_UR50D', num_layers_to_freeze=None):
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
    
    def forward(self, tokens):
        result = self.esm_model(tokens, repr_layers=[self.esm_model.num_layers])
        x = result["representations"][self.esm_model.num_layers]  # (B, L+2, d)
        B, Lp2, d = x.shape
        x_i = x.unsqueeze(2).expand(B, Lp2, Lp2, d)
        x_j = x.unsqueeze(1).expand(B, Lp2, Lp2, d)
        x_i_flat = x_i.reshape(B * Lp2 * Lp2, d)
        x_j_flat = x_j.reshape(B * Lp2 * Lp2, d)
        logits_flat = self.contact_head(x_i_flat, x_j_flat)
        logits = logits_flat.view(B, Lp2, Lp2)
        probs = torch.sigmoid(logits)
        return probs