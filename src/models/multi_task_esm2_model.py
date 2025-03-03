"""
Multi-task ESM2 Model for Protein Structure Prediction.

This script defines a PyTorch neural network model that integrates ESM2-based embeddings
with multi-task learning for predicting protein contacts, distances, and angles. It supports:
- Sequence-based embeddings from ESM2 models.
- Multiple Sequence Alignment (MSA) embeddings (optional).
- Contact map prediction using a bilinear layer.
- Distance prediction using a distogram-based approach.
- Angle prediction using a discrete binning technique.

The model includes a CrossFusion module that applies attention-based fusion
to integrate multiple task-specific embeddings. The final output includes:
- `contact_logits`: Predicted binary contact map.
- `distance_logits`: Predicted distance distribution across bins.
- `angle_logits`: Predicted dihedral angle bins.
"""

import esm
import torch
import torch.nn as nn


class CrossFusion(nn.Module):
    """Attention-based fusion module for integrating task-specific embeddings."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        attention_output, _ = self.attention(query, key_value, key_value)
        output = self.norm(query + attention_output)
        return output


class MultiTaskESM2Model(nn.Module):
    def __init__(
        self,
        esm2_name="esm2_t6_8M_UR50D",
        esm_msa_name="esm_msa1b_t12_100M_UR50S",
        use_msa=False,
        use_distance=False,
        use_angle=False,
        num_layers_to_freeze_esm2=0,
        num_layers_to_freeze_msa=0,
        max_distance=40,
        num_angle_bins=24,
        fusion_num_heads=4,
        fusion_dim=512,
    ):
        """Multi-task model integrating ESM2 embeddings with task-specific heads for
        protein contact, distance, and angle predictions."""
        super().__init__()
        self.use_msa = use_msa
        self.use_distance = use_distance
        self.use_angle = use_angle
        self.max_distance = max_distance
        self.num_angle_bins = num_angle_bins

        # Load pre-trained ESM2 model and its alphabet.
        self.esm2_model, self.esm2_alphabet = esm.pretrained.load_model_and_alphabet(
            esm2_name
        )
        self.esm2_model.train()

        # Freeze early layers if specified
        for i, layer in enumerate(self.esm2_model.layers):
            if i < num_layers_to_freeze_esm2:
                for param in layer.parameters():
                    param.requires_grad = False

        # Load MSA model if using MSA features.
        if self.use_msa:
            self.msa_model, self.msa_alphabet = esm.pretrained.load_model_and_alphabet(
                esm_msa_name
            )
            self.msa_model.train()

            for i, layer in enumerate(self.msa_model.layers):
                if i < num_layers_to_freeze_msa:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Projection layers for sequence (and optionally MSA) embeddings.
        self.sequence_projection = nn.Linear(self.esm2_model.embed_dim, fusion_dim)
        if self.use_msa:
            self.msa_projection = nn.Linear(self.msa_model.args.embed_dim, fusion_dim)

        # Contact prediction head.
        self.contact_projection = nn.Linear(fusion_dim, fusion_dim)
        self.contact_head = nn.Bilinear(fusion_dim, fusion_dim, 1)

        # Distance prediction head: number of bins is derived as int(max_distance / 8).
        if self.use_distance:
            self.distance_projection = nn.Linear(fusion_dim, fusion_dim)
            self.distance_head = nn.Linear(2 * fusion_dim, int(self.max_distance / 8))

        # Angle prediction head: outputs logits over 2 * num_angle_bins.
        if self.use_angle:
            self.angle_projection = nn.Linear(fusion_dim, fusion_dim)
            self.angle_head = nn.Linear(fusion_dim, 2 * num_angle_bins)

        self.main_projection = nn.Linear(fusion_dim, fusion_dim)
        self.tasks_fusion = CrossFusion(fusion_dim, fusion_num_heads)

    def forward(self, sequence_tokens, msa_tokens=None):
        """Performs a forward pass through the multi-task model.

        Args:
            sequence_tokens (Tensor): Tokenized protein sequence input.
            msa_tokens (Tensor, optional): Tokenized MSA input.

        Returns:
            dict: A dictionary with keys:
                - 'contact_logits': Logits for contact map prediction.
                - 'distance_logits' (if use_distance=True): Logits for distance prediction.
                - 'angle_logits' (if use_angle=True): Logits for angle prediction,
                  reshaped to (B, L, 2, num_angle_bins).
        """
        # Get sequence embeddings from the ESM2 model.
        # Input: sequence_tokens [B, S] -> Output: sequence_embedding [B, S, D]
        with torch.set_grad_enabled(True):
            sequence_output = self.esm2_model(
                sequence_tokens, repr_layers=[self.esm2_model.num_layers]
            )
            sequence_embedding = sequence_output["representations"][
                self.esm2_model.num_layers
            ]

        if self.use_msa:
            # Get MSA embeddings.
            # Input: msa_tokens [B, M, S] -> Output: msa_embeddings [B, M, D]
            with torch.set_grad_enabled(True):
                msa_output = self.msa_model(
                    msa_tokens, repr_layers=[self.msa_model.num_layers]
                )
                msa_embeddings = msa_output["representations"][
                    self.msa_model.num_layers
                ]
            # Average over MSA rows: [B, M, D] -> [B, D]
            msa_embedding = msa_embeddings.mean(dim=1)

        # Project sequence embedding to fusion dimension.
        # [B, S, D] -> [B, S, fusion_dim]
        sequence_embedding = self.sequence_projection(sequence_embedding)

        if self.use_msa:
            # Project MSA embedding: [B, D] -> [B, fusion_dim]
            msa_embedding = self.msa_projection(msa_embedding)

        # Obtain contact features from sequence embedding.
        # [B, S, fusion_dim]
        contact_embedding = self.contact_projection(sequence_embedding)

        extra_embeddings = []
        if self.use_distance:
            # Distance-specific features: [B, S, fusion_dim]
            distance_embedding = self.distance_projection(sequence_embedding)
            extra_embeddings.append(distance_embedding)

        if self.use_angle:
            # Angle-specific features: [B, S, fusion_dim]
            angle_embedding = self.angle_projection(sequence_embedding)
            extra_embeddings.append(angle_embedding)

        if extra_embeddings:
            # Fuse extra task-specific embeddings:
            # Stack: [N, B, S, fusion_dim] then mean -> [B, S, fusion_dim]
            extra_embedding = torch.stack(extra_embeddings, dim=0).mean(dim=0)
            # Apply CrossFusion: [B, S, fusion_dim] -> [B, S, fusion_dim]
            fused_embedding = self.tasks_fusion(contact_embedding, extra_embedding)
        else:
            fused_embedding = contact_embedding

        # Compute contact logits.
        # fused_embedding: [B, S, fusion_dim]
        # Expand to form pairwise features:
        # fused_i: [B, S, S, fusion_dim], fused_j: [B, S, S, fusion_dim]
        B, L, D = fused_embedding.size()
        fused_i = fused_embedding.unsqueeze(2).expand(B, L, L, D)
        fused_j = fused_embedding.unsqueeze(1).expand(B, L, L, D)
        # Bilinear layer produces: [B, S, S, 1] -> squeeze to [B, S, S]
        contact_logits = self.contact_head(fused_i, fused_j).squeeze(-1)

        outputs = {"contact_logits": contact_logits}

        if self.use_distance:
            # Concatenate pairwise fused embeddings:
            # [B, S, S, fusion_dim] concatenated with itself -> [B, S, S, 2 * fusion_dim]
            pair_features = torch.cat(
                [
                    fused_embedding.unsqueeze(2).expand(B, L, L, D),
                    fused_embedding.unsqueeze(1).expand(B, L, L, D),
                ],
                dim=-1,
            )
            # Compute distance logits: [B, S, S, num_distance_bins]
            distance_logits = self.distance_head(pair_features)
            outputs["distance_logits"] = distance_logits

        if self.use_angle:
            # Compute angle logits using the angle-specific embedding.
            # angle_embedding: [B, S, fusion_dim] -> [B, S, 2 * num_angle_bins]
            # Reshape to: [B, S, 2, num_angle_bins]
            angle_logits = self.angle_head(angle_embedding)
            angle_logits = angle_logits.view(B, L, 2, self.num_angle_bins)
            outputs["angle_logits"] = angle_logits

        return outputs
