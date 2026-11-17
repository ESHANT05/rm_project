"""
Cross-Attention and Knowledge-Gated Fusion Modules for K-CARN.
Provides Multi-Head Bidirectional Cross-Attention and Knowledge-Gating for Hateful Meme Detection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    """
    Standard scaled dot-product multi-head cross-attention layer.
    Allows query modality to attend to key/value context modality.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value, return_attn=False):
        """
        Args:
            query: Tensor of shape (B, N_q, D) or (B, D)
            key_value: Tensor of shape (B, N_kv, D) or (B, D)
            return_attn: bool, if True, returns attention weights for explainability
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (B, 1, D)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)  # (B, 1, D)

        B, N_q, _ = query.shape
        _, N_kv, _ = key_value.shape

        # Linear projections & split into heads: (B, num_heads, N, head_dim)
        q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        context = torch.matmul(attn_weights_dropped, v)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.embed_dim)
        output = self.out_proj(context)

        # Residual connection and normalization
        output = self.norm(query + output)

        if return_attn:
            # Average attention over heads: (B, N_q, N_kv)
            avg_attn = attn_weights.mean(dim=1)
            return output, avg_attn
        return output


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between visual and textual representations:
    1. Text attends to Vision (grounds text in image regions)
    2. Vision attends to Text (highlights image features relevant to the caption)
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(BidirectionalCrossAttention, self).__init__()
        self.text_to_image_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.image_to_text_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)

        # Feed-forward refinement network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim * 2)

    def forward(self, img_feats, text_feats, return_attn=False):
        """
        Args:
            img_feats: (B, D) or (B, N_v, D)
            text_feats: (B, D) or (B, N_t, D)
        """
        if return_attn:
            t_att_v, t_v_weights = self.text_to_image_attn(text_feats, img_feats, return_attn=True)
            v_att_t, v_t_weights = self.image_to_text_attn(img_feats, text_feats, return_attn=True)
        else:
            t_att_v = self.text_to_image_attn(text_feats, img_feats)
            v_att_t = self.image_to_text_attn(img_feats, text_feats)
            t_v_weights, v_t_weights = None, None

        # Pool across sequence dimension if token sequence
        t_out = t_att_v.mean(dim=1) if t_att_v.dim() == 3 else t_att_v
        v_out = v_att_t.mean(dim=1) if v_att_t.dim() == 3 else v_att_t

        fused = torch.cat([v_out, t_out], dim=-1)
        fused = self.norm(fused + self.ffn(fused))

        if return_attn:
            return fused, {"t_to_v_attn": t_v_weights, "v_to_t_attn": v_t_weights}
        return fused


class KnowledgeGatedFusionBlock(nn.Module):
    """
    Gated integration of external knowledge/rationales.
    Computes cross-attention between multimodal context and knowledge tokens,
    then applies an adaptive gate:
        G = sigmoid(W_g * [fused; knowledge] + b_g)
        H_final = fused + G * Proj(knowledge)
    """
    def __init__(self, fused_dim, knowledge_dim, num_heads=8, dropout=0.1):
        super(KnowledgeGatedFusionBlock, self).__init__()
        self.k_proj = nn.Linear(knowledge_dim, fused_dim)
        self.k_attn = MultiHeadCrossAttention(fused_dim, num_heads=num_heads, dropout=dropout)

        # Learned Gating Mechanism to prevent knowledge hallucinations
        self.gate_dense = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, fused_multimodal, knowledge_feats):
        """
        Args:
            fused_multimodal: (B, fused_dim)
            knowledge_feats: (B, knowledge_dim) or (B, N_k, knowledge_dim)
        """
        k_projected = self.k_proj(knowledge_feats)
        if fused_multimodal.dim() == 2:
            query = fused_multimodal.unsqueeze(1)
        else:
            query = fused_multimodal

        k_attended = self.k_attn(query, k_projected).squeeze(1)

        # Gate computation
        gate = self.gate_dense(torch.cat([fused_multimodal, k_attended], dim=-1))
        gated_knowledge = gate * k_attended

        # Gated residual combination
        out = self.norm(fused_multimodal + gated_knowledge)
        return out


class KCARNClassifier(nn.Module):
    """
    Full K-CARN Classifier with:
    - Independent linear projections for visual, textual, and optional knowledge features.
    - Bidirectional Cross-Attention fusion.
    - Optional Knowledge-Gating.
    - Deep MLP classification and retrieval projection head.
    """
    def __init__(
        self,
        image_dim=1024,
        text_dim=768,
        knowledge_dim=768,
        map_dim=1024,
        proj_dim=1024,
        num_layers=3,
        num_heads=8,
        dropout=(0.1, 0.4, 0.2),
        batch_norm=False,
        use_knowledge=False,
        num_classes=1,
    ):
        super(KCARNClassifier, self).__init__()
        self.use_knowledge = use_knowledge
        self.map_dim = map_dim

        # Input modality projections
        self.img_proj = nn.Sequential(
            nn.Linear(image_dim, map_dim),
            nn.LayerNorm(map_dim),
            nn.Dropout(dropout[0])
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, map_dim),
            nn.LayerNorm(map_dim),
            nn.Dropout(dropout[0])
        )

        # Bidirectional Cross-Attention
        self.bi_cross_attn = BidirectionalCrossAttention(
            embed_dim=map_dim,
            num_heads=num_heads,
            dropout=dropout[0]
        )

        fused_dim = map_dim * 2

        # Optional Knowledge Gating
        if use_knowledge:
            self.knowledge_fusion = KnowledgeGatedFusionBlock(
                fused_dim=fused_dim,
                knowledge_dim=knowledge_dim,
                num_heads=num_heads,
                dropout=dropout[0]
            )

        # Classification & Retrieval MLP Head
        mlp_layers = []
        in_dim = fused_dim
        for _ in range(num_layers):
            mlp_layers.append(nn.Linear(in_dim, proj_dim))
            if batch_norm:
                mlp_layers.append(nn.BatchNorm1d(proj_dim))
            mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Dropout(dropout[2]))
            in_dim = proj_dim

        self.mlp = nn.Sequential(*mlp_layers)
        self.output_layer = nn.Linear(proj_dim, num_classes)

    def forward(self, img_feats, text_feats, knowledge_feats=None, return_embed=False, return_attn=False):
        # 1. Project modalities to shared latent space
        v = self.img_proj(img_feats)
        t = self.text_proj(text_feats)

        # L2-normalize for stable attention dot products
        v_norm = F.normalize(v, p=2, dim=-1)
        t_norm = F.normalize(t, p=2, dim=-1)

        # 2. Bidirectional Cross-Attention Fusion
        if return_attn:
            fused, attn_maps = self.bi_cross_attn(v_norm, t_norm, return_attn=True)
        else:
            fused = self.bi_cross_attn(v_norm, t_norm, return_attn=False)
            attn_maps = None

        # 3. Knowledge-Gating (if knowledge feature provided)
        if self.use_knowledge and knowledge_feats is not None:
            fused = self.knowledge_fusion(fused, knowledge_feats)

        # 4. Dense Retrieval Embedding and Logits
        embed = self.mlp(fused)
        logits = self.output_layer(embed)

        if return_attn and return_embed:
            return logits, embed, attn_maps
        elif return_embed:
            return logits, embed
        elif return_attn:
            return logits, attn_maps
        return logits
