import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.score_proj = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_tokens, patch_scores):
        """
        patch_tokens: (B, 196, D)
        patch_scores: (B, 14, 14)
        """
        B = patch_tokens.shape[0]
        patch_scores_flat = patch_scores.reshape(B, -1)  # (B, 196)
        patch_scores_emb = self.score_proj(patch_scores_flat.unsqueeze(-1))  # (B,196,D)

        attn_out, _ = self.attn(query=patch_tokens, key=patch_scores_emb, value=patch_scores_emb)
        out = self.norm(patch_tokens + attn_out)
        return out

class FusionClassifier(nn.Module):
    def __init__(self, vit, resnet, embed_dim=768, num_classes=100):
        super().__init__()
        self.vit = vit.eval()
        self.resnet = resnet.eval()
        for p in self.vit.parameters(): p.requires_grad = False
        for p in self.resnet.parameters(): p.requires_grad = False

        self.fusion = PatchAttentionFusion(embed_dim, num_heads=8)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x_resnet, x_vit):
        B = x_resnet.shape[0]
        # ResNet feature map -> saliency -> 14x14 patch scores
        fmap = self.resnet(x_resnet)                       # (B, 2048, 7, 7)
        heatmap = fmap.abs().mean(dim=1)                   # (B, 7, 7)
        heatmap_up = F.interpolate(heatmap.unsqueeze(1), size=(224, 224),
                                   mode='bilinear', align_corners=False)
        patch_scores = heatmap_up.squeeze(1).reshape(B, 14, 16, 14, 16).mean(dim=(2, 4))  # (B,14,14)

        # ViT patch tokens (drop [CLS])
        outputs = self.vit(pixel_values=x_vit)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (B,196,768)

        fused = self.fusion(patch_tokens, patch_scores)     # (B,196,768)
        pooled = fused.mean(dim=1)                          # (B,768)
        logits = self.classifier(pooled)                    # (B,100)
        return logits
