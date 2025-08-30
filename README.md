# CNN-Guided ViT Fusion (Patch-Attention Fusion)

Lightweight, modular fusion of frozen ResNet-50 saliency with ViT patch tokens via a trainable cross-attention block. 
Designed for low-resource classification (Mini-ImageNet) with minimal additional parameters.

## Highlights
- **Frozen backbones**: ViT & ResNet-50 remain fixed.
- **Small trainable head**: Fusion + classifier only (~2–3% params).
- **Plug-and-play**: No ViT backbone retraining or surgery.
- **Reproducible**: Config-driven runs and logged convergence curves.
