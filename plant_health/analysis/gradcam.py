"""Grad-CAM (Selvaraju et al., 2017) for CNN and transformer backbones (Sec. 3.5, Figures 7-8).

For CNNs the activations of the final convolutional stage are used; for ViT and Swin the
token activations entering the last attention block are reshaped to a 2-D grid.
"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer, layout='nchw', num_prefix_tokens=0):
        self.model = model
        self.layout = layout  # 'nchw' (CNN), 'nhwc' (Swin, recent timm) or 'tokens' (ViT)
        self.num_prefix_tokens = num_prefix_tokens
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        for h in self._handles:
            h.remove()

    def _to_nchw(self, t):
        """Convert CNN (B,C,H,W), ViT (B,1+N,C) or Swin (B,H,W,C)/(B,N,C) activations to (B,C,H,W)."""
        if t.dim() == 4:
            return t if self.layout == 'nchw' else t.permute(0, 3, 1, 2)
        t = t[:, self.num_prefix_tokens:, :]  # B,N,C
        side = int(round(t.shape[1] ** 0.5))
        return t.reshape(t.shape[0], side, side, t.shape[2]).permute(0, 3, 1, 2)

    def __call__(self, x, class_idx=None):
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        elif isinstance(class_idx, int):
            class_idx = torch.full((x.shape[0],), class_idx, device=x.device, dtype=torch.long)
        self.model.zero_grad()
        logits.gather(1, class_idx.view(-1, 1)).sum().backward()

        acts, grads = self._to_nchw(self.activations), self._to_nchw(self.gradients)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        cam = cam - cam.flatten(1).min(1)[0].view(-1, 1, 1)
        cam = cam / (cam.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
        probs = torch.softmax(logits.detach(), 1)
        return cam.cpu().numpy(), class_idx.cpu().numpy(), probs.cpu().numpy()


def activation_layout(model_name):
    return {'vit_b': 'tokens', 'swin_b': 'nhwc'}.get(model_name, 'nchw')


def num_prefix_tokens(model, model_name):
    if model_name == 'vit_b':
        return int(getattr(model, 'num_prefix_tokens', 1))
    return 0


def denormalize(img_tensor, mean, std):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)
