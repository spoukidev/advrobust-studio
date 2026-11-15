# Checkpoints

Small CPU-trained CIFAR-10 models produced by `make train` (committed so the UI
and benchmark work out of the box; each is < 1.5 MB).

| File | Arch | Training |
|---|---|---|
| `cifar_cnn_vanilla.pt` | CifarCNN (VGG-style, ~0.29M params) | standard, 10 epochs |
| `cifar_cnn_pgdat.pt` | CifarCNN | PGD-AT, ε=8/255, 3-step PGD, 10 epochs |
| `cifar_resnet_vanilla.pt` | CifarResNet (ResNet-8 style, ~78K params) | standard, 10 epochs |

Format: `{"arch", "state_dict", "meta": {mode, epochs, history, train_seconds}}`.
Load with `studio.models.load_cifar_checkpoint(path)`, which uses `torch.load(weights_only=True)`
(no arbitrary code execution). Models expect inputs in [0, 1].
