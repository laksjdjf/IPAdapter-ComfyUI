# IPAdapter-ComfyUI
[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)の実験的な実装です。とりあえず動けばいいという思想で実装しているので問題いっぱいありそうです。

# Install
**pytorch >= 2.0.0が必要です。**

1. custom_nodesにくろーん
2. `IPAdapter-ComfyUI/models`に[ip-adapterのチェックポイント](https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter_sd15.bin)を入れる。
3. `ComfyUI/models/clip_vision`に[clip vision model](https://huggingface.co/h94/IP-Adapter/blob/main/models/image_encoder/pytorch_model.bin)を入れる。

# Usage
わーくふろぉ貼ってます。

# Hint
+ clip vision modelは長方形画像を中央切り抜きするっぽいので、あらかじめ自分で切り抜きした方がいいっぽいかもしれないっぽい。
⇒`image/preprocessor`に切り抜き用のノードを追加しました。paddingや検出した顔を基準にした切り抜きを自動でできるようにしています。

# CITIATION
```
@article{ye2023ip-adapter,
  title={IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models},
  author={Ye, Hu and Zhang, Jun and Liu, Sibo and Han, Xiao and Yang, Wei},
  booktitle={arXiv preprint arxiv:2308.06721},
  year={2023}
}
```
