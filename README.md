# IPAdapter-ComfyUI
[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)の実験的な実装です。とりあえず動けばいいという思想で実装しているので問題いっぱいありそうです。

2023/08/23:

The implementation of the [Plus model](https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter-plus_sd15.bin) is being tested in the ```plus``` branch.

**Node specifications will be changed to match plus.**

2023/08/19:

As there have been many reports of black images being generated, the ability to specify the type at inference has been implemented. I don't know if this solves the problem, though, as I'm not experiencing it.
# Install
**pytorch >= 2.0.0が必要です。**

1. custom_nodesにくろーん
2. `IPAdapter-ComfyUI/models`に[SDv1.5用モデル](https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter_sd15.bin)もしく[SDXL用モデル](https://huggingface.co/h94/IP-Adapter/blob/main/sdxl_models/ip-adapter_sdxl.bin)を入れる。
3. `ComfyUI/models/clip_vision`に[SDv1.5用モデル](https://huggingface.co/h94/IP-Adapter/blob/main/models/image_encoder/pytorch_model.bin)もしくは[SDXL用モデル](https://huggingface.co/h94/IP-Adapter/blob/main/sdxl_models/image_encoder/pytorch_model.bin)を入れる。

※Windowsだと[これ](https://github.com/nagadomi/lbpcascade_animeface/blob/master/lbpcascade_animeface.xml)をあらかじめリポジトリ直下にダウンロードしておかないとエラーが起きるかも。

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
