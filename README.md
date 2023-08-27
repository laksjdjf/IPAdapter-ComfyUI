# IPAdapter-ComfyUI
[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)の[ComfyUI](https://github.com/comfyanonymous/ComfyUI)カスタムノードです。

2023/08/27:
plusモデルの仕様のため、ノードの仕様を変更しました。また複数画像やマスクによる領域指定に対応しました。

# Install

1. custom_nodesにclone
2. `IPAdapter-ComfyUI/models`にip-adapterのモデル（例：[SDv1.5用モデル](https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter_sd15.bin))を入れる。
3. `ComfyUI/models/clip_vision`にCLIP_visionモデル（例：[SDv1.5用モデル](https://huggingface.co/h94/IP-Adapter/blob/main/models/image_encoder/pytorch_model.bin))を入れる。

# Usage
`ip-adapter.json`を参照してください。

## Input
+ **model**：modelをつなげてください。LoRALoaderなどとつなげる順番の違いについては影響ありません。
+ **image**：画像をつなげてください。
+ **clip_vision**：`Load CLIP Vision`の出力とつなげてください。
+ **mask**：任意です。マスクをつなげると適用領域を制限できます。必ず生成画像と同じ解像度にしてください。
+ **weight**：適用強度です。
+ **model_name**：使うモデルのファイル名を指定してください。
+ **dtype**：黒い画像が生成される場合、`fp32`を選択してください。ほとんど生成時間が変わらないのでずっと`fp32`のままでもよいかもしれません。

## Output
+ **MODEL**：KSampler等につなげてください。
+ **CLIP_VISION_OUTPUT**：ふつうは気にしなくていいです。Revision等を使うときに無駄な計算を省くことができます。

## Multiple condition.
ノードを自然につなげることで、複数画像を入力することができます。Maskと組み合わせることで、左右で条件付けを分けるみたいなこともできます。
![image](https://github.com/laksjdjf/IPAdapter-ComfyUI/assets/22386664/c2282aee-ab98-488d-936e-1787994e957f)
背景も分割されてしまうことが問題ですね＾＾；

# Hint
+ 入力画像は自動で中央切り抜きによって正方形にされるので、避けたい場合は予め切り取り処理をするか、`preprocess/furusu Image crop`を使うとよいかもしれません。`preprocess/furusu Image crop`にはパディングをする`padding`とキャラの顔位置を基準に切り取りをする`face_crop`があります。`face_crop`に必要な[lbpcascade_animeface.xml](https://github.com/nagadomi/lbpcascade_animefacehttps://github.com/nagadomi/lbpcascade_animeface)は自動ダウンロードできない場合があるので、その場合は手動でリポジトリ直下に入れてください。

# Bug
+ `Apply ControlNet`はなぜかバグるので、代わりに`Apply ControlNet(Advanced)`を使ってください。

# CITIATION
```
@article{ye2023ip-adapter,
  title={IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models},
  author={Ye, Hu and Zhang, Jun and Liu, Sibo and Han, Xiao and Yang, Wei},
  booktitle={arXiv preprint arxiv:2308.06721},
  year={2023}
}
```
