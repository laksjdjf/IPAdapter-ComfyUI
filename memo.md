## plusかどうか
`state_dict["image_proj"]["lantents"]`の存在で判断

## テキストエンコーダの隠れ状態次元数：
keyの入力次元で判断

`cross_attention_dim = state_dict["ip_adapter"]["1.to_k_ip.weight"].shape[1]`

## SDXLかどうか
`sdxl = self.cross_attention_dim == 2048`

## IP-Adapterのトークン数
plusでない場合image_projの出力次元からcross_attention_dimを割る

`clip_extra_context_tokens = state_dict["image_proj"]["proj.weight"].shape[0] // cross_attention_dim`

plusの場合latentsのトークン数で判断

`self.clip_extra_context_tokens = ip_state_dict["image_proj"]["latents"].shape[1]`

## CLIP特徴量の次元数
実際の出力で判断

`clip_embeddings_dim = cond.shape[-1]`

## 残り
plusの場合のresamplerの設定は保留・・・

```
depth=4
dim_head=64
heads=12
ff_mult=4
```
