> 11/08: 修正 `Tokenization` 錯誤的 position embedding 實作

```py
def __init__(...):
    ...
    self.pos_emb = nn.Parameter(torch.randn(1, 1, emb_dim))

def forward(...):
    ...
    pos_emb = self.pos_emb.expand(x.size(0), x.size(1), -1)
```

expand 多出的維度都共用同一組記憶體位置。原本的寫法會讓所有 patch 使用同一個 position embedding，導致模型無法學習到位置資訊。

修改後:
```py
def __init__(..., patches, ...):
    ...
    self.pos_emb = nn.Parameter(torch.randn(1, patches + 1, emb_dim))

def forward(...):
    ...
    pos_emb = self.pos_emb.expand(x.size(0), -1, -1)
```

重新跑了一部分的實驗，確定這是導致 patch_size = 7 和 4 時難以收斂的原因。
