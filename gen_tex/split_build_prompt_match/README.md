# build_prompt_split_match
这个目录的作用:
1. 将每个 garment 拆分为上装和下装
2. 为每个 garment 构造 rule-based prompt
3. 将 garment 和 openvton 的模特图做匹配


## 1. split_garment_upper_bottom.py
`split_garment_upper_bottom.py` 用于遍历一个 GarmentCode batch 下的 `random_body` 样本，并尝试把每个 `*_sim.obj` 按 mesh 连通分量拆成 upper 和 bottom 两个 OBJ。


```bash
python split_garment_upper_bottom.py \
  /inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/garmentcode_144k/batch_0 \
  --log log/split_garment_upper_bottom/batch_0.log
```

注意，传入的是 `batch_x` 目录，而不是 `batch_x/random_body`。




**拆分逻辑**
1. 只有当 `len(parts) == 2` 时，脚本才会写出拆分结果：

    ```text
    case_dir/<case_id>_upper.obj
    case_dir/<case_id>_bottom.obj
    ```

2. 当 `len(parts) == 1` 时, 不做任何操作，只写日志并跳过.  

3. 其他数量：认为是 unexpected part count，只写日志并跳过。

4. 输出 OBJ 不包含颜色和贴图。

## 2. check_split.py

`check_split.py` 用于检查指定 batch 下所有 `*_sim.obj` 的 mesh 连通分量数量，确认数据是否满足拆分要求。

```bash
python check_split.py \
  --root /inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/garmentcode_144k \
  --batch_list 0-147 \
  --log log/check_split \
  --workers 12
```

1. 连通分量数量为 1 或 2 时认为正常，不写异常记录。
2. 其他数量、OBJ 缺失或读取失败时，按 batch 写入 `log/check_split/batch_<id>.json`。
3. 默认只做检查。添加 `--delete` 时，会直接删除全部异常 case 的整个 `rand_*` 目录。

## 3. build_prompts.py

`build_prompts.py` 用于基于规则构建 body 的 prompt 以及服装版型的 prompt.

```bash
python build_prompts.py \
  --root /inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/garmentcode_144k/batch_0 \
  --log log/build_prompts/batch_0.log
```

**prompt 构造逻辑**
1. 输出的 prompt 以 json 格式存储在每个样本的目录下
2. body_prompt 描述 body 的身高, 胸围, 腰围, 臀围, 臂长和腿长
3. 样本的 prompt 分为 upper_prompt 和 bottom_prompt 以及总的 sample_prompt. 如果不能拆分上下装就只有一个 whole_prompt 和 sample_prompt(并且二者是一样的)
 
## 4. encode_prompt.py
`encode_prompt.py` 用于为 garment 的 prompt(区分 upper 和 bottom)提取 clip 特征. 

```bash
python encode_prompt.py\
  --root /inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/garmentcode_144k/batch_0 \
  --log log/encode_prompt/batch_0.log
```

1. 使用的模型为 `openai/clip-vit-base-patch32`
2. 提取的特征直接放在 case_id 对应的目录下, 命名为 xxx_upper(/bottom/whole)_prompt_feat.npy

## 5. encode_img.py
`encode_img.py` 用于为真实图片数据集 openvton 的每张图片提取 clip 特征.

```bash
python encode_img.py \
  --root /inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/DataSet/openvton\
  --log log/encode_img/encode_img.log
```

1. 使用的模型为 `openai/clip-vit-base-patch32`
2. 提取的特征直接放在和每张图片一致的目录下, 命名为 source_rgba_feat.npy

## 6. match_garment_img.py
`match_garment_img.py` 的作用是为每个 GCD 的 garment(分上下装)寻找 10 个 openvton 中匹配的图片, 这 10 个图片会在 NI-Tex 生成纹理时按顺序尝试(一个失败了则用下一个, 直到成功). 
```bash
python match_garment_img.py \
  --root /inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/garmentcode_144k \
  --batch_list 0-147 \
  --openvton_root /inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/DataSet/openvton \
  --log log/match_garment_img/all_batches.log
```

1. 指定 `--batch_list` 时，所有选中的 batch 会在同一个 query set 中统一匹配，共享图片使用次数；不指定时兼容原逻辑，处理 `--root` 下的全部 case. 即使在 one_key 中启动, 也只会开一个进程. 

2. 匹配按服装类型全局进行. 尽量确保以 70% 图片覆盖率去使用 openvton 的图片，尽量尊重 clip 的分数. 

## 7. one_key.sh

`one_key.sh` 用于执行数据检查、拆分、prompt 构建和 prompt 编码；检查会合并选中的 batch 后统一运行，图片匹配固定由一个进程执行。每次只执行一种操作；完整处理需要依次调用 `split`、`build_prompt`、`encode_prompt`，再单独运行 `encode_img.py`，最后运行 `match_garment_img`。

```bash
# 检查 batch_0 到 batch_147
bash one_key.sh --op check_split --batch_list 0-147

# 检查并删除所有异常 case
bash one_key.sh --op check_split --batch_list 0-147 --delete

# 先拆分 batch_0 到 batch_12
bash one_key.sh --op split --batch_list 0-147

# 再构建 prompt
bash one_key.sh --op build_prompt --batch_list 0-147

# 提取 prompt 的 clip 特征
bash one_key.sh --op encode_prompt --batch_list 0-147

# 提取 img 特征的 clip 特征
python encode_img.py --root /inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/DataSet/openvton --log log/encode_img/encode_img.log

# 为每个 garment 匹配 OpenVTON 图片
bash one_key.sh --op match_garment_img --batch_list 0-147
```

`--batch_list` 支持单个编号、逗号列表和闭区间，例如 `0`、`0,2,5`、`0-12` 或 `0,2,5-7`。

| 参数 | 说明 |
| --- | --- |
| `--op` | 必填；`check_split`、`split`、`build_prompt`、`encode_prompt` 或 `match_garment_img` |
| `--batch_list` | 必填；需要处理的 batch 编号 |
| `--root` | 包含 `batch_<id>` 的根目录；默认 `/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/garmentcode_144k` |
| `--openvton_root` | OpenVTON 数据根目录；默认 `/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/DataSet/openvton` |
| `--parallel` | `check_split` 的 mesh worker 数，或 `split`、`build_prompt`、`encode_prompt` 的最大并行 batch 数，默认 `12`；匹配操作固定使用一个进程 |
| `--delete` | 仅用于 `check_split`；删除全部异常 case 的整个目录 |


日志会被(依据 op 类型)写入：

```text
log/check_split/batch_<id>.json
log/split_garment_upper_bottom/batch_<id>.log
log/build_prompts/batch_<id>.log
log/encode_prompt/batch_<id>.log
log/match_garment_img/all_batches.log
```
