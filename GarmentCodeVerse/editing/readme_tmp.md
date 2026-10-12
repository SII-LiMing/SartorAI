读完后可以明确：`GarmentCodeRC/editing` 是一个用于生成“单语义属性编辑配对数据”的离线流水线，不是交互式编辑器。

它的目标是：给定已有 GarmentCode 样本，在保持人体参数完全不变的前提下，只修改一个服装语义属性，经过版型校验和物理仿真后，发布 source-target 编辑对及可复现的自然语言指令。

## 整体流程

```text
源 design/body/specification
        │
        ▼
 plan：枚举合法编辑 + 平衡规则分布
        │
        ▼
 generate：修改一个字段 + 构造版型 + 几何/语义校验 + 渲染
        │
        ▼
 GarmentCode 物理仿真 + filter_fail_sim
        │
        ▼
 project
   ├─ pass：生成 target prompt，正式发布
   └─ fail：归档失败结果，下一轮换一种编辑
        │
        ▼
 status：查看所有 source 的状态
```

命令入口在 [semantic_edit/\_\_main\_\_.py](/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/editing/semantic_edit/__main__.py:10)，对外提供四个命令：

- `plan`
- `generate`
- `project`
- `status`

项目说明集中在 [README.md](/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/editing/README.md:1)。

## 四个阶段具体做什么

### 1. `plan`

核心实现在 [planning.py](/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/editing/semantic_edit/planning.py:747)。

它会：

- 扫描 `<input-root>/batch_N/random_body/data/rand_xxx`
- 读取 `design_params.yaml` 和 `body_measurements.yaml`
- 根据实际 GarmentCode constructor 分支解析哪些参数当前有效
- 调用固定版本的 prompt builder，识别源服装的语义状态
- 让十条规则枚举所有合法的单属性编辑
- 用最小费用流在 rule、direction、source/target state 三个层级平衡编辑分布
- 输出 `pilot_eligibility_report.json`

`plan` 阶段不读取源 specification，也不构造版型，主要做编辑可行性分析。

### 2. `generate`

核心在 [generation.py](/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/editing/semantic_edit/generation.py:540)。

每个未完成 source 会选择一个尚未尝试的候选，然后：

1. 从缓存 operation 中修改指定 YAML 字段。
2. 生成可重放的编辑指令。
3. 调用原始 `MetaGarment` 构造目标版型。
4. 检查空版型、总长度、自相交、panel/stitch 完整性。
5. 比较 source/target specification，确认只发生声明的语义变化。
6. 渲染 PNG 和 SVG。
7. 创建物理仿真 workspace。

此阶段候选目录只有五个文件：

- design YAML
- 原样复制的 body YAML
- specification JSON
- pattern PNG
- pattern SVG

目标构造边界在 [construction.py](/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/editing/semantic_edit/construction.py:522)。

### 3. 物理仿真

这一段复用原始 GarmentCode：

- `pattern_data_sim.py`
- `filter_fail_sim.py --no-delete`

候选会被复制到 `.state/simulation_workspace`。workspace 管理位于 [simulation_workspace.py](/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/editing/simulation_workspace.py:160)。

必须保留完整 filter report，因为 `project` 会根据：

- `passed_cases`
- `final_failed_cases`
- `final_fail_reasons`

确定每个候选的最终状态。

### 4. `project`

核心在 [projection.py](/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/editing/semantic_edit/projection.py:491)。

它要求物理仿真属性中的 `frozen: true`，然后：

- 仿真失败：候选和仿真产品移入 `.state/rejected_*`
- 仿真通过：调用 prompt builder 生成目标 prompt
- prompt 成功：补齐两个文件并正式发布
- prompt 失败：记录为 `prompt_failed`，下一轮换候选重试
- 更新 `accepted_pairs.jsonl` 和 dataset properties
- 删除本轮 simulation workspace

最终接受的 target 有七个文件：

```text
design_params.yaml
body_measurements.yaml
<id>_specification.json
<id>_pattern.png
<id>_pattern.svg
<id>_prompts.json
<id>_edit_metadata.json
```

## 当前支持的十条编辑规则

规则顺序固定在 [registry.py](/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/editing/semantic_edit/registry.py:14)，配置在 [semantic_edit.yaml](/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/editing/configs/semantic_edit.yaml:1)。

| 规则 | 修改字段 | 语义 |
|---|---|---|
| `shirt_fit` | `design.shirt.width.v` | 修身 ↔ 宽松 |
| `shirt_length` | `design.shirt.length.v` | 加长或缩短上衣 |
| `neckline_type` | `design.collar.f_collar.v` | 六种前领口类型切换 |
| `hood_presence` | `design.collar.component.style.v` | 添加/移除 Hood |
| `collar_presence` | 同上 | 添加/移除 SimpleLapel |
| `sleeve_presence` | `design.sleeve.sleeveless.v` | 添加/移除袖子 |
| `sleeve_length` | 主袖或左右袖 `length.v` | 袖长变化 |
| `skirt_length` | 当前激活裙型的 `length.v` | 裙长变化 |
| `godet_insert_count` | `design.godet-skirt.num_inserts.v` | Godet 插片数量变化 |
| `pants_length` | `design.pants.length.v` | 裤长变化 |

这些规则不是简单改 YAML。每条规则同时负责：

- 判断当前设计是否适用
- 枚举目标状态
- 声明读取和写入字段
- 生成语义 transition
- 检查版型拓扑、panel membership 或边界位移
- 提供可审计 evidence

例如：

- neckline 只允许对称上衣且 collar component 为空。
- sleeve presence 要求曲线袖窿、无 cuff，并带袖长和连接宽度安全限制。
- hood 添加要求潜在 `hood_length <= 1.25`。
- godet count 只能取 `4/6/8/10/12`，变化至少为 4，并检查 base interface 可整除和复杂度限制。
- length 类规则使用人体尺寸把数值变化转换为诸如 `cropped`、`midi-length`、`full-length` 等语义状态。

## 状态与重试

状态机在 [store.py](/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/editing/semantic_edit/store.py:87)。

主要状态：

- `not_generated`
- `retryable_or_exhausted`
- `pending_simulation`
- `in_simulation`
- `accepted`
- `physical_limit_reached`

每个 source 最多有 3 次物理尝试：

- `generation_failed` 不消耗物理尝试次数
- `simulation_failed` 消耗一次
- `prompt_failed` 也消耗一次
- 下一轮不会重复相同 operation signature
- `accepted` 后成为不可变终态

## 可复现性和数据安全设计

这部分实现得比较严格：

- body YAML 以原始 bytes 复制，确保同一人体。
- 首次 generate 后把 source 存入 `.state/source_snapshots`，后续不再依赖原始数据变化。
- config、operation、design、specification、prompt builder 都有规范化 hash。
- 编辑指令由 source key 和 operation signature 确定性选择模板，可以严格重放。
- prompt builder 必须匹配代码中写死的 SHA256，文件一旦变化就拒绝运行。
- input、analysis、output 必须互不嵌套。
- 发布目录使用不覆盖的原子 rename。
- 被拒绝的候选不会污染最终数据集。

## 一个值得注意的实现细节

配置要求 `expected_source_count` 必须写成 `100`，但 `plan()` 并没有检查实际扫描到的 source 数量是否等于 100；实际数量只会写入报告。因此它目前更像“配置格式必须声明 100”，而不是“运行时严格要求输入正好 100 个样本”。

当前仓库已经包含需要的外部入口：

- `GarmentCodeRC/pattern_data_sim.py`
- `GarmentCodeRC/filter_fail_sim.py`
- `GarmentCodeRC/check_integrity.py`
- `gen_garment_texture/build_prompt_split_match/build_prompts.py`

根据项目 `AGENTS.md`，实际运行时应使用 conda 的 `chatgarment` 环境，而不是 README 示例中的 `GCD`。本次只做了代码阅读，没有修改任何文件，也没有运行仿真或测试。

## 处理 `garmentcode_144k/batch_0`

下面假定已经进入 `chatgarment` 环境：

```bash
conda activate chatgarment
```

当前 editing 已兼容 `garmentcode_144k` 使用的扁平目录和带样本 ID 前缀的 YAML 文件，因此不需要复制、重命名或重新组织 `batch_0`。以下命令都从 `GarmentCodeRC` 目录运行。

### 1. 设置路径

```bash
PROJECT=/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2
GCD="$PROJECT/GarmentCodeRC"

INPUT_ROOT="$GCD/garmentcode_144k"
TMP_ROOT="$GCD/editing/tmp"
ANALYSIS_ROOT="$TMP_ROOT/batch_0_edit_analysis"
OUTPUT_ROOT="$TMP_ROOT/batch_0_edit_output"
PROMPT_BUILDER="$PROJECT/gen_garment_texture/build_prompt_split_match/build_prompts.py"
WORKSPACE="$OUTPUT_ROOT/.state/simulation_workspace"

cd "$GCD"
```

`ANALYSIS_ROOT` 在首次执行 `plan` 前必须不存在，`OUTPUT_ROOT` 在首次执行 `generate` 前必须不存在。不要把这些目录放进 `INPUT_ROOT` 内部。

### 2. 分析并规划 `batch_0`

```bash
python -B -m editing.semantic_edit plan \
  --input-root "$INPUT_ROOT" \
  --batch-id batch_0 \
  --analysis-root "$ANALYSIS_ROOT" \
  --config editing/configs/semantic_edit.yaml \
  --prompt-builder "$PROMPT_BUILDER" \
  --seed 42
```

规划结果保存在：

```text
<ANALYSIS_ROOT>/pilot_eligibility_report.json
```

`batch_0` 当前包含 607 个源样本。并非所有源样本都一定有合法编辑候选，最终 eligible 数量以命令输出为准。

### 3. 生成第一轮候选

```bash
python -B -m editing.semantic_edit generate \
  --input-root "$INPUT_ROOT" \
  --analysis-root "$ANALYSIS_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --seed 42
```

如果存在可仿真的候选，它们会出现在：

```text
<OUTPUT_ROOT>/.state/simulation_workspace
```

workspace 存在时不要再次执行 `generate`，应先完成下面的仿真、过滤和投影。

### 4. 物理仿真和过滤

```bash
python -B pattern_data_sim.py \
  --data "$WORKSPACE"

python -B filter_fail_sim.py \
  --gcd_dir "$WORKSPACE" \
  --body_type random_body \
  --no-delete
```

当前 `filter_fail_sim.py` 生成的是精简版 `filter_fail_report.yaml`，而 editing 的 `project` 读取 `filter_fail_report_full.yaml`。在执行 `project` 前运行下面的兼容转换：

```bash
python - "$WORKSPACE" <<'PY'
import json
import sys
from pathlib import Path

import yaml

workspace = Path(sys.argv[1]).resolve()
body_root = workspace / "random_body"
manifest = json.loads(
    (workspace / "simulation_manifest.json").read_text(encoding="utf-8")
)
compact = yaml.safe_load(
    (body_root / "filter_fail_report.yaml").read_text(encoding="utf-8")
)

candidate_ids = set(manifest["candidate_ids"])
failure_reasons = compact.get("log_failures_by_case", {})
failed = set(failure_reasons)
unknown = failed - candidate_ids
if unknown:
    raise RuntimeError(f"filter report contains unknown candidates: {sorted(unknown)}")

full_report = {
    "report_type": "filter_fail_report_full",
    "simulation_id": manifest["simulation_id"],
    "passed_cases": sorted(candidate_ids - failed),
    "final_failed_cases": sorted(failed),
    "final_fail_reasons": {
        candidate_id: failure_reasons[candidate_id]
        for candidate_id in sorted(failed)
    },
}
(body_root / "filter_fail_report_full.yaml").write_text(
    yaml.safe_dump(full_report, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)
PY
```

### 5. 投影仿真结果并查看状态

```bash
python -B -m editing.semantic_edit project \
  --output-root "$OUTPUT_ROOT" \
  --prompt-builder "$PROMPT_BUILDER"

python -B -m editing.semantic_edit status \
  --output-root "$OUTPUT_ROOT"
```

`project` 会接受通过物理过滤的候选，归档失败候选，并删除本轮 workspace。

### 6. 处理重试

如果 `status` 显示仍有未完成且可重试的 source，并且 workspace 已经不存在，则再次依次执行：

```text
generate
→ pattern_data_sim.py
→ filter_fail_sim.py --no-delete
→ 生成 filter_fail_report_full.yaml
→ project
→ status
```

每个 source 最多进行三次物理尝试，所以通常最多需要三轮。已有 `ANALYSIS_ROOT` 和 `OUTPUT_ROOT` 时不要重新执行 `plan`，后续 `generate` 会从现有状态继续。

### 7. 最终验证

当所有 source 都已终止并且 workspace 不存在时运行：

```bash
python -B filter_fail_sim.py \
  --gcd_dir "$OUTPUT_ROOT/batch_0" \
  --body_type random_body \
  --no-delete

python -B check_integrity.py \
  --gcd_dir "$OUTPUT_ROOT/batch_0" \
  --body_type random_body \
  --require_filter_report
```

最终接受的数据位于：

```text
<OUTPUT_ROOT>/batch_0/random_body/data/
```

配对清单位于：

```text
<OUTPUT_ROOT>/batch_0/accepted_pairs.jsonl
```
