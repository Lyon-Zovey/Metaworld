# WebDataset 数据管线交接文档

> 最后更新：2026-05-11  
> 分支：`whx/webdataset-migration`  
> 维护：weihexiang

---

## 1. 总览

本文档覆盖 WebDataset 数据管线的完整链路：**数据筛选 → 打包 → 后处理 → 训练加载**。

```
┌─────────────────────────────────────────────────────────────────────┐
│                         数据生命周期                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  原始数据 (NFS)          打包 (Argo)           最终产物 (/mnt2)      │
│  ─────────────          ──────────           ──────────────         │
│                                                                     │
│  /mnt/Object_World_Dataset/   ──┐                                   │
│    agibot/any4d/...             │    Batch Scheduler                 │
│    sam2/masks/...               ├──→ wds-pack WFT ──→ pack_parallel/ │
│    sam3d/meshes/...             │    (171 pods)        ├── <uuid>/   │
│    pose/...                     │                     │   shard-*.tar│
│                                 │                     │   index.json │
│  golden JSONL ──────────────────┘                     │              │
│  (56476 records)                                      ↓              │
│                                               flatten_shards.py      │
│                                                       ↓              │
│                                               pack_parallel_flat/    │
│                                                 ├── shard-000000.tar │
│                                                 ├── ...              │
│                                                 ├── shard-000010.tar │
│                                                 └── index.json       │
│                                                       ↓              │
│                                               WdsFlowDataset         │
│                                               (训练时 IterableDataset)│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据源与筛选

### 2.1 Golden JSONL

路径：`/mnt2/ann_v1_wbs/golden/agibot_pose_ok_56476.jsonl`

每行一条轨迹记录，格式：

```json
{
  "seg_key": "351/785186/videos/head_color/head_color/segments/seg000",
  "time_seg": "/mnt/Object_World_Dataset/.../seg000.mp4",
  "any4d": "/mnt/Object_World_Dataset/.../any4d/.../seg000/",
  "sam2": "/mnt/Object_World_Dataset/.../sam2/.../seg000/",
  "sam3d": "/mnt/Object_World_Dataset/.../sam3d/.../seg000/",
  "pose": "/mnt/Object_World_Dataset/.../pose/.../seg000/"
}
```

| 字段 | 说明 |
|------|------|
| `seg_key` | 唯一标识，用作 traj_id |
| `time_seg` | RGB 视频 mp4 路径 |
| `any4d` | scene flow 输出目录（含 anchor.npy, flow mp4, cam_poses.npy 等） |
| `sam2` | SAM2 分割 mask 目录 |
| `sam3d` | 物体 3D mesh (ply/glb) 目录 |
| `pose` | 物体 6DoF pose 目录 |

### 2.2 筛选标准

Golden 集通过 QC 管线产出（详见 `/mnt2/ann_v1_wbs/golden/golden_stats_v2.json`）：

| 检查项 | 阈值 |
|--------|------|
| 相机内参一致性 (intra) | 帧间 fx/fy CV < 3% |
| 相机内参一致性 (inter) | 偏离同相机中位数 < 3σ |
| SAM2 mask 连通域 | 多连通帧占比 < 50% |
| mask 全空 / 连续空帧 | 不允许连续 ≥10 帧空 |
| mask 过大 | 单帧不超过图像面积 50% |

### 2.3 数据规模

| 质量层 | 数据集 | 记录数 | JSONL 路径 |
|--------|--------|--------|------------|
| Golden | Agibot | 56,476 | `/mnt2/ann_v1_wbs/golden/agibot_pose_ok_56476.jsonl` |
| Golden | SthV2 | 17,429 | `/mnt2/ann_v1_wbs/golden/sthv2_pose_ok_17429.jsonl` |
| Golden | EPIC | 53 | `/mnt2/ann_v1_wbs/golden/epic_pose_ok_53.jsonl` |
| Silver | Agibot | 133,921 | `/mnt2/ann_v1_wbs/silver/agibot_pose_ok_133921.jsonl` |
| Silver | SthV2 | 52,393 | `/mnt2/ann_v1_wbs/silver/sthv2_pose_ok_52393.jsonl` |
| Silver | EPIC | 231 | `/mnt2/ann_v1_wbs/silver/epic_pose_ok_231.jsonl` |

---

## 3. 打包流程

### 3.1 核心脚本

**`wds_tools/pack_shards.py`** (v0.3) — 将原始文件打包为 WebDataset tar shards。

支持三种模式：

| 模式 | 用法 | 数据源 |
|------|------|--------|
| `agibot` | `--dataset agibot` | metadata.csv + publish.jsonl join |
| `agibot_golden` | `--dataset agibot_golden --golden-jsonl <path>` | 预筛 golden JSONL |
| `maniskill` | `--dataset maniskill` | ManiSkill metadata.csv |

v0.3 变化（相对 v0.2）：
- 打包全部 5 个 ref 的 scene flow（不再只打 ref00000）
- 新增 caption（来自 segments_meta.json description）
- meta.json 扩展为 11 段完整溯源
- 新增 vlm_objects（左右手物体名 + key_frame + contact_frame）
- **不兼容 v0.2 shard**（`flow.mp4` / `anchor.npy` 已移除）

### 3.2 每条 sample 打包内容（v0.3）

```
shard-000000.tar
├── agibot/000000.rgb.mp4                        # RGB 视频（原始时长，未抽帧）
├── agibot/000000.flow_ref00000.mp4              # scene flow ref=0
├── agibot/000000.flow_ref00000.anchor.npy       # 3D anchor 点云 (H×W×3 float32)
├── agibot/000000.flow_ref00000.sidecar.json     # 编码参数 (max_scale 等)
├── agibot/000000.flow_ref00020.mp4              # scene flow ref=20
├── agibot/000000.flow_ref00020.anchor.npy
├── agibot/000000.flow_ref00020.sidecar.json
├── agibot/000000.flow_ref00040.*                # ...共 5 套 ref
├── agibot/000000.flow_ref00060.*
├── agibot/000000.flow_ref00079.*
├── agibot/000000.cam_poses.npy                  # 相机外参 (T×4×4)
├── agibot/000000.cam_intrinsics.npy             # 相机内参
├── agibot/000000.depth.b2nd                     # 深度视频 (blosc2)
├── agibot/000000.depth_meta.json                # 深度元数据
├── agibot/000000.mask_Left_hand.npz             # SAM2 mask（每个物体一份）
├── agibot/000000.mesh_Left_hand.ply             # 3D mesh（每个物体一份）
├── agibot/000000.pose_Left_hand.npy             # pose T×4×4（每个物体一份）
└── agibot/000000.meta.json                      # 元数据 JSON（11 段溯源）
```

> 物体名为 VLM 检测到的真实语义名，agibot 数据集中常见为 `Right_hand` / `Left_hand`。每条 sample 的物体数量不定（0-2 个），物体列表存在 `meta.json` 的 `objects` 字段。
>
> 5 个 ref 的索引来自 any4d 原始产物，每条轨迹不同（如 0/20/40/60/79 或 0/19/38/57/75），取决于轨迹总帧数。训练时 `WdsFlowDataset` 自动选最接近 `source_frame` 的 ref。

`meta.json` 实际样本（v0.3）：
```json
{
  "task_name": "agibot",
  "traj_id": "532/793660/videos/head_color/head_color/segments/seg000",
  "split": "training",
  "caption": "The right robot arm moves towards the cabinet door handle, while the left arm holds a hanger with black clothing.",
  "source_video": {
    "n_frames": 80,
    "fps": 16.0,
    "duration_s": 5.0,
    "height": 480,
    "width": 640,
    "path": "/mnt/Object_World_Dataset/.../seg000.mp4"
  },
  "flow_refs": [0, 20, 40, 60, 79],
  "flow_encoding": {
    "encoding": "ref_delta_4ch_v3",
    "codec": "libx265",
    "pix_fmt": "yuv444p10le",
    "bits": 10,
    "crf": 0,
    "max_scale": 0.173828125,
    "shape_THW": [80, 480, 640],
    "native_alpha": false
  },
  "vlm_objects": [
    {
      "name": "Left_hand",
      "object_name": "wooden hanger with black clothing",
      "key_frame": 34,
      "contact_frame": null,
      "sample_frame_indices": [0, 20, 40, 59, 79]
    }
  ],
  "dataset_source": {
    "dataset": "agibot",
    "subset": "golden",
    "loader": "agibot_golden",
    "golden_jsonl": "/mnt2/ann_v1_wbs/golden/agibot_pose_ok_56476.jsonl",
    "golden_line_no": 0,
    "seg_key": "351/785186/videos/head_color/head_color/segments/seg000"
  },
  "original_paths": {
    "rgb_video": "/mnt/Object_World_Dataset/.../seg000.mp4",
    "any4d_dir": "/mnt/Object_World_Dataset/.../any4d/.../seg000",
    "sam2_dir": "/mnt/Object_World_Dataset/.../sam2/.../seg000",
    "sam3d_dir": "/mnt/Object_World_Dataset/.../sam3d/.../seg000",
    "pose_dir": "/mnt/Object_World_Dataset/.../pose/.../seg000",
    "vlm_bbox_dir": "/mnt1/Object_World_Dataset/.../vlm_bbox/.../seg000",
    "flow_videos": ["...ref00000...mp4", "...ref00020...mp4", "..."],
    "flow_anchors": ["...ref00000.anchor.npy", "..."]
  },
  "packed": {
    "script": "wds_tools/pack_shards.py",
    "script_version": "v0.3",
    "timestamp": "2026-05-09T17:15:22+0800",
    "shard_index_in_records": 0
  },
  "objects": ["Left_hand"]
}
```

### 3.3 单次测试打包

```bash
cd /mnt/home/weihexiang/Object-centric-World-Model/wds_tools

# 打包 100 条测试
python pack_shards.py \
  --dataset agibot_golden \
  --golden-jsonl /mnt2/ann_v1_wbs/golden/agibot_pose_ok_56476.jsonl \
  --output /mnt2/ann_v1_wbs/golden/agibot/test_pack \
  --limit 100 \
  --max-size 500000000 \
  --max-count 200

# Dry-run 预览
python pack_shards.py \
  --dataset agibot_golden \
  --golden-jsonl /mnt2/ann_v1_wbs/golden/agibot_pose_ok_56476.jsonl \
  --dry-run
```

---

## 4. Argo 并行打包

### 4.1 镜像

```
js4.blockelite.cn:22005/rbs/wds-pack:v0.3
```

构建方式：
```bash
cd /mnt/home/weihexiang/Object-centric-World-Model/wds_tools
sudo docker build -t js4.blockelite.cn:22005/rbs/wds-pack:v0.3 .
sudo docker push js4.blockelite.cn:22005/rbs/wds-pack:v0.3
```

依赖（`requirements.txt`）：
```
numpy>=1.24
pandas>=2.0
webdataset>=0.2.86
```

### 4.2 WorkflowTemplate

文件：`/mnt/home/weihexiang/software/argo/wft-wds-pack.yaml`

```bash
# 部署/更新 WFT
KUBECONFIG=/mnt/home/weihexiang/.kube/config \
kubectl apply -f /mnt/home/weihexiang/software/argo/wft-wds-pack.yaml
```

WFT 参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_jsonl` | golden JSONL | 输入记录列表 |
| `output_dir` | — | 输出根目录 |
| `limit` | `"100"` | 每 pod 最多处理数（`"0"` 不限） |
| `max_size` | `"500000000"` | 单 tar 最大 500MB |
| `max_count` | `"200"` | 单 tar 最多 200 sample |
| `no_h5` | `"true"` | 不打包 h5（agibot 无 h5） |
| `shard_id` | `""` | 由 scheduler 注入，写到 output_dir/<shard_id>/ |

资源配额：每 pod CPU 4-8 核，内存 16-32GB。

Volume mounts：`/mnt`, `/mnt1`, `/mnt2`, `/media`

### 4.3 通过 Batch Scheduler 提交全量打包

```bash
# 获取 Token
TOKEN=$(curl -s -X POST http://js4.blockelite.cn:22004/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"rbs123"}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")

# 提交 Batch
curl -X POST http://js4.blockelite.cn:22004/api/batches \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "wds-pack-agibot-golden",
    "template_name": "wds-pack",
    "input_manifest": "/mnt2/ann_v1_wbs/golden/agibot_pose_ok_56476.jsonl",
    "shard_size": 500,
    "priority": 10,
    "owner": "weihexiang",
    "resource_pool_name": "production",
    "params": {
      "output_dir": "/mnt2/ann_v1_wbs/golden/agibot/pack_parallel",
      "limit": "0",
      "max_size": "500000000",
      "max_count": "200"
    }
  }'
```

**关键参数**：
- `shard_size: 500` → 56K 记录 ÷ 500 = 113 pods
- `limit: "0"` → 必须显式传，否则 WFT 默认 100 会截断
- Scheduler 自动为每个 pod 注入唯一 `shard_id`（UUID），输出到 `output_dir/<uuid>/`

### 4.4 查看进度

```bash
# Web UI
http://js4.blockelite.cn:22003/batches

# API
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://js4.blockelite.cn:22004/api/batches/<batch_id>" | python3 -m json.tool
```

---

## 5. 后处理：Flatten 合并

并行打包完成后，每个 pod 的输出在独立 UUID 子目录中：

```
/mnt2/ann_v1_wbs/golden/agibot/pack_parallel/
├── 0120a087-72fb-44bd-badf-5b2ed01c1462/
│   ├── shard-000000.tar
│   ├── shard-000001.tar
│   └── index.json
├── 01794b8e-95d2-47b3-856e-e26df079964c/
│   └── ...
└── ...  (共 171 个子目录)
```

**`flatten_shards.py`** 将所有 shard 重命名（rename，O(1) 不拷贝）为连续编号：

```bash
cd /mnt/home/weihexiang/Object-centric-World-Model/wds_tools

# 预览
python flatten_shards.py \
  --input /mnt2/ann_v1_wbs/golden/agibot/pack_parallel \
  --output /mnt2/ann_v1_wbs/golden/agibot/pack_parallel_flat \
  --dry-run

# 执行
python flatten_shards.py \
  --input /mnt2/ann_v1_wbs/golden/agibot/pack_parallel \
  --output /mnt2/ann_v1_wbs/golden/agibot/pack_parallel_flat
```

产出：

```
/mnt2/ann_v1_wbs/golden/agibot/pack_parallel_flat/
├── shard-000000.tar
├── shard-000001.tar
├── ...
├── shard-000010.tar
└── index.json      ← 合并索引
```

`index.json`：
```json
{
  "dataset_type": "webdataset",
  "n_samples": 87,
  "n_shards": 11,
  "total_bytes": 4162252800,
  "shard_pattern": "shard-{000000..000010}.tar"
}
```

---

## 6. 验证

### 6.1 Shard 完整性验证

```bash
# 快速检查前 5 个 sample
python verify_shards.py /mnt2/ann_v1_wbs/golden/agibot/pack_parallel_flat/

# 全量扫描（检查所有 sample 的字段完整性）
python verify_shards.py /mnt2/ann_v1_wbs/golden/agibot/pack_parallel_flat/ --full

# 解码验证（实际解码 video/numpy 检查 shape）
python verify_shards.py /mnt2/ann_v1_wbs/golden/agibot/pack_parallel_flat/ --decode --limit 20
```

### 6.2 数据一致性验证

对比同一条轨迹在 WDS 和 Original 管线中的输出：

```bash
# 精确同轨迹对比（默认 3 条）
python consistency_check_exact.py \
  --shard-dir /mnt2/ann_v1_wbs/golden/agibot/pack_parallel_flat \
  --n-samples 5

# 统计分布对比（50 条随机）
python consistency_check.py
```

已验证结论（详见 `EVALUATION_REPORT.md` Section 5.4）：

| 数据类型 | 一致性 |
|----------|--------|
| RGB 视频 | 完全一致（bitwise identical） |
| anchor 点云 | 完全一致 |
| 帧索引 | 完全一致 |
| Flow I-frame 区间 (帧 0-14) | 完全一致 |
| Flow P/B-frame (帧 15+) | 有 H.265 解码微差，mean_diff ~1%，不影响训练 |

### 6.3 吞吐 Benchmark

```bash
python benchmark_throughput.py --mode both --limit 50
```

| 指标 | WDS | Original | 变化 |
|------|-----|----------|------|
| 吞吐量 | 0.49 samples/s | 0.37 samples/s | +32% |
| 冷启动 | <1s | 45s | -98% |

---

## 7. Dataloader 适配使用

### 7.1 核心文件

| 文件 | 说明 |
|------|------|
| `datasets/wds_flow.py` | `WdsFlowDataset(IterableDataset)` — 训练 dataloader |
| `configurations/dataset/agibot_flow_wds.yaml` | Hydra 配置 |
| `experiments/ptv3_flow.py` | 注册表：`agibot_flow_wds=WdsFlowDataset` |

### 7.2 训练命令

```bash
python main.py name=wds_train experiment=ptv3_flow algorithm=ptv3_flow_model \
  dataset=agibot_flow_wds \
  dataset.data_root=/mnt2/ann_v1_wbs/golden/agibot/flatten \
  experiment.tasks=[training] experiment.strategy=ddp \
  experiment.training.lr=1e-4 experiment.training.max_steps=10000 \
  experiment.training.batch_size=8 algorithm.cross_attn.type=single
```

关键 Hydra 覆盖：
- `dataset=agibot_flow_wds` → 选择 WDS dataloader
- `dataset.data_root=<flat_dir>` → 指向 flatten 后的 shard 目录

### 7.3 WdsFlowDataset 数据流（v0.3）

```
wds.WebDataset(shard_pattern, nodesplitter=wds.split_by_node)
  → .shuffle(1000)                        # training 时 shard 内 shuffle
  → .map(_process_sample)                 # 解码 + 处理
      ├── decord.VideoReader(BytesIO)     # RGB → (49, 3, 480, 832)
      ├── meta.flow_refs → 选最近 ref     # 按 source_frame 选 ref
      │   例: source_frame=0 → ref00000
      │       source_frame=40 → ref00040
      ├── np.load(BytesIO)               # anchor → (H, W, 3)
      ├── decompress_agibot_flow(BytesIO) # flow → (T_src, H, W, 3)
      ├── _temporal_sample()             # 抽 49 帧索引
      ├── flow[indices]                  # temporal subsample
      ├── _sample_full()                 # voxel_uniform → 6400 点
      └── displacement = pts[1:] - pts[0] # 相对位移
  → DataLoader(batch_size, num_workers)
```

> **注意**：v0.3 shard 保留原始时长视频，抽帧在 dataloader 运行时完成（灵活性优先）。
> flow_decode 占单样本加载时间 ~76%（PyAV H.265 顺序解码），多 worker 掩盖。

### 7.4 DDP 分布式

- `wds.split_by_node` 自动按 rank 分 shard
- 要求：**shard 数 ≥ num_nodes × num_workers**
- Golden Agibot 661 shards，支持大规模多节点训练
- Silver Agibot 1,679 shards，更充裕

### 7.5 输出 batch dict

```python
{
    "videos":              (B, 49, 3, 480, 832),  # float32, [-1, 1]
    "full_points_src":     (B, 6400, 3),          # float32, source 点云
    "full_flow":           (B, 48, 6400, 3),      # float32, 相对位移
    "frame_indices":       (B, 49),               # int64
    "cam_pose":            (B, 4, 4),             # float32, 可选
    "full_flow_source_frame": int,
    "full_flow_type":      "scene",
    "full_flow_target":    "relative",
    "full_flow_reference": "first",
    "_full_flow_scale":    1.0,
    "full_flow_indices":   (B, 6400),             # int64
    "full_flow_pixel_hw":  (B, 6400, 2),          # int64
    "traj_id":             str,
    "task_name":           str,
}
```

与原 `AgibotFlowDataset` 输出完全兼容，模型代码无需修改。

### 7.6 向后兼容

原有 `AgibotFlowDataset`（Map-style）完全保留。切换方式：

```bash
# 原管线
dataset=agibot_flow

# WDS 管线
dataset=agibot_flow_wds dataset.data_root=<shard_dir>
```

---

## 8. 最终产物目录结构

所有数据集已完成打包和 flatten，训练统一使用 `flatten/` 路径。

```
/mnt2/ann_v1_wbs/
├── golden/
│   ├── agibot_pose_ok_56476.jsonl         ← 输入清单
│   ├── golden_stats_v2.json               ← QC 统计
│   ├── agibot/
│   │   ├── flatten/                       ← ★ 训练用这个
│   │   │   ├── shard-{000000..000660}.tar   52,337 samples, 661 shards, 3.2TB
│   │   │   └── index.json
│   │   ├── pack_parallel/                 ← 并行打包原始输出（可清理）
│   │   ├── pack_parallel_flat/            ← 早期小规模测试 flatten（87 samples, 可清理）
│   │   └── pack_parallel_test/            ← 早期测试（可清理）
│   ├── sthv2/
│   │   └── flatten/                       ← ★ 训练用这个
│   │       ├── shard-{000000..000052}.tar    5,826 samples, 53 shards, 222.6GB
│   │       └── index.json
│   ├── sthv2_pose_ok_17429.jsonl
│   ├── epic/
│   │   └── flatten/                       ← ★ 训练用这个
│   │       ├── shard-000000.tar              35 samples, 1 shard, 5.1GB
│   │       └── index.json
│   └── epic_pose_ok_53.jsonl
├── silver/
│   ├── agibot/
│   │   └── flatten/                       ← ★ 训练用这个
│   │       ├── shard-{000000..001678}.tar   126,784 samples, 1,679 shards, 8.1TB
│   │       └── index.json
│   ├── agibot_pose_ok_133921.jsonl
│   ├── sthv2/
│   │   └── flatten/                       ← ★ 训练用这个
│   │       ├── shard-{000000..000165}.tar   20,425 samples, 166 shards, 745.2GB
│   │       └── index.json
│   ├── sthv2_pose_ok_52393.jsonl
│   ├── epic/
│   │   └── flatten/                       ← ★ 训练用这个
│   │       ├── shard-{000000..000005}.tar   179 samples, 6 shards, 26.4GB
│   │       └── index.json
│   ├── epic_pose_ok_231.jsonl
│   └── silver_stats_v1_0430.json
└── flatten_shards.py                      ← 远端备份的 flatten 脚本
```

### 数据总量汇总

| 质量层 | 数据集 | Samples | Shards | 大小 | 路径 |
|--------|--------|---------|--------|------|------|
| Golden | Agibot | 52,337 | 661 | 3.2 TB | `golden/agibot/flatten/` |
| Golden | SthV2 | 5,826 | 53 | 222.6 GB | `golden/sthv2/flatten/` |
| Golden | EPIC | 35 | 1 | 5.1 GB | `golden/epic/flatten/` |
| Silver | Agibot | 126,784 | 1,679 | 8.1 TB | `silver/agibot/flatten/` |
| Silver | SthV2 | 20,425 | 166 | 745.2 GB | `silver/sthv2/flatten/` |
| Silver | EPIC | 179 | 6 | 26.4 GB | `silver/epic/flatten/` |
| **总计** | | **205,586** | **2,566** | **~12.3 TB** | |

---

## 9. 常见操作速查

### 新增数据集打包（如 SthV2）

```bash
# 1. 确认 JSONL 存在
cat /mnt2/ann_v1_wbs/golden/sthv2_pose_ok_17429.jsonl | wc -l

# 2. 提交打包 batch（修改 input_manifest 和 output_dir）
curl -X POST http://js4.blockelite.cn:22004/api/batches ...
  "input_manifest": "/mnt2/ann_v1_wbs/golden/sthv2_pose_ok_17429.jsonl",
  "output_dir": "/mnt2/ann_v1_wbs/golden/sthv2/pack_parallel",

# 3. 打包完成后 flatten
python flatten_shards.py \
  --input /mnt2/ann_v1_wbs/golden/sthv2/pack_parallel \
  --output /mnt2/ann_v1_wbs/golden/sthv2/pack_flat

# 4. 验证
python verify_shards.py /mnt2/ann_v1_wbs/golden/sthv2/pack_flat --full
```

### 更新镜像

```bash
cd /mnt/home/weihexiang/Object-centric-World-Model/wds_tools
# 编辑 pack_shards.py / requirements.txt
sudo docker build -t js4.blockelite.cn:22005/rbs/wds-pack:v0.3 .
sudo docker push js4.blockelite.cn:22005/rbs/wds-pack:v0.3
# 更新 wft-wds-pack.yaml 中的 image tag
KUBECONFIG=/mnt/home/weihexiang/.kube/config \
kubectl apply -f /mnt/home/weihexiang/software/argo/wft-wds-pack.yaml
```

### 更换 shard 目录训练

只需修改 `data_root`：

```bash
# Golden Agibot
python main.py ... dataset=agibot_flow_wds \
  dataset.data_root=/mnt2/ann_v1_wbs/golden/agibot/flatten

# Silver Agibot（更大规模）
python main.py ... dataset=agibot_flow_wds \
  dataset.data_root=/mnt2/ann_v1_wbs/silver/agibot/flatten

# Golden SthV2
python main.py ... dataset=agibot_flow_wds \
  dataset.data_root=/mnt2/ann_v1_wbs/golden/sthv2/flatten
```

### 排查打包失败的 pod

```bash
# 查看某个 pod 的日志
KUBECONFIG=/mnt/home/weihexiang/.kube/config \
argo logs -n argo <workflow-name>

# 常见问题：FileNotFoundError → 检查 volume mounts
# 确保 WFT 有 /mnt, /mnt1, /mnt2, /media 四个挂载
```

---

## 10. 常见问题

### Q: Pod 报 FileNotFoundError 找不到原始文件

WFT 的 volume mounts 不全。当前需要挂载 `/mnt`, `/mnt1`, `/mnt2`, `/media` 四个路径。确认 WFT 已包含后重新 apply。

### Q: 多 pod 会不会互相覆盖 shard 文件？

不会。Batch Scheduler 为每个 pod 注入唯一 `shard_id`（UUID），pod 输出到 `output_dir/<shard_id>/`，互相隔离。

### Q: limit 参数不生效

通过 Scheduler 提交时必须在 `params` 中传 `"limit": "0"`（字符串），否则 WFT 默认 `limit=100` 会截断每个 pod。

### Q: 单个 sample 有多大？

Agibot golden 平均约 **61 MB/sample**（含 rgb.mp4 + flow.mp4 + anchor + depth + mask + mesh + pose）。全量 golden 52K samples 共 3.2 TB，silver 127K samples 共 8.1 TB。

### Q: flow 精确一致性有差异？

H.265 视频从 BytesIO 解码和从文件路径解码在 P/B-frame 上有微小差异（I-frame 完全一致）。差异 mean ~1%，不影响训练。详见 `EVALUATION_REPORT.md` Section 5.4。

### Q: source_frame 在哪里配置？当前训练用什么值？

`source_frame` 决定 displacement 的参考帧（`full_flow = pts[1:] - pts[source_frame]`）。

| 位置 | 值 | 说明 |
|------|-----|------|
| `configurations/dataset/flow_base.yaml:26` | `full_flow.source_frame: 0` | 默认值 |
| `configurations/dataset/agibot_flow_wds.yaml` | 未覆盖 | 继承 flow_base |
| `datasets/wds_flow.py:69` | `cfg.get("source_frame", 0)` | 读取，fallback 也是 0 |

**当前所有 WDS 训练都使用 `source_frame=0`**，即以第 0 帧作为 anchor，预测后续 48 帧相对于第 0 帧的位移。

### Q: any4d 原始产物有 5 个 ref，shard 里打包了几个？

v0.3 **全部 5 个 ref 都打包**。每个 ref 包含 `.mp4` + `.anchor.npy` + `.sidecar.json` 三个文件，命名为 `flow_ref{NNNNN}.*`。

训练时 `WdsFlowDataset` 根据 `source_frame` 自动选最近的 ref：
- `source_frame=0`（当前默认）→ 选 `flow_ref00000`
- `source_frame=40` → 选 `flow_ref00040`（H.265 解码误差最小）

这样支持未来 `source_frame` 随机采样的数据增强，无需重新打包。

### Q: 如何支持 object flow？

当前 `pack_shards.py` 已打包 mask/mesh/pose 三种物体数据到 shard。WdsFlowDataset 当前只读 scene flow。支持 object flow 需要：
1. 在 `_process_sample()` 中读取 `mesh_<obj>.ply` + `pose_<obj>.npy`
2. 计算 object trajectory（参考 `ManiskillFlowDataset._load_object_flow_raw()`）
3. 与 scene flow concat：`full_points_tgt = [object_pts, scene_pts]`

---

## 11. 相关文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 评估报告 | `wds_tools/EVALUATION_REPORT.md` | 性能、一致性验证详情 |
| 操作手册 | `wds_tools/OPS.md` | 打包/flatten/验证命令速查 |
| 迁移计划 | `wds_tools/MIGRATION_PLAN.md` | 设计方案和实施记录 |
| Dataloader 文档 | `wds_tools/CURRENT_DATALOADER.md` | 原管线详细实现分析 |
| Argo WFT | `/mnt/home/weihexiang/software/argo/wft-wds-pack.yaml` | K8s WorkflowTemplate |

---

## 12. TODO

- [x] 全量打包 Golden Agibot（52,337 samples, 661 shards, 3.2TB）
- [x] 打包 Golden SthV2（5,826 samples, 53 shards, 222.6GB）
- [x] 打包 Golden EPIC（35 samples, 1 shard, 5.1GB）
- [x] 打包 Silver Agibot（126,784 samples, 1,679 shards, 8.1TB）
- [x] 打包 Silver SthV2（20,425 samples, 166 shards, 745.2GB）
- [x] 打包 Silver EPIC（179 samples, 6 shards, 26.4GB）
- [ ] WdsFlowDataset 支持 object flow
- [ ] 多节点 DDP 测试（4+ GPU）
- [ ] 大规模训练 loss 对比验证
- [ ] 清理旧的 pack_parallel_flat / pack_parallel_test 目录
