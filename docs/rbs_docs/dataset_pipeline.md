# Metaworld Dataset Pipeline

这份文档描述从仿真录制到 WebDataset 打包的完整数据处理流程，涵盖坐标约定、文件格式以及每步所用脚本。

---

## 环境准备

```bash
source /mnt2/liangzhuowei/miniconda3/bin/activate metaworld
cd /mnt2/liangzhuowei/Metaworld
```

---

## 完整流程总览

```
Step 0  环境激活
Step 1  rollout_scripted_policy.py        → rollout_data/<task>/
Step 2  replay_record_trajectories.py     → dataset/<task>/camera_data/traj_N/
Step 3  convert_camera_depths.py          → scene_point_flow_ref*.npy (中间文件)
Step 4a flow_compress.py compress         → scene_point_flow_ref*.{mp4,json}
Step 4b point_compress.py --mode compress → depth_video_int16mm_dt.b2nd
Step 4c seg_compress.py compress          → seg.b2nd
Step 5  build_mikasa_format.py            → cam2world_*.npy / pose_*.npy /
                                            mesh_*.ply / mask_*.npz /
                                            target_obj_mask.mp4 / meta.json
Step 6  pack_shards.py --dataset metaworld → shard-*.tar / index.json
```

---

## Step 1：Rollout 策略录制

使用内置脚本策略采集轨迹，输出 H5 + JSON 格式的状态序列。

```bash
python rbs_sceneflow_scripts/rollout_scripted_policy.py \
    --env-name hammer-v3 \
    --num-episodes 500 \
    --output-dir rollout_data
```

| 参数 | 说明 |
|---|---|
| `--env-name` | 任务名，如 `hammer-v3`（见下方任务列表） |
| `--num-episodes N` | 每个环境采集 N 条轨迹 |
| `--all-envs` | 对所有有脚本策略的任务批量执行 |
| `--seed` | 随机种子（默认 0） |
| `--no-stop-on-success` | 成功后继续走满 max-steps |

**输出：**

```
rollout_data/hammer-v3/
    trajectory.state.mocap_xyz.mujoco_cpu.h5    # 轨迹状态序列（qpos/qvel/actions）
    trajectory.state.mocap_xyz.mujoco_cpu.json  # 元数据（episode 数、success 标志等）
```

---

## Step 2：Replay + 录制相机数据

将 Step 1 的轨迹在仿真中重放，同时录制 RGB / Depth / Segmentation / 位姿。

```bash
python rbs_sceneflow_scripts/replay_record_trajectories.py \
    --h5   rollout_data/hammer-v3/trajectory.state.mocap_xyz.mujoco_cpu.h5 \
    --json rollout_data/hammer-v3/trajectory.state.mocap_xyz.mujoco_cpu.json \
    --output-dir dataset/hammer-v3 \
    --all-trajs --success-only \
    --camera corner \
    --width 832 --height 480
```

| 参数 | 说明 |
|---|---|
| `--all-trajs` | 处理所有轨迹（不加则需指定 `--traj-id N`） |
| `--success-only` | 只处理成功轨迹 |
| `--camera` | 摄像头视角（见下方视角说明） |
| `--random-camera` | 每条轨迹随机选一个视角（用 `--cameras` 指定候选池） |
| `--multi-cameras A B` | 同一次 replay 录制多个视角，各自输出到子目录 |
| `--width / --height` | 分辨率，默认 832×480 |
| `--num-procs N` | 并行 replay 进程数 |

**输出：**

```
dataset/hammer-v3/
    trajectory.rgb+depth+segmentation.*.h5     # replay 时所有相机的合并 h5
    camera_data/
        traj_0/
            rgb.mp4                             # RGB 视频
            depth_video.npy                     # float16 深度 (T,H,W) [m]（中间文件）
            seg.npy                             # int32 body-id segmentation（中间文件）
            cam_poses.npy                       # (T,4,4) cam-to-world OpenGL absolute
            cam_intrinsics.npy                  # (3,3) 内参矩阵
            camera_name.txt                     # 所用摄像头名称
            traj_0.h5                           # 含 id_poses（每 body 世界/相机坐标）
            traj_task.json                      # actors 列表（seg_id → body name）
```

### 预设摄像头视角

| 名称 | 位置 (x,y,z) | 特点 |
|---|---|---|
| `corner` | (-1.1, -0.4, 0.6) | 侧前方斜视，**默认** |
| `topview` | (0, 0.5, 1.5) | 俯视 |
| `corner2` | (1.3, -0.2, 1.1) | 右侧方向，fovy=60 |
| `corner3` | (0.9, 0, 1.5) | 高角度右侧，fovy=45 |
| `corner4` | (0.75, 0.075, 0.7) | 近距右侧，fovy=60 |

---

## Step 3：深度图 → SceneFlow（中间文件）

将每帧深度图反投影为相机坐标系下的 3D 点云，并计算帧间场景流。

```bash
python rbs_sceneflow_scripts/traj2sceneflow/convert_camera_depths.py \
    dataset/hammer-v3/camera_data \
    --workers 8
```

| 参数 | 说明 |
|---|---|
| `root` | `camera_data/` 根目录，递归处理所有 `traj_N/` |
| `--workers N` | 并行进程数（默认 1） |

**输出（中间文件，Step 4a 后可删）：**

```
traj_N/
    scene_point_flow_ref00000.npy   # (T,H,W,3) float32 场景流，OpenGL 相机坐标
    scene_point_flow_ref00021.npy   # 每隔约 T/5 帧取一个参考帧
    ...
```

---

## Step 4：压缩

### Step 4a：压缩 SceneFlow → mp4

```bash
python rbs_sceneflow_scripts/traj2sceneflow/flow_compress.py \
    compress \
    --out_root dataset/hammer-v3/camera_data \
    --codec libx265 --bits 10 --crf 0 \
    --delete_npy
```

| 参数 | 说明 |
|---|---|
| `--out_root` | 根目录（含多个 `traj_N/`）；单目录用 `--out_dir` |
| `--codec` | `libx265`（默认）/ `ffv1` / `libvpx-vp9` |
| `--crf 0` | 近无损，推荐默认 |
| `--bits 10` | 10-bit 量化，推荐默认 |
| `--delete_npy` | 压缩后删除原始 `.npy`（节省磁盘） |

**输出：**

```
traj_N/
    scene_point_flow_ref00000_v3_10b_h265_crf0.mp4     # 压缩场景流视频
    scene_point_flow_ref00000_v3_10b_h265_crf0.json    # 编码参数 sidecar
    scene_point_flow_ref00000.anchor.npy               # 参考帧点云（解码用）
```

### Step 4b：压缩 Depth → Blosc2

```bash
python rbs_sceneflow_scripts/traj2sceneflow/point_compress.py \
    --mode compress \
    --root dataset/hammer-v3/camera_data \
    --delete-existing
```

| 参数 | 说明 |
|---|---|
| `--mode compress` | 压缩模式 |
| `--root` | 递归处理该目录下所有含 `depth_video.npy` 的子目录 |
| `--seg_dir` | 只处理单个目录 |
| `--mm-step-mm` | 量化步长（默认 5mm，越大压缩比越高，精度越低） |

**输出：**

```
traj_N/
    depth_video_int16mm_dt.b2nd        # int16 mm Δt 差分 + Blosc2 压缩
    depth_video_int16mm_dt.meta.json   # 解压所需元数据
```

### Step 4c：压缩 Segmentation → Blosc2

```bash
python rbs_sceneflow_scripts/traj2sceneflow/seg_compress.py \
    compress \
    --root dataset/hammer-v3/camera_data \
    --method b2nd \
    --delete-source
```

| 参数 | 说明 |
|---|---|
| `--root` | 递归处理所有 `traj_N/` 中的 `seg.npy` |
| `--seg-dir` | 只处理单个目录 |
| `--method b2nd` | Blosc2 格式（默认，推荐） |
| `--delete-source` | 压缩后删除原始 `seg.npy` |

**输出：**

```
traj_N/
    seg.b2nd           # Blosc2 压缩的 int32 segmentation
    seg.b2nd.meta.json
```

---

## Step 5：生成派生文件（位姿 / 网格 / Mask / meta.json）

将 Step 2-4 的原始数据转换为 MIKASA 兼容格式。**一步完成所有派生文件。**

```bash
# 单个 traj
python rbs_sceneflow_scripts/build_mikasa_format.py \
    --traj-dir dataset/hammer-v3/camera_data/traj_0

# 单个 task（所有 traj）
python rbs_sceneflow_scripts/build_mikasa_format.py \
    --task-dir dataset/hammer-v3

# 全量数据集（并行，推荐）
python rbs_sceneflow_scripts/build_mikasa_format.py \
    --root dataset \
    -j 8
```

| 参数 | 说明 |
|---|---|
| `--root` | 数据集根目录（含多个 `<task>/camera_data/traj_N/`） |
| `--task-dir` | 单个任务目录 |
| `--traj-dir` | 单个轨迹目录 |
| `-j N` | 并行 task 数（每个 task 独立进程，共享 env model） |
| `--overwrite` | 覆盖已存在的输出文件 |

**输出（写入每个 `traj_N/`）：**

```
traj_N/
    cam2world_cv.npy        # (T,4,4) OpenCV cam-to-world absolute
    cam2world_gl.npy        # (T,4,4) OpenGL cam-to-world absolute
    cam_poses.npy           # (T,4,4) OpenCV relative to cam0（覆写原始值）
    pose_<obj>.npy          # (T,4,4) body→cam OpenCV，每个目标物体一份
    mesh_<obj>.ply          # body-local PLY 网格，每个目标物体一份
    mask_<obj>.npz          # (T,H,W) uint8 {0,255} 目标物体二值 mask
    target_obj_mask.mp4     # 同上，mp4 格式
    meta.json               # 完整元数据（见字段规范）
```

---

## Step 6：打包为 WebDataset

```bash
python /mnt2/liangzhuowei/rbs-data-utils/src/wbs_utils/pack_shards.py \
    --dataset metaworld \
    --data-root dataset \
    --output <webdataset_output_dir>
```

每个 `shard-XXXXXX.tar` 包含（per sample）：`rgb.mp4` · `flow_ref*.mp4` · `flow_ref*.anchor.npy` · `depth.b2nd` · `seg.b2nd` · `cam_poses.npy` · `cam2world_cv.npy` · `cam_intrinsics.npy` · `traj.h5` · `pose_<obj>.npy` · `mesh_<obj>.ply` · `mask_<obj>.npz` · `object_mask.mp4` · `meta.json`

输出目录还包含 `index.json`（shard 列表 + 计数）。

---

## 目录结构

```
datasets_500_fixed/
└── <task>/
    └── camera_data/
        └── traj_N/
            ├── cam_poses.npy                   # 原始录制：OpenGL cam-to-world (absolute)
            ├── cam2world_cv.npy                # ✅ 生成：OpenCV cam-to-world (absolute)
            ├── cam2world_gl.npy                # ✅ 生成：OpenGL cam-to-world (absolute, 同 cam_poses 原始值)
            ├── cam_poses.npy                   # ✅ 覆写：OpenCV, relative to cam0 (frame-0 = Identity)
            ├── cam_intrinsics.npy              # 内参矩阵 (3×3)
            ├── pose_<obj>.npy                  # ✅ 生成：body→cam (T,4,4) float32 OpenCV
            ├── mesh_<obj>.ply                  # ✅ 生成：body-local PLY (binary little-endian)
            ├── mask_<obj>.npz                  # ✅ 生成：(T,H,W) uint8 {0,255}
            ├── target_obj_mask.mp4             # ✅ 生成：binary mask 视频 (libx264 mono)
            ├── meta.json                       # ✅ 生成/覆写：完整规范字段
            ├── rgb.mp4                         # 原始录制
            ├── depth_video_int16mm_dt.b2nd     # 原始录制
            ├── seg.b2nd                        # 原始录制：body-id segmentation
            ├── scene_point_flow_ref*.mp4       # 原始录制：compressed sceneflow
            ├── scene_point_flow_ref*.anchor.npy# 原始录制：sceneflow anchor (OpenGL)
            └── traj_N.h5                       # 原始录制：id_poses, env_states, actions
```

---

## 坐标约定

| 约定 | 轴方向 | 用途 |
|---|---|---|
| **OpenGL / RUB** | X右 Y上 Z后，相机看 -Z | 仿真内部、h5 `camera_*`、sceneflow、anchor |
| **OpenCV / RDF** | X右 Y下 Z前，相机看 +Z | 所有输出位姿文件、cam2world_cv |

转换公式（列向量约定）：

```
cam2world_cv  = cam2world_gl  @ FLIP4          # FLIP4 = diag(1,-1,-1,1)
body→cam_cv   = FLIP4         @ body→cam_gl    # 左乘
点坐标         p_cv = p_gl * [1,-1,-1]         # FLIP3
```

sceneflow（`scene_point_flow_ref*.mp4` + `*.anchor.npy`）**保持 OpenGL 不变**，由 dataloader 在运行时做 FLIP3 转换。

---

## 流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│  原始录制 (replay_record_trajectories.py)                            │
│                                                                     │
│  cam_poses.npy       OpenGL absolute cam-to-world                   │
│  traj_N.h5           id_poses/<bid>/{camera_position,               │
│                        camera_quaternion}  ← OpenGL body→cam        │
│  rgb/depth/seg/sceneflow                                            │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼  rbs_sceneflow_scripts/build_mikasa_format.py
┌─────────────────────────────────────────────────────────────────────┐
│  build_mikasa_format.py  (单步完成，50 tasks × 50 trajs，≈4 min)     │
│                                                                     │
│  相机位姿                                                            │
│  ├── cam2world_gl.npy   = cam_poses.npy (原始 OpenGL)               │
│  ├── cam2world_cv.npy   = cam2world_gl @ FLIP4                      │
│  └── cam_poses.npy      = inv(cam2world_cv[0]) @ cam2world_cv       │
│                           (覆写为 OpenCV relative-to-cam0)           │
│                                                                     │
│  目标物体位姿                                                        │
│  └── pose_<obj>.npy     = FLIP4 @ T_gl   (从 h5 camera_* 读取)     │
│                           (T,4,4) body→cam OpenCV                   │
│                                                                     │
│  目标物体网格 (extract_meshes.collect_target_mesh)                   │
│  └── mesh_<obj>.ply     body-local PLY; p_cam = pose[t] @ [v;1]    │
│                                                                     │
│  目标物体 Mask                                                       │
│  ├── mask_<obj>.npz     (T,H,W) uint8 {0,255}, 压缩                 │
│  └── target_obj_mask.mp4 libx264 mono, 同 rgb.mp4 fps              │
│                                                                     │
│  meta.json              见下方字段说明                               │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼  pack_shards.py --dataset metaworld
┌─────────────────────────────────────────────────────────────────────┐
│  WebDataset 打包                                                     │
│                                                                     │
│  python .../rbs-data-utils/src/wbs_utils/pack_shards.py \           │
│      --dataset metaworld \                                          │
│      --data-root datasets_500_fixed \                               │
│      --output <webdataset_output>/<task>/                           │
│                                                                     │
│  每个 shard-XXXXXX.tar 包含（per sample）：                          │
│    rgb.mp4  depth.b2nd  seg.b2nd                                    │
│    flow_ref*.mp4  flow_ref*.anchor.npy  flow_ref*.sidecar.json      │
│    cam_poses.npy  cam2world_cv.npy  cam2world_gl.npy                │
│    cam_intrinsics.npy  traj.h5                                      │
│    pose_<obj>.npy  mesh_<obj>.ply  mask_<obj>.npz                   │
│    object_mask.mp4  meta.json                                       │
│  + index.json (shard 列表 + 计数)                                    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  下游 World Model 训练 / 推理                                        │
│  WebDataset DataLoader → object-centric world model                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## meta.json 字段规范

```json
{
  "task_id": "<task-name>",
  "traj_name": "traj_N",
  "num_frames": 84,
  "actors": [{"seg_id": 1, "name": "body:<name>"}, ...],

  "coordinate_convention": {
    "sceneflow":     "opengl_camera",
    "anchor_points": "opengl_camera_ref_frame",
    "flow_vectors":  "opengl_camera_ref_frame"
  },
  "camera_convention":  "opencv",
  "flow_convention":    "opengl",
  "camera_pose_layout": "cam2world absolute; cam_poses relative to cam0",
  "cam2world_file":     "cam2world_cv.npy",
  "cam2world_gl_file":  "cam2world_gl.npy",
  "cam_poses_file":     "cam_poses.npy",
  "pose_layout":        "body->cam",

  "target_object": {
    "body_names": ["RoundNut", "asmbly_peg"],
    "seg_ids":    [33, 34]
  },
  "body_poses": {
    "format": "(T,4,4) float32 body-to-camera homogeneous transform in OpenCV camera frame",
    "files":  {"RoundNut": "pose_RoundNut.npy", ...}
  },
  "meshes": {
    "format": "binary_little_endian PLY; vertices are in body-local frame (p_cam_h = T_body2cam[t] @ [v_body; 1]).",
    "files":  {"RoundNut": "mesh_RoundNut.ply", ...},
    "seg_ids": {"RoundNut": 33, ...},
    "num_vertices": {...}, "num_faces": {...}
  },
  "target_obj_mask": {
    "npz_file": "mask_RoundNut.npz",
    "mp4_file": "target_obj_mask.mp4",
    "format":   "npz_uint8_binary + mp4_h264_mono",
    "binary_values": [0, 255],
    "num_frames": 84, "height": 480, "width": 832, "fps": 16.0
  },
  "cam_intrinsics_file": "cam_intrinsics.npy",
  "cam_intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]]
}
```

与 mikasa 格式的差异：

| 字段 | mikasa | metaworld (本数据集) |
|---|---|---|
| `cam2world_file` | `cam2world.npy` | `cam2world_cv.npy`（额外保留 `cam2world_gl.npy`） |
| `coordinate_convention` | 无 | 有（sceneflow/anchor/flow 三段） |
| `target_obj_mask` 字段名 | `file` | `npz_file` + `mp4_file` + `fps` |
| `body_poses.seg_ids` | 有 | 无（在 `meshes.seg_ids` 里有） |

---

## 脚本索引

| 脚本 | 功能 |
|---|---|
| [rbs_sceneflow_scripts/build_mikasa_format.py](../../rbs_sceneflow_scripts/build_mikasa_format.py) | 一步生成所有派生文件和 meta.json（**主入口**） |
| [rbs_sceneflow_scripts/extract_meshes.py](../../rbs_sceneflow_scripts/extract_meshes.py) | PLY 网格提取（被 build_mikasa_format 调用） |
| [rbs_sceneflow_scripts/generate_target_obj_mask.py](../../rbs_sceneflow_scripts/generate_target_obj_mask.py) | 独立生成 mask npz（单独使用时） |
| [rbs_sceneflow_scripts/target_objects.json](../../rbs_sceneflow_scripts/target_objects.json) | 每个 task 的目标 body 名称映射表 |
| [rbs_sceneflow_scripts/build_pose_variants.py](../../rbs_sceneflow_scripts/build_pose_variants.py) | 从已有 body-world pose 生成 cam 位姿变体 |
| [rbs_sceneflow_scripts/vis_sceneflow.py](../../rbs_sceneflow_scripts/vis_sceneflow.py) | Viser 可视化（`--auto` 自动处理 GL→CV 转换） |

---

## 快速命令

```bash
# 处理单个 traj
python rbs_sceneflow_scripts/build_mikasa_format.py \
    --traj-dir datasets_500_fixed/assembly-v3/camera_data/traj_0

# 处理单个 task
python rbs_sceneflow_scripts/build_mikasa_format.py \
    --task-dir datasets_500_fixed/assembly-v3

# 处理全量数据集（50 tasks × 50 trajs，8 进程并行，≈4 min）
python rbs_sceneflow_scripts/build_mikasa_format.py \
    --root datasets_500_fixed \
    -j 8 --overwrite

# WebDataset 打包
python /mnt2/liangzhuowei/rbs-data-utils/src/wbs_utils/pack_shards.py \
    --dataset metaworld \
    --data-root datasets_500_fixed \
    --output <output_dir>

# 可视化验证（OpenCV 数据，OpenGL sceneflow 自动转换）
python rbs_sceneflow_scripts/vis_sceneflow.py \
    datasets_500_fixed/assembly-v3/camera_data/traj_0 \
    --auto
```
