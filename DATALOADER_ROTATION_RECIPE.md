# Dataloader Rotation Recipe (相机上下颠倒修正)

数据集 `datasets_500_fixed/*/camera_data/traj_*/` 的相机是上下颠倒的。
下游 dataloader 在读取原始数据后,需要做一次 **整体 180° 旋转修正**,让
RGB / depth / seg / mask / sceneflow / obj_pose 全部对齐到同一个"正向"
坐标。本文件给出严格的转换公式,直接复制到 dataloader 即可使用。


如果所有数据都按本文档处理,该命令会显示 RGB 点云、depth 点云、obj_pose
坐标轴、mesh 完全重合并贴在场景里。

---

## 命名约定

```python
import numpy as np

# 形状常量(以 480x832 为例,实际从数据本身读)
T, H, W = num_frames, 480, 832

# 全局两个常量
FLIP3     = np.array([1.0, -1.0, -1.0], dtype=np.float32)   # OpenGL -> OpenCV cam frame
ROT_Z_180 = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32)
```

---

## 1. 像素阵列类:`np.rot90(k=2)`

涉及所有 (T, H, W, …) 形状的"图像"类数据。等价于"把每张图上下颠倒"。

```python
# rgb.mp4              -> (T, H, W, 3) uint8
rgb = np.rot90(rgb, k=2, axes=(1, 2)).copy()

# depth_video_int16mm_dt.b2nd  -> (T, H, W) int16 mm
depth = np.rot90(depth, k=2, axes=(1, 2)).copy()

# seg.b2nd             -> (T, H, W) int
seg = np.rot90(seg, k=2, axes=(1, 2)).copy()

# target_obj_mask.mp4 / mask_*.npz  -> (T, H, W) uint8
mask = np.rot90(mask, k=2, axes=(1, 2)).copy()
```

> ⚠️ 单帧操作时 axes 写 `(0, 1)`,批量 `(T, H, W, …)` 写 `(1, 2)`。

---

## 2. Sceneflow:同时旋转 (H, W) **和** 翻转 XYZ 数值

`scene_point_flow_ref*.{anchor.npy, mp4, json}` 解码得到的 flow 形状 `(T, H, W, 3)`:
- `(H, W)` 是像素索引,必须和 RGB 同步旋转
- 最后一维 `(x, y, z)` 是相机系下 3D 坐标,需要翻转把上下颠倒纠正过来

```python
# Step A: pixel-plane rotation (与 RGB 配对)
flow = np.rot90(flow, k=2, axes=(1, 2)).copy()

# Step B: value flip (RUB->RDF 后再绕 z 旋转 180°,合并写成一次乘法)
flow = flow * np.array([-1.0, 1.0, -1.0], dtype=np.float32)
```

> 解释: `--flip-flow-yz` 把 OpenGL/RUB 相机系转 OpenCV/RDF: `*(1,-1,-1)`;
> `--rot-z-180` 再绕 z 转 180°: `*(-1,-1,1)`。两者相乘 = `(-1, 1, -1)`。

---

## 3. 相机内参 `cam_intrinsics.npy`:翻 cx/cy

像素做了 `rot90(k=2)`,主点必须同步翻;焦距不动。

```python
K = K.copy()
K[0, 2] = W - 1 - K[0, 2]   # cx
K[1, 2] = H - 1 - K[1, 2]   # cy
# fx (K[0,0]) 和 fy (K[1,1]) 保持不变
```

---

## 4. 物体位姿 `pose_*.npy`:左乘 `ROT_Z_180`

`pose_*.npy` 是 `(T, 4, 4)` 的 body→cam 变换(OpenCV 相机系)。
sceneflow 在相机系下绕 z 转了 180°,obj_pose 必须做同样的相机系旋转,
所以**左乘**:

```python
ROT_Z_180 = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32)

# pose: (T, 4, 4)
pose = (ROT_Z_180[None] @ pose).astype(np.float32)
# 等价: pose[:, :2, :] *= -1
```

---

## 5. **保持不变**的文件

| 文件 | 处理 |
|---|---|
| `cam2world.npy` / `cam2world_cv.npy` / `cam2world_gl.npy` | ❌ 不动 |
| `cam_poses.npy` | ❌ 不动 |
| `mesh_*.ply` | ❌ 不动(顶点是 body-local,旋转已在 pose 里) |
| `K` 的 `fx, fy` | ❌ 不动 |
| `meta.json` / `camera_name.txt` | ❌ 元数据 |

---

## 一站式 dataloader 代码模板

```python
import numpy as np
from pathlib import Path

FLIP3     = np.array([1.0, -1.0, -1.0], dtype=np.float32)
ROT_Z_180 = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32)

def load_traj(traj_dir: Path):
    traj_dir = Path(traj_dir)

    # --- raw load ---
    rgb   = load_rgb_video(traj_dir / "rgb.mp4")              # (T,H,W,3) uint8
    depth = load_depth_b2nd(traj_dir / "depth_video_int16mm_dt.b2nd")  # (T,H,W)
    seg   = load_seg_b2nd(traj_dir / "seg.b2nd")              # (T,H,W) int
    mask  = load_video_gray(traj_dir / "target_obj_mask.mp4") # (T,H,W) uint8
    flow  = load_first_flow(traj_dir)                         # (T,H,W,3) float
    K     = np.load(traj_dir / "cam_intrinsics.npy").astype(np.float32)
    c2w   = np.load(traj_dir / "cam2world_cv.npy").astype(np.float32)
    pose  = np.load(traj_dir / "pose_<name>.npy").astype(np.float32)  # (T,4,4)

    T, H, W = rgb.shape[:3]

    # --- rotation recipe ---
    # 1) image-plane rot90 k=2 for RGB / depth / seg / mask
    rgb   = np.rot90(rgb,   k=2, axes=(1, 2)).copy()
    depth = np.rot90(depth, k=2, axes=(1, 2)).copy()
    seg   = np.rot90(seg,   k=2, axes=(1, 2)).copy()
    mask  = np.rot90(mask,  k=2, axes=(1, 2)).copy()

    # 2) sceneflow: rot90 (H,W) + flip values (-1, 1, -1)
    flow = np.rot90(flow, k=2, axes=(1, 2)).copy()
    flow = flow * np.array([-1.0, 1.0, -1.0], dtype=np.float32)

    # 3) intrinsics: flip cx/cy, keep fx/fy
    K = K.copy()
    K[0, 2] = W - 1 - K[0, 2]
    K[1, 2] = H - 1 - K[1, 2]

    # 4) obj_pose: left-multiply ROT_Z_180
    pose = (ROT_Z_180[None] @ pose).astype(np.float32)

    # 5) cam2world / mesh untouched
    return dict(rgb=rgb, depth=depth, seg=seg, mask=mask,
                flow=flow, K=K, cam2world=c2w, pose=pose)
```

---

## 校验清单

加载后用以下三件事任一确认 dataloader 正确:

1. **像素↔3D**:对任意像素 `(i, j)`,`flow[t, i, j]` 应该是该像素在相机系
   下的 3D 坐标;用 `K` 投影回去应落在 `(j, i)` 附近(像素误差 < 1)。
2. **depth↔flow**:用 `K` 反投影 depth 得到的点云,应该与 `flow` 的点云
   在同一相机系下重合。
3. **obj_pose↔mesh**:`cam2world[t] @ pose[t]` 把 mesh 顶点变到世界系,
   应该和 flow 点云在世界系下贴合(且 obj 部分的 seg id / target mask
   也对应同一区域的 flow 点)。


---

## 常见坑

- **忘记 `.copy()`**:`np.rot90` 返回 view,后续 in-place 改会出问题。
- **axes 写错**:`(T, H, W, C)` 阵列必须 `axes=(1, 2)`,写成 `(0, 1)` 会
  把时间维和高度维互换。
- **只转图像不转 sceneflow**:RGB 颜色看着对,但 `(i, j)` 取出的 3D 点
  会指向错误位置。
- **只翻 flow 值不旋转 (H, W)**:点云形状是正的,但和 RGB / seg / mask
  对不上。
- **改了 cx/cy 又重复改一次**:多次调用 dataloader 时务必保证 `K` 是从
  原始文件重新加载,不要在内存里反复改。
- **左乘 vs 右乘 ROT_Z_180**:obj_pose 是 body→cam,要在 **cam 系** 里
  旋转,所以**左乘**;右乘会变成在 body 系里旋转,结果错位。
