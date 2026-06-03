# 调试记录:相机上下颠倒 → sceneflow / obj_pose / RGB 全套对齐

本文档总结本次对话从启动可视化脚本一路调到下游 dataloader 转换方案的全过程。
每一节是一个"问题 → 解决"。

---

## 0. 背景

数据集 `datasets_500_fixed/*/camera_data/traj_*/` 是用一台**物理上下颠倒安装**的
相机录制的,导致 RGB、depth、sceneflow、obj_pose 等所有"相机系"或"像素系"
下的量都有 180° 的偏置。需要给可视化和下游 dataloader 提供一套自洽的纠正方案。

涉及主要脚本:
- [vis_sceneflow.py](vis_sceneflow.py) — 3D 点云 + obj_pose viser 可视化
- [vis_rgb_sceneflow.py](vis_rgb_sceneflow.py) — RGB↔sceneflow↔depth↔mask↔mesh 配对校验
- [DATALOADER_ROTATION_RECIPE.md](DATALOADER_ROTATION_RECIPE.md) — 给下游同学的转换文档

---

## 1. 怎么启动 viser 可视化脚本

**问题**:`PYTHONPATH=… python vis_sceneflow.py …` 怎么跑?需要先激活 conda 环境吗?

**解决**:
- 用 `python` 的绝对路径(`miniconda3/envs/metaworld/bin/python`)等价于已经在该 env 里运行,
  通常无需 `conda activate`。
- 但若要走 shell 别名 / 让脚本读到 `CONDA_PREFIX`,推荐:
  ```bash
  source /mnt2/liangzhuowei/miniconda3/etc/profile.d/conda.sh
  conda activate metaworld
  python vis_sceneflow.py <traj_dir> --port 8092 --downsample 8 --flip-flow-yz
  ```
- 远程跑 viser 需要把端口转发: `ssh -L 8092:localhost:8092 user@host`。

---

## 2. sceneflow 三轴同时反转

**问题**:想一次把 sceneflow 的 x/y/z 都反向。

**解决**:在 [vis_sceneflow.py](vis_sceneflow.py) 加 `--flip-xyz` 开关,
内部等于 `--flip-x --flip-y --flip-z`,只对 flow 值生效。

---

## 3. 让 sceneflow 绕 z 轴转 180°

**问题**:画面"水平方向掉头",数学上是 `(x,y,z) → (-x,-y,z)`。

**解决**:加 `--rot-z-180` 开关,本质就是 `--flip-x --flip-y` 的语义糖。
注意:这与"上下颠倒"(绕 x 轴)不同——`--rot-z-180` 是水平掉头,不是上下翻。

---

## 4. obj_pose 也要跟着 sceneflow 转

**问题**:`--rot-z-180` 让 sceneflow 转了,但 obj_pose 没动,两者在 viser 里错位。

**第一次尝试(错)**:把 obj_pose 在**世界系**左乘 `ROT_Z_180`
```python
pose_world = ROT_Z_180 @ cam2world @ pose      # ❌ 错
```
现象:位置不对,因为这是在 world 里转,而 sceneflow 是在 cam 里转。

**核心思路**:
- sceneflow 的翻转发生在 `c2w @ pts_cam` 之前 → **相机系**操作
- obj_pose 是 body→cam 的变换 → 想做"相机系旋转"必须左乘 `ROT_Z_180` 到 pose 上,
  而不是套在 cam2world 外面

**正确写法**:
```python
ROT_Z_180 = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32)
pose_world = cam2world @ ROT_Z_180 @ pose      # ✅
```
对应 `vis_sceneflow.py:255 / 314 / 355`。

**反向逻辑提醒(本次教训)**:相机自己**不要**转——脚本里 `cam2world` 完全不动,
只让"被相机看见的东西"跟着翻,否则等于又把上下颠倒抵消回去了。

---

## 5. 多了一个 `--rot-z-180` 之后,obj_pose 的修正公式怎么推

**问题**:`--flip-flow-yz` 单独存在时 obj_pose 不用改;一旦加 `--rot-z-180`
又得多做一步。怎么推?

**推理**:
- 原始 flow 在 OpenGL/RUB 相机系
- `--flip-flow-yz` 把 flow 变到 OpenCV/RDF 相机系: `*(1,-1,-1)`
- pose 本身就是 OpenCV/RDF 相机系下的 body→cam(meta.json 里写的)
  → 这一步两边已对齐,所以 pose 不用改
- `--rot-z-180` 把 flow 在 OpenCV 相机系里再转 180°: `*(-1,-1,1)`
  → pose 必须在 OpenCV 相机系里同样转 180°: 左乘 `ROT_Z_180`

**结论**:
| 命令 | flow 操作 | obj_pose 操作 |
|---|---|---|
| 仅 `--flip-flow-yz` | `*(1,-1,-1)` | 无需 |
| `--flip-flow-yz --rot-z-180` | `*(-1, 1,-1)` | 左乘 `ROT_Z_180` |

---

## 6. 仅 3D 点云对齐 ≠ RGB 配对对齐

**问题**:`vis_sceneflow.py` 里点云看上去对齐了,但如果下游把 RGB 和
sceneflow 按像素 `(i, j)` 配对用,真的对得上吗?

**关键洞察**:sceneflow `(T, H, W, 3)` 有**两层信息**:
1. **最后一维 3** —— 每个像素的 3D 坐标
2. **`(H, W)` 索引** —— 该 3D 点对应 RGB 的哪个像素

`vis_sceneflow.py` 只用了第 1 件(把每个点当散点扔进 viser),所以看着对齐;
但下游若做"按 `(i, j)` 取 3D 点"或"depth 反投影对配 flow"就会错位。

**解决**:写了一个独立校验脚本 [vis_rgb_sceneflow.py](vis_rgb_sceneflow.py),
用 `rgb[t, i, j]` 给 `flow[t, i, j]` 的 3D 点着色,如果 (u,v) 对齐,点云就"
是一张嵌在 3D 里的彩色照片";否则颜色会乱。

---

## 7. 校验脚本里的三种实验

```bash
# 0) 基线:不动任何东西
python vis_rgb_sceneflow.py <traj_dir> --port 8093

# 1) 错误演示:只把 RGB 转 180°,sceneflow 不动
python vis_rgb_sceneflow.py <traj_dir> --port 8093 --rot-rgb-180
# 现象: flow 着色错位,depth 点云和 flow 点云偏移

# 2) 正确做法: RGB + sceneflow (H,W) 都转,flow 值同步翻
python vis_rgb_sceneflow.py <traj_dir> --port 8093 --rot-rgb-180 --rot-flow-hw-180
# 现象: 全套重合
```

---

## 8. `np.rot90` 的 axes 怎么选

**问题**:axes 参数到底写 `(0,1)` 还是 `(1,2)`?

**解释**:`axes=(a,b)` 表示在第 a、b 维构成的 2D 平面里转。要旋转"图像平面",
就锁定 H 和 W 两维。
- **批量 `(T, H, W, …)`**: H=1, W=2 → `axes=(1, 2)`
- **单帧 `(H, W, …)`**: H=0, W=1 → `axes=(0, 1)`

**万能写法(推荐给下游)**:
```python
np.rot90(arr, k=2, axes=(-3, -2))   # 锚定到倒数第 3/2 维,自动兼容两种 shape
```

---

## 9. 完整 dataloader recipe(给下游同学)

详见 [DATALOADER_ROTATION_RECIPE.md](DATALOADER_ROTATION_RECIPE.md)。要点:

| 数据 | 操作 |
|---|---|
| rgb / depth / seg / mask | `np.rot90(k=2, axes=(1,2))` |
| sceneflow `(H,W)` | `np.rot90(k=2, axes=(1,2))` |
| sceneflow 值 | `*(-1, 1, -1)` |
| K 内参 | `cx → W-1-cx, cy → H-1-cy`;fx, fy 不动 |
| obj_pose | 左乘 `ROT_Z_180 = diag(-1,-1,1,1)` |
| cam2world / cam_poses / mesh | 不动 |

---

## 10. `--full` 全套端到端校验

**问题**:校验脚本一开始只看 RGB 和 sceneflow,没看 seg / mask / obj_pose / mesh。

**解决**:[vis_rgb_sceneflow.py](vis_rgb_sceneflow.py) 加 `--full` 开关:
- 加载 seg.b2nd / target_obj_mask / pose_*.npy / mesh_*.ply
- `--rot-rgb-180` 时连带 seg / mask 一起 `rot90`
- `--rot-flow-hw-180` 时 obj_pose 自动左乘 `ROT_Z_180`
- viser GUI 加 Color mode 下拉(rgb / seg / target)、Show obj_pose / mesh checkbox

```bash
python vis_rgb_sceneflow.py <traj_dir> --port 8093 \
    --rot-rgb-180 --rot-flow-hw-180 --full
```

如果全套 recipe 自洽:三种 color mode 都对齐 + mesh 贴在物体上 + obj 坐标轴
指向物体几何中心。

---

## 11. "viser 里那张照片样的东西是什么"

**问题**:校验脚本截图里中间有块"像照片一样飘在 3D 里"的彩色 patch,
担心是没旋转的原图。

**解释**:viser 里**没有**单独展示原始 RGB 图像。那块"照片"其实是
**sceneflow 点云本身**,由于:
- 每个像素一个 3D 点 → 形成 H×W 稠密网格
- 每个点按 `rgb[i, j]` 着色 → 颜色和真实场景一致
- 从靠近相机视角看 → 网格平面接近正对,看起来像 2D 贴图

绕一下视角就能看出它其实是带厚度的 3D 重建(桌面、机械臂、地面都有起伏)。
框外的灰色散点是 depth back-projection 点云,两者重合就是"recipe 自洽"
的直接证据。

---

## 12. mask 怎么单独可视化

**问题**:`--full` 模式默认把 mask 只用作"色彩模式之一",不够直观。

**解决**:[vis_rgb_sceneflow.py](vis_rgb_sceneflow.py) 再加两个图层:
- **`Show target_mask pc`** —— 只把 `mask>0` 的 flow 点单独画一份**红色加大点云**,
  在原本点云上叠红色高亮(看 3D 位置)
- **`Show target_mask 2D image`** —— 把当前帧 mask 当 2D 图片贴在世界原点附近
  (红=mask, 灰=非mask),用来对照像素布局

如果 recipe 全对:红色点云精准贴在目标物体上,2D mask 形状和点云红色区域形状一致。

---

## 关键概念速查

- **"相机系操作" vs "世界系操作"**:左乘还是不乘 cam2world 决定。
  - 相机系: 直接对 `pose_body2cam` 左乘
  - 世界系: 对 `cam2world @ pose` 左乘 (或 `cam2world` 左乘)
- **sceneflow 的两层信息**:值(3D) + (H,W)排布(对应像素)。任一项不处理,
  就和某种下游用法对不上。
- **`np.rot90` 返回 view**:务必跟一个 `.copy()`,否则后续 in-place 操作会失败。
- **图像旋转 ↔ K 主点翻转**:必须成对出现,fx/fy 不变。
- **mesh 不动**:mesh 顶点在 body-local 坐标系下,所有旋转都已经"翻译"到 pose 上了。
