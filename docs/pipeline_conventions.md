# Pipeline Conventions

This note summarizes the data conventions used by the rollout-to-point-cloud pipeline, and the small conversion ideas discussed in this session.

## 1. Coordinate system

The whole synthetic-data pipeline is consistent with an **OpenGL-style camera frame**:

- X right
- Y up
- Z back
- forward direction is `-Z`

That convention is used by:

- `cam_poses.npy`
- depth back-projection
- anchor point clouds
- sceneflow tracking
- `camera_position` / `camera_quaternion` in H5

## 2. Camera intrinsics / extrinsics

### `cam_poses.npy`

`cam_poses.npy` stores per-frame **camera-to-world** transforms:

- shape: `(T, 4, 4)`
- dtype: `float32`
- last row is always `[0, 0, 0, 1]`
- for fixed cameras, all frames are identical

Interpretation:

```python
p_world = cam_pose @ [p_cam; 1]
```

### `cam_intrinsics.npy`

`cam_intrinsics.npy` stores the pinhole matrix `K`:

- shape: `(3, 3)` in Metaworld
- shape: `(T, 3, 3)` in MIKASA-Robo
- `fx = K[0, 0]`, `fy = K[1, 1]`
- `cx = K[0, 2]`, `cy = K[1, 2]`

For the OpenGL camera convention, the depth back-projection is:

```python
X = (u - cx) * z / fx
Y = (cy - v) * z / fy
Z = -z
```

where `z` is the positive metric depth along the viewing ray.

## 3. Two-key camera pose split

A useful abstraction is to split camera pose into two keys:

### Key 1: relative camera motion

`cam_t -> cam_0`

- shape: `(T, 4, 4)`
- `cam_poses_rel[0] = I`
- works for both synthetic and real data

### Key 2: initial camera pose in world

`cam_0 -> world`

- shape: `(4, 4)`
- synthetic-only
- anchors the camera trajectory to the simulator world

Reconstruction:

```python
cam_poses[t] = cam0_to_world @ cam_poses_rel[t]
```

## 4. Object pose and mesh conventions

### Body pose

`pose_<obj>.npy` stores **body-to-world** transforms:

- shape: `(T, 4, 4)`
- `p_world = pose @ [p_body; 1]`
- quaternion order is `w, x, y, z`

### Mesh vertices

`mesh_<obj>.ply` stores vertices in the **body-local frame**.

That means a vertex can be lifted into world space with the matching body pose:

```python
p_world = pose_<obj>[t] @ [v_local; 1]
```

## 5. Scene flow convention

Scene flow is stored in **camera coordinates**, not world coordinates.

- anchor point cloud: `scene_point_flow_refXXXXX.anchor.npy`
- tracked flow: `scene_point_flow_refXXXXX.npy`

Both follow the OpenGL camera frame:

- X right
- Y up
- Z back
- forward is `-Z`

So the stored points are not world-space XYZ; they are camera-space XYZ for each frame.

## 6. Segmentation and pixel-to-id mapping

- background is `0`
- foreground pixels store MuJoCo / SAPIEN body ids
- pixel-to-sid mapping is taken from the segmentation frame at the anchor index

This matters because one anchor frame may see a different subset of actors than frame 0.

## 7. Metaworld vs MIKASA-Robo

The coordinate convention is the same in both datasets.

Main differences:

- Metaworld often uses a single `cam_intrinsics.npy` matrix
- MIKASA-Robo may store per-frame intrinsics
- MIKASA-Robo H5 files store camera/body pose fields directly inside `id_poses`
- MIKASA-Robo actor naming differs from Metaworld body naming

## 8. OpenGL to OpenCV adapter

If a downstream consumer expects OpenCV-style camera coordinates, the cheapest option is to convert at load time only.

OpenGL -> OpenCV flip:

```python
flip = np.array([1, -1, -1], dtype=np.float32)
pts_cv = pts_gl * flip
```

For camera poses, the same idea applies by flipping the camera axes in memory.

This keeps the files unchanged and avoids regenerating the dataset.

## 9. Practical takeaway

The current pipeline is internally self-consistent:

- simulator output
- camera intrinsics/extrinsics
- depth back-projection
- sceneflow tracking
- body pose export
- mesh export

All follow the same OpenGL-style convention.

If you need OpenCV-style tensors for training or evaluation, do the conversion in the dataloader, not on disk.
