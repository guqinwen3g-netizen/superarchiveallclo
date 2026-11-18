# 只有服装深度时的平铺归档及逐射线对照

以 IRI UPC-CSIC 的蓝色圆领 T 恤为样本，保留输入帧 `n00700` 的原彩照和真实衣物深度，并在实验输入中移除衣物以外的深度。用另一帧 `n00764` 的台面像素提供平面和尺度。后者的衣物像素留作对照。只使用前片影像，适用于核对这一条制板通道。两帧的同件依据是序列、印花与结构，原发布资料缺少独立的实物编号。[IRI 来源、量法和配对依据](iri-pcd-study.md)

自动从有台面像素的输入求平面，以及使用外部台面给遮去背景的输入定平面，两次都得到 840 × 704 mm 的衣物投影与 520 mm 的下部衣身宽度。外部台面本身的拟合残差 P90 为 4.335 mm。归档回读了 5 片毫米 DXF、15 条配缝和同版 GLB 的 4 个开口。可编辑 Blender 工程随档保存。这里的无台面输入来自实测点云的受控遮除，保留了有效衣物像素的原始轴向深度。

程序另从同版纸样得到平铺 GLB。用留出帧的台面和下摆确定方向，用该帧轮廓及中值布面高度确定平移。对齐后在留出帧的原像素视线上比较实物点与同版 GLB 的首个交点。实际尺度固定，距离结果描述这一放置条件下的布面。留出帧参与平移配准，绝对空间摆位仍需用独立夹具复测。

| 留出实拍指标 | 值 |
| --- | ---: |
| 有效衣物射线 | 110,386 |
| 首次命中 GLB 的射线 | 103,532（93.79%） |
| 命中射线绝对距离中位数 | 2.95 mm |
| 命中射线绝对距离 P90 | 9.24 mm |
| 有符号距离中位数 | +0.36 mm |
| 平铺轮廓交并比 | 82.72% |

交并比较低的位置与袖口和下摆摆放改变有关。射线距离只覆盖有交点的布面，命中率和轮廓指标一起呈现遗漏部分。两帧仍缺完整背照及独立裁缝尺量值。

## 复现

```sh
python -m examples.evaluate_measured_support -o build/measured-support
python -m superarchiveallclo.compare_rgbd \
  build/measured-support/provided-plane/archive build/measured-support/source/independent-depth.json \
  --flat-comparison build/measured-support/flat-comparison -o build/depth-recheck
```

可用 `--source build/iri-tee/source` 重用原始数据。`study.json` 记录输入点云、相机基准和输出哈希。`ray-comparison/` 保存测量摘要和每条射线的有符号误差。`compare_rgbd` 默认核对归档里的悬挂 GLB。对平铺实验，`--flat-comparison` 指向同版平铺评测的目录，工具会核对纸样和 GLB 哈希。

独立深度清单顶层包含 `schema_version`、`item_id` 和 `views`。每个视角包含原深度的 `depth`、`depth_sha256`、`depth_unit`、`intrinsics`，以及以米记录、保持尺度的 `model_to_camera_m` 4 × 4 刚性变换和 `pose_source`。若深度还有背景，可附 `mask` 和 `mask_sha256`。首交点在实测点后方时，输出为正。有效深度缺失的位置保留为未观察区域。

## 固定机位只拍彩照

从 `n00764` 的桌面 XYZ 估计同一 Kinect 机位的台面，保留此前从设备注册数据核对的像素内参。用 `n00700` 的原始彩照和从彩照自动提取的蓝色衣物掩码生成毫米俯视图。该帧的衣物深度在归档完成后才用于评分。另一帧的桌面参与标定，其衣物部分保留作复扫对照。衣面投影按台面高度处理，褶皱和台面之间的距离会影响尺度。[原数据](https://www.iri.upc.edu/groups/perception/clothingDataset/) · [相机内参的检查](camera-calibration-study.md#已知三维标记的直接标定)

纯彩照投影为 **864 × 734 mm**，同帧实测 XYZ 的轮廓为 **840 × 704 mm**。保留尺度、仅在二维对照时平移，两者的轮廓交并比为 **88.41%**，轮廓距离 P90 为 **28.28 mm**。原图中自动划分的衣物掩码与发布方掩码的交并比为 **99.24%**。桌面点到拟合平面的 P90 为 **4.34 mm**。这组结果分开记录了从彩照恢复的平面几何、与深度的差值和制板的尺寸来源。

单面模型从台面投影估计下部衣身宽 511.0 mm，并以领深先验估计后中衣长 681.61 mm。回归的袖肥与袖窿不相容。程序把无法在图上确认的袖肥和袖口收至可配缝范围，保留原回归值及受约束的先验来源。候选归档回读 5 片毫米 DXF、15 条配缝和同版 GLB 上的 4 个开口。独立卷尺量法、背面及实际裁片仍待检查。

| 同版平铺 GLB 对照 | 有效射线命中率 | 命中射线绝对深度 P90 | 轮廓交并比 |
| --- | ---: | ---: | ---: |
| `n00700` 同帧留出深度 | 92.20% | 8.21 mm | 85.85% |
| `n00764` 另一帧衣物深度 | 91.06% | 9.06 mm | 79.46% |

平铺袖方向使用预设，单张彩照未能稳定定位两袖外缘。每帧各自用台面和下摆定方向，按轮廓及布面高度平移，尺度固定。射线误差衡量可见布面的深度，轮廓指标同时反映袖部和下摆缺失。

```sh
python -m examples.evaluate_camera_plane -o build/iri-camera-plane
python -m superarchiveallclo.check_archive build/iri-camera-plane/rectified/archive
```

下载过两帧时可添加 `--source` 复用原文件。`source/photograph.json` 保存可直接输入 `superarchiveallclo.rectify` 的清单。`rectified/archive/` 保存原始 RGB、由 RGB 得到的掩码、台面参数、毫米 DXF 及同版 GLB、Blender。`flat-comparison/` 和 `ray-comparison/` 保存用于评分的平铺网格、独立测量和哈希。
