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
