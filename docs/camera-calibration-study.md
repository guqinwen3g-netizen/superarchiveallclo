# 照片三维对照的相机标定

照片三维点评测现可读取每张相机的 3 × 4 投影矩阵。矩阵以毫米世界坐标为输入，输出在标注分辨率上的齐次像素坐标。它支持斜视角。每个相机矩阵绑定原照哈希，核对衣物编号、图像尺寸及标定来源。至少两张视图提供的射线需要能解析三维坐标。程序用独立留出照片报告反投影误差。

## 同件馆藏的相机来源核对

`2017.66.37` 的尺和立杆给出绝对尺度。用 #02 作基准，从人台的颈顶、手腕和黑色手端估计 #01 与 #21 的相对平移、旋转和缩放。正面的两处肘部不参与拟合。另试一组用侧面肘部和手端调整纵向投影的参数，手腕留出。人台轮廓交接点近似对应，侧拍的前后透视仍存在。所有衣物签名点和两版 GLB 均未参与相机参数拟合。[相机与人台标注](../examples/stellenbosch-fixture.json) · [馆藏原照与量尺](../examples/stellenbosch-drape.json)

| 相机对照 | 签名点留出图反投影中位数 | 原袖长 GLB 点距中位数 | 修正袖长 GLB 点距中位数 |
| --- | ---: | ---: | ---: |
| 原正交近似 | 22.07 mm | 55.53 mm | 58.91 mm |
| 人台约束两张正面留出图 | 10.52 mm | 55.53 mm | 58.91 mm |
| 正面约束加侧面纵向调整 | 13.40 mm | 55.34 mm | 58.67 mm |

正面的人台肘部留出偏差从 11.23–14.99 像素降到 2.57–7.08 像素。修正两张留出图时保留原来的点位计算相机，因此到 GLB 的距离一致。调整侧面之后，三维点参与了重新计算，两版的点距排序仍一致。反投影毫米数按原立杆尺度换算。这个实验改善同件实拍之间的位置一致性，直接密集布面深度仍待补齐。[两版 GLB 的实物归档](stellenbosch-drape-study.md)

```sh
python -m examples.evaluate_stellenbosch_cameras \
  build/stellenbosch-review build/stellenbosch-sleeve-drape -o build/stellenbosch-camera-check
```

输入目录由 [前两版归档命令](stellenbosch-drape-study.md#复现) 生成。输出包含各相机矩阵、留出实拍的叠图和同版 GLB 校验。

## 已知三维标记的直接标定

把相机拍到的定位标记坐标和独立测得的三维坐标配对。完整投影标定至少需要六个跨多个深度的点。拍摄场景坐标用毫米，选择参考物附近的坐标原点。可保留额外标记作独立检查。每张照片的标定记录包含 `photo`、`photo_sha256` 以及 `fit`，可选 `holdout`。每条观察包含 `world_mm` 和 `pixel`。顶层包含 `schema_version`、`item_id`、`source`、`world_unit`、`image_size_px` 与 `views`。标注可用等比例缩小的原图。相机矩阵按该标注尺寸输出。

```sh
python -m superarchiveallclo.calibrate_cameras controls.json -o camera.json
python -m superarchiveallclo.photo_points archive reference.json landmarks.json \
  --camera-solution camera.json -o build/metric-photo-check
```

用 IRI UPC-CSIC 的真实 Kinect 帧 `n00764` 验证这一入口。从背景选取 295 个 RGB 注册坐标拟合相机，保留 209 个衣物上的真实深度点检查。留出重投影 P90 为 0.000009 像素，反映该数据编码中 RGB 网格与原生 XYZ 的配准一致性。两组点仍来自同一次 Kinect 采集。这个检查与奥克兰样本的立杆尺度和衣服三维误差各自独立。[数据发布](https://www.iri.upc.edu/groups/perception/clothingDataset/) · [实物点云的原归档实验](iri-pcd-study.md)

```sh
python -m examples.evaluate_iri_camera -o build/iri-camera-check
```

已有原始数据时可以加 `--source build/iri-tee/source`。脚本保存控制点清单、图像与点云哈希及可直接复用的 `camera.json`。

## 新核实的正背实拍深度来源

已检查 EPFL 的 Texture-less Deformable Surfaces 原发布压缩包索引及抽样原文件。`tshirt` 下有明确命名的 `Ld_front`、`Ld_back`、`Ll_front` 和 `Ll_back`。选取的四张图均有 224 × 224 配准 RGB、深度和发布方内参。程序回投影每张约 9,300–10,800 个有效点，按米解释原深度时，布面到最佳拟合平面的距离 P90 为 31.6–68.9 mm。画面可见不同程度的褶皱，正背序列也分别拍摄。可以继续用它评测可见布面的真实深度，制板还需处理不同姿态及卷尺尺寸来源。发布页未注明明确的再分发许可，仓库保存核验代码和哈希。[资料发布](https://www.epfl.ch/labs/cvlab/data/texless-defsurf-data/) · [采集论文](https://arxiv.org/abs/1803.08908)

```sh
python -m examples.inspect_epfl_tee textureless_deformable_surfaces.zip -o build/epfl-depth-check.json
```

也可以直接指定解压后的数据根目录。报告保留实际抽样帧名、每个文件的哈希以及深度单位假设。

随后分序列扩大到 177 个深度帧，按布面距自身拟合平面的 P95 排查。35 mm 是现有平铺预检采用的限值。`Ld_front/0582` 和 `Ld_back/0216` 分别得到 30.11 mm 与 33.55 mm，原彩照和注册深度已通过保持尺度的投影。正背投影外框分别为 678 × 662 mm 和 644 × 738 mm。背面的横跨仍超出当前双面模型训练覆盖的输入范围，候选制板被入口拒绝。两帧是不同变形姿态，已保留投影和拒绝依据。[模型的实测深度评测入口](registered-depth-review.md)

```sh
python -m examples.inspect_epfl_tee textureless_deformable_surfaces.zip --screen -o build/epfl-screen.json
python -m examples.evaluate_epfl_capture textureless_deformable_surfaces.zip -o build/epfl-capture-check
```
