# 第二件圆领 T 恤的挂拍三维对照

以奥克兰博物馆藏品 `2017.66.37` 为样本。正背平铺照片驱动模型制板。馆方后中衣长 770 mm 用于定尺度。胸宽和肩宽只用于盲测，候选纸样分别偏小 19.325 mm 和 4.351 mm。归档重新回读了 5 片毫米 DXF、15 条配缝和同版 GLB 的 4 个开口，并保存带实拍贴图的 GLB 和可编辑 Blender 工程。[馆藏尺寸和平铺实验](paired-photo-study.md)

两张独立特写可见内领连接处和成分标签。标签写明 67% 涤纶、33% 棉。归档记录成分来源，仿真仍使用参考针织物的机械参数。领条净长度、实际缝份、克重和拉伸仍待核对。[领口内侧](https://commons.wikimedia.org/wiki/File:T-shirt,_netball_(AM_2017.66.37-18).jpg) · [成分标签](https://commons.wikimedia.org/wiki/File:T-shirt,_netball_(AM_2017.66.37-19).jpg)

## 对照方法

相机约从正、背、左、右四个方向拍摄同一姿态。使用 #21 的 50 cm 实体尺和各张图中同一根立杆的图像宽度传递尺度。留出的 #01 和 #21 上的服装轮廓和局部标点均未用于构造参考外包络或三角测量。#21 的尺和立杆读数参与尺度标定。人体支架比前一件挂拍多出完整手臂。侧面被手臂挡住的区域标为未知，在外包络计算中保留这些射线，并从轮廓交并比与轮廓距离中排除。[馆藏照片、标尺读数、遮挡及轮廓标注](../examples/stellenbosch-drape.json)

程序只按立杆和颈顶注册导出的 GLB，保留纸样尺度。四个方向用 5 mm 网格估计外包络；六处签名笔画从不同方向的图像估计三维位置，再量到 GLB 三角面的最近距离。照片来自 Auckland War Memorial Museum Tāmaki Paenga Hira，采用 CC BY 4.0。[局部点位置](../examples/stellenbosch-landmarks.json) · [许可](https://creativecommons.org/licenses/by/4.0/)

| 留出实拍 | 同版 GLB 轮廓交并比 | GLB 轮廓距离 P90 | 照片外包络轮廓交并比 |
| --- | ---: | ---: | ---: |
| #01 无尺正照 | 85.25% | 50.58 mm | 88.15% |
| #21 带尺正照 | 84.10% | 52.18 mm | 86.41% |

正交估算下，六点到 GLB 的最近距离中位数为 55.53 mm，P90 为 63.86 mm。中、下衣身的最近模型点分别比照片估算位置向后约 48 mm 和 46 mm，分组各有两点和三点。两张留出图的反投影偏差合并中位数为 22.07 mm，按立杆平面换算。采用 EXIF 焦距的针孔近似时，点到 GLB 中位数变为 41.05 mm，留出反投影偏差中位数增至 40.68 mm。相机近似为这些三维差值引入了可见的测量不确定性。

用人台自身标记校正正面相机之间的位置后，留出反投影偏差中位数降到 10.52 mm。另调整侧相机后，两版袖长的三维点距排序一致。[相机标定与留出误差](camera-calibration-study.md)

照片外包络横跨约 615 mm，GLB 横跨约 674 mm。颈顶以下 200–300 mm 的一段，GLB 比外包络宽约 114 mm。400–500 mm 段的顺序反转，GLB 窄约 190 mm。600–700 mm 的衣身横跨差约 6 mm。横跨误差随高度变化，主要差距集中在袖子和其支撑附近。这里同时受袖片、通用人台形状和相机近似影响，仍缺同件实物的直接布面扫描和物性测试。

## 同版上臂支撑实验

从正面 #02 的布袖下方标记人台上臂布套的缝线，左右在图像平面上分别与竖直成 14.5° 和 16.8°。取均值 15.7°，让现有椭球上臂绕同一近肩点转动。保持纸样哈希、仿真 38 帧、初始袖位、物性和贴图一致。服装遮住近肩部分，也缺人台各关节的直接三维坐标。[标记所在实拍](https://commons.wikimedia.org/wiki/File:T-shirt,_netball_(AM_2017.66.37-2).jpg)

| 指标 | 原支撑 45° | 实拍估算支撑 15.7° |
| --- | ---: | ---: |
| 留出 #01 正面轮廓交并比 | 85.25% | 77.84% |
| 留出 #21 正面轮廓交并比 | 84.10% | 75.86% |
| 三维对应点到 GLB 的距离中位数 | 55.53 mm | 42.92 mm |
| 实体穿入最大顶点深度 | 0 mm | 36.31 mm |
| 200–300 mm 高度带宽度偏差 | +113.78 mm | +76.40 mm |
| 400–500 mm 高度带宽度偏差 | −189.78 mm | −161.32 mm |

支撑角降低使这两段袖部宽度差缩小，同时出现明显的实体穿入、布面变形和轮廓退化。当前样本的默认交付继续用 45° 支撑。调整角度的版本用于定位建模差异。

## 用双面袖口约束袖长

保留 45° 人台。程序从正背平铺照各自找左右袖口，用袖口外端的横跨和折线方向约束袖长，保留回归得到的袖窿与袖围。两面分别估算 258.655 mm 和 255.527 mm，候选纸样从 184.753 mm 调整到 257.091 mm。馆方没有提供独立的袖长实测。下表衡量同版平铺 GLB 对两张输入照的重现程度。

| 对照照片 | 基础模型横跨误差 | 双面袖口约束横跨误差 | 轮廓交并比变化 |
| --- | ---: | ---: | ---: |
| 正面 | −91.02 mm | +12.86 mm | 89.33% → 92.32% |
| 背面 | −73.55 mm | +30.33 mm | 89.55% → 91.94% |

馆方胸宽和肩宽的盲测误差未变，毫米 DXF 仍为五片并通过配缝校验。在 `2017.66.22` 上，两面估计 235.936 mm 和 232.527 mm。`2002.94.1.1` 的袖口无法通过双面检查，程序保留了基础模型的袖长。

留出挂拍 #01 和 #21 的轮廓交并比由 85.25% 和 84.10% 小幅升至 86.14% 和 84.64%。正面轮廓改善的同时，六个照片三维点到 GLB 的距离中位数从 55.53 mm 增至 58.91 mm，候选表面到照片外包络的单向 P90 从 11.12 mm 增至 15.03 mm。两版导出面的人台穿入读数均为 0 mm。当前保留可选袖长约束，继续用独立的三维证据核查垂坠。

## 复现

```sh
python -m examples.evaluate_auckland_archive --case stellenbosch -o build/stellenbosch-review
python -m superarchiveallclo.check_archive build/stellenbosch-review/review/archive
python -m examples.evaluate_stellenbosch_support build/stellenbosch-review -o build/stellenbosch-arm-study
python -m examples.evaluate_auckland_tees --case stellenbosch --body-profile --sleeve-profile -o build/stellenbosch-sleeve-flat
python -m examples.evaluate_auckland_archive --case stellenbosch --sleeve-profile -o build/stellenbosch-sleeve-drape
```

实验使用 Blender 4.3.2。`evidence.json` 记录尺寸和文件校验。`reference/calibration.json` 保留物理尺和立杆读数，`hull/comparison.json` 记录每个视角和高度带的数值，`landmarks/comparison.json` 保存点位和留出反投影。两种对照均校验同一交付 GLB 的哈希。
