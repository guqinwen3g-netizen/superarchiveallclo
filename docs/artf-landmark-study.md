# 实拍短袖关键点模型

模型从短袖 T 恤的轮廓自动恢复领口方向，并回归双侧肩点、领口端点、袖口上下端点、腋点和下摆边角，共十二点。它在不同角度匹配训练集中由人工标注确定方向的轮廓，再用核岭回归估计各点。左右以领口朝上时的图像方向命名。运行可使用提取出的照片轮廓或一张与原照对齐的单通道轮廓图。

## 留出实拍评测

使用 aRTF 发布方的训练、验证和测试划分。可用训练照 167 张，其中一张训练照的标注无法确定方向，验证 42 张，测试 400 张。验证集选择 12 个近邻和正则系数 0.16，实拍测试照仅用于评分。发布文件没有可可靠匹配的单件服装 ID，因此额外单列训练未覆盖的三个采集地点；它们也不构成服装身份留出证明。[aRTF 数据及作者署名](https://zenodo.org/records/14501499) · [发布方数据说明](https://github.com/tlpss/aRTF-Clothes-dataset#using-this-dataset)

| 条件 | 图像 | 可见点 | 像素误差中位数 | 像素误差 P90 | 误差在轮廓对角线 5% 内 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 同姿态、训练集平均点位 | 400 | 4,684 | 15.03 | 36.44 | 39.86% |
| 轮廓匹配与关键点回归 | 400 | 4,684 | 7.04 | 22.32 | 78.10% |
| 回归模型，训练未覆盖地点 | 200 | 2,357 | 6.01 | 17.91 | 83.45% |

两种完整测试方案使用同一组自动方向。回归模型的方向偏差中位数 4.06°，94.25% 在 22.5° 内。评测输入为数据发布方描绘的外轮廓，像素误差衡量给定轮廓时的关键点定位，未计入 RGB 自动分割误差。逐点统计、照片划分、标注哈希和权重哈希随仓库保存。[权重](../models/tee-landmarks-artf.npz) · [完整留出报告](../models/tee-landmarks-artf-evaluation.json)

## 实物归档对照

将同一权重用于两件奥克兰博物馆藏品的原始正、背照，四张照片的轮廓均由程序提取。尺度来自各件馆藏的后中衣长，肩宽和胸宽留作外部对照。下表量的是照片中候选点之间的直线距离，胸宽栏是馆方平铺胸宽与模型腋点跨度的并排读数。馆方测量位置和照片上的投影可能不同。[照片、独立尺寸及归档结果](paired-photo-study.md)

| 藏品 | 馆方肩宽 | 照片肩点跨度 正／背 | 馆方胸宽 | 照片腋点跨度 正／背 |
| --- | ---: | ---: | ---: | ---: |
| 2017.66.22 | 505 mm | 558.93／572.40 mm | 515 mm | 610.88／611.29 mm |
| 2017.66.37 | 590 mm | 492.06／523.09 mm | 610 mm | 606.05／597.91 mm |

两件的照片肩点跨度与馆方肩宽相差 53.93–97.94 mm。轮廓会遮住或模糊接缝位置，肩宽需要缝迹、深度或直接量测复核。制板尺寸仍由其独立的毫米回归和几何约束产生。选用关键点选项后，归档在 `capture.json` 中保存权重哈希、照片哈希、原图像素坐标和投影跨度。第一件已重跑完整归档，5 片毫米 DXF、15 条配缝和 4 个 GLB 开口通过回读，纸样哈希与原实验一致。

## 复现

从 aRTF 发布页下载 `artf-all-categories-resized_512x256.zip`。权重含由发布方标注轮廓生成的降采样描述特征，数据集按 CC BY 4.0 发布。训练程序从压缩包读取三个标注文件并核对发布版哈希。[数据集与作者](https://zenodo.org/records/14501499) · [许可条款](https://creativecommons.org/licenses/by/4.0/)

```sh
python -m examples.train_artf_landmarks artf-all-categories-resized_512x256.zip \
  --weights build/artf/tee-landmarks.npz --report build/artf/evaluation.json
python -m superarchiveallclo.landmark_vision front.jpg \
  --model models/tee-landmarks-artf.npz --overlay build/front-landmarks.png
```

单张照片可选 `--mask aligned-mask.png`。已标定的正背照片可以直接在归档里保存点位。以下第一条命令准备馆方原照、尺度和留出证据；后两条重跑归档并检查交付文件。

```sh
python -m examples.evaluate_auckland_archive --case netball -o build/auckland-end-to-end
python -m superarchiveallclo.capture build/auckland-end-to-end/input/paired.json \
  --model models/tee-vision.npz --landmarks-model models/tee-landmarks-artf.npz \
  --blind --form display_torso -o build/auckland-landmark-transfer
python -m superarchiveallclo.check_archive build/auckland-landmark-transfer
```

第二件可将样本选项改为 `--case stellenbosch`，并替换两个输出目录及输入清单路径。
