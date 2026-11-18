# 窄身圆领 T 恤的双面模型和实测深度对照

新增一组可选权重，覆盖较窄的衣身、收放不同的下摆和不同摆放角度的袖口。训练图像来自可计算纸样的合成轮廓。窄衣身样本的平量胸宽从 355 mm 开始，前后照片的袖片摆角分别在 9–47° 内采样，还模拟了轻微的袖片缩短投影。训练时以袖窿深度到肩点的垂直落差约束袖肥，以袖肥约束袖口，再恢复毫米尺寸。生成裁片时继续检查净缝与袖山相容性。[原模型与输入格式](vision-model.md)

## 留出集

训练、验证和测试分别使用 960、160 和 200 张正背照片对与独立种子。对相同的留出样本，运行原、新权重的正式范围检查和制板。

| 已知平量胸宽 | 留出样本 | 原模型通过制板 | 新模型通过制板 |
| --- | ---: | ---: | ---: |
| 小于 425 mm | 55 | 4 | 48 |
| 425–484 mm | 38 | 14 | 38 |
| 485 mm 及以上 | 107 | 89 | 107 |

新权重在最窄组的胸宽和后中衣长误差 P90 分别为 7.34 mm 和 9.49 mm，袖口宽度 P90 为 26.94 mm。合成样本通过尺寸范围时才计算误差。窄组仍有七件未通过制板检查。[完整留出指标和权重哈希](../models/tee-vision-varied-evaluation.json)

使用同一双面衣身量测流程重测两件馆藏实物。`2017.66.22` 的肩宽绝对误差从 2.32 mm 增至 9.05 mm，`2017.66.37` 从 4.35 mm 增至 55.37 mm。新权重作为窄身采集的可选入口。馆藏两个样本继续使用此前各自的留出尺寸评测基准。[馆藏数据和尺寸口径](paired-photo-study.md)

## EPFL 正背实拍和独立深度

使用 `Ld_front/0582` 和 `Ld_back/0216` 的原始 TIFF 及配准深度。两次拍摄中的衣服有不同程度的褶皱，程序从各自实测的可见布面恢复近似平面后制板。此前的双面模型因背面较窄拒绝该组输入，新模型输出候选纸样。毫米 DXF 回读 5 片、15 条配缝、4 个成衣开口。同版 GLB 和可编辑 Blender 工程均生成。纸样给出的平量胸宽为 399.984 mm，后中衣长为 697.320 mm，袖长为 209.903 mm。数据发布页未提供独立成衣尺量和明确的再分发条件，仓库保留分析代码与来源哈希。[EPFL 原始资料](https://www.epfl.ch/labs/cvlab/data/texless-defsurf-data/) · [筛选、内参和深度单位假设](camera-calibration-study.md#新核实的正背实拍深度来源)

另留出同一正面序列的 `0576` 和 `0577` 深度。两帧均未用于制板。平铺 GLB 来自同一纸样，使用各帧布面估计的朝向，并做相同物理尺度下的平移。按原深度相机的射线评测真实可见布面。两帧共测得 20,210 条有效射线。

| 实测留出帧 | 命中率 | 命中射线绝对误差中位数 | 命中射线绝对误差 P90 | 平铺轮廓交并比 |
| --- | ---: | ---: | ---: | ---: |
| 0576 | 92.61% | 8.11 mm | 23.91 mm | 77.29% |
| 0577 | 91.57% | 7.31 mm | 23.37 mm | 77.25% |

留出帧在衣长方向 86%–95% 的可见衣身横截面，三维宽度落在 373–394 mm。纸样的对应下部宽度约为 400 mm。每帧的折皱、铺展状态与量取边界会影响这个比较，净裁尺寸仍需独立量尺核验。悬挂场景采用通用带臂人台，导出布面最大穿入 6.62 mm。独立深度对照检验的是同版平铺 GLB，悬挂姿态和支撑仍需对应实测。

## 复现

```sh
OPENBLAS_NUM_THREADS=3 python -m superarchiveallclo.train_vision --varied-layouts \
  --compare-model models/tee-vision.npz --weights build/tee-vision-varied.npz \
  --report build/tee-vision-varied-evaluation.json

python -m examples.evaluate_epfl_capture textureless_deformable_surfaces.zip \
  --model models/tee-vision-varied.npz --check-depth -o build/epfl-varied
python -m superarchiveallclo.check_archive build/epfl-varied/archive
```

EPFL 命令也接受发布压缩包解压后的目录。`study.json` 保存尺寸预测、两个留出参考和文件哈希。`archive/` 包含同版制板及常规人台仿真。`flat-comparison/` 保留逐射线实际评分的同版平铺 GLB。
