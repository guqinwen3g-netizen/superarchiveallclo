# 用腋下交点约束袖山和袖窿

本文的平铺复评采用提交 `c96aa00` 的求解方式。[新版同纸样平铺应变对照](flat-mesh-relaxation.md)

从平铺扫描取得双侧袖口和可见腋下交点。程序将腋下交点分别投影到沿袖和横袖方向，估计袖口至袖山底的距离以及折平袖片宽度。用双侧腋下高度推算袖窿深，再用前后袖窿净线求解能配缝的袖山高。该高与沿袖距离相加得到袖长。两侧投影不一致或净缝无解时，程序回退到袖口量测与已有的学得尺寸。采集记录标明采用了哪些观察。

旧版平铺模拟从肩点摆放袖片。新版让袖片下段从腋下接缝定位，在袖山内逐步过渡到肩点。比较时，两套纸样使用同一平铺求解和同一袖角。袖角 **18.19°** 取自重建输入的两侧袖口，八帧参考扫描只参与评分。

## 实物与数字参考

沿用 RGBench 的真实圆领 T 恤、输入扫描和数字参考。新版单面归档通过毫米 DXF 回读，含五片裁片、十五条配缝和由同版纸样驱动的 GLB。九帧中八帧检出了可用于投影的双侧袖口及腋下交点。[实验数据与 CC BY 4.0 许可](https://huggingface.co/datasets/RGBench/RGBench-Cloth-Sim2Real-v1)

![两种袖片与 RGBench 数字参考的叠图](../examples/rgbench-sleeve-comparison.png)

左侧为前版袖片，右侧为使用腋下交点的新版袖片。蓝色是参考独有区域，橙色是候选独有区域，深灰是重叠区域。两图为相同毫米尺度。数字参考来自 Hu 等人，许可为 CC BY 4.0。[逐帧指标、源哈希和制板校验](sleeve-join-study.json)

| 项目 | 仅袖口约束 | 袖口和腋下约束 |
| --- | ---: | ---: |
| 袖片与数字参考净线交并比 | 约 71.5% | 约 78.1% |
| 袖片展开宽差，候选减参考 | −86.33 mm | −33.46 mm |
| 袖片展开高差，候选减参考 | −40.49 mm | +7.82 mm |
| 平铺轮廓交并比，八帧中位数 | 91.30% | 91.38% |
| 平铺边缘距离 P90，八帧中位数 | 28.04 mm | 27.76 mm |
| 平铺网格边长应变 P95，相对净纸样 | 9.32% | 12.17% |

袖片尺寸更接近作者的数字版，平铺轮廓整体相近。新版在平铺求解时仍有局部拉伸。外轮廓看到的腋下交点受折叠与袖子姿态影响，袖山曲线和前后袖窿净线还需要原件接缝资料复核。试穿垂坠也需要同姿态的实物参考。

## 有授权的同件正背照

Wikimedia Commons 用户「あじさか」公开了同一件鹿儿岛 OpenStreetMap 调查 T 恤的正背图，许可为 CC BY-SA 4.0。本次使用该作者移除背景后发布的文件。程序检测到两面至少一侧袖缘贴近图边，省略了整体宽高比。可见的下段衣身宽与衣长之比分别为 **0.7033** 和 **0.6923**，差 **0.0110**。这组照片用于无量纲的正背形态核对，毫米尺度与原件实测仍待独立取得。[正面](https://commons.wikimedia.org/wiki/File:OpenStreetMap_Kagoshima_survey_T-shirt_front.jpg) · [背面](https://commons.wikimedia.org/wiki/File:OpenStreetMap_Kagoshima_survey_T-shirt_back.jpg) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## 复现

```sh
python -m examples.evaluate_rgbench_scan --compare-references -o build/rgbench-join
```

前版在提交 `1819516` 中用 `python -m examples.evaluate_rgbench_scan --archive -o build/rgbench-cuff` 生成。新版的评估命令可通过 `--baseline-archive` 读取该版本的 `build/rgbench-cuff/frames/18-21-20-start/archive`，以同一姿态复评。独立检查上述公开正背照可用 `python -m superarchiveallclo.vision osm-front.jpg osm-back.jpg --shape-only`。
