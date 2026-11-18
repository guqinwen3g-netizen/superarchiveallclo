# 斜拍照片校正实验

以奥克兰博物馆 `2002.94.1.1` 的同件圆领 T 恤正背原图作为基准。两张原照以可见卷尺的 20 到 30 英寸刻度分别定尺度。对两张图施加不同的模拟相机倾斜。四个控制点由模拟变换产生。这个实验测量同一套模型对倾斜的敏感程度，成衣打版精度仍需独立实物尺寸。照片来自 Auckland War Memorial Museum Tāmaki Paenga Hira，许可为 CC BY 4.0。[正面](https://commons.wikimedia.org/?curid=64330673) · [背面](https://commons.wikimedia.org/?curid=64330689) · [许可](https://creativecommons.org/licenses/by/4.0/)

下表为同一模型相对原照预测的有符号变化，单位为毫米。只用一段卷尺定尺度时，倾斜照片的局部尺度随位置变动。四点校正后，11 个预测尺寸的绝对改变量中位数为 0.235 mm，最大为 1.302 mm。输出 DXF 回读通过，含 5 片和 15 条配缝。同版 GLB 有四个开口，嵌有原图生成的贴图，原生 Blender 工程随归档交付。

| 尺寸 | 斜拍后仍用单段卷尺 | 四点校正后 |
| --- | ---: | ---: |
| 腋下平量胸宽 | +149.160 | +0.698 |
| 后中衣长 | −41.959 | +0.491 |
| 肩宽 | +93.182 | +0.591 |
| 袖窿深 | −52.060 | +1.302 |

## 复现

```sh
python -m examples.evaluate_planar_rectification -o build/auckland-planar-tilt
python -m superarchiveallclo.check_archive build/auckland-planar-tilt/rectified/archive
```

可用 `--photos` 指向已有的两张馆藏原图。脚本会查验原图哈希，并把从实拍生成的模拟倾斜输入标为 `synthetic_fixture`。输出 `study.json` 记录全部尺寸和归档核验。`rectified/session.json` 保留四点、视野、比例和从输出像素回到源图的投影矩阵。
