# 用实物标尺标定馆藏 T 恤挂拍相机

以奥克兰博物馆 `2017.66.22` 的两张正面挂拍 #01、#20 为留出照片。两张照片均用于检验同一纸样生成的 GLB。纸样来自独立的正背平铺照，原有尺寸盲测为胸宽 +37.38 mm、肩宽 −2.32 mm。归档包含 5 片毫米 DXF、15 条接缝及 4 处成衣开口。照片由 Auckland War Memorial Museum Tāmaki Paenga Hira 发布。[同件服装的完整归档](photo-landmark-study.md) · [原图与标尺](../examples/auckland-drape.json) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

从全分辨率原图检测黑色人台底板的四条边，从 EXIF 取得焦距，用 #20 中 500 mm 标尺的两点定尺度。假设尺面标记与底板上表面共面，解得底板宽 322.70 mm、深 315.93 mm，据此估计两张照片的针孔相机。立杆足点定位竖轴，两张照片各自读取同一上方基准，两次基准高差 8.92 mm。立杆其余水平轮廓用于留出核对，投影横向偏差中位数分别为 0.35、0.62 像素，按 576 × 864 的标注分辨率报告。两张图的立杆直径估计为 18.34 和 18.28 mm，实际杆径没有外部量测。[标注输入](../examples/auckland-stand.json) · [相机位姿及诊断](auckland-stand-study.json)

| 同一组 9 个三维点与同版 GLB | 正交近似 | 标尺与底板标定 |
| --- | ---: | ---: |
| 两张留出照反投影误差中位数 | 17.86 mm | 15.70 mm |
| 两张留出照反投影最大误差 | 32.32 mm | 19.71 mm |
| 达到每张照片 25 mm 限值的点 | 6/9 | 9/9 |
| 全部 9 点到 GLB 三角面的距离中位数 | 9.44 mm | 9.44 mm |
| #01 同版 GLB 轮廓交并比／轮廓距离 P90 | 80.11%／54.23 mm | 79.13%／56.85 mm |
| #20 同版 GLB 轮廓交并比／轮廓距离 P90 | 83.98%／53.43 mm | 82.57%／53.57 mm |

反投影和轮廓距离按各照片基准平面的像素尺度换算。九个实物局部点从原有近景与侧视计算，仍受这些视角的正交近似限制，本轮用两张独立标定的照片核对点位。两台正面机位相距 45.51 mm，仅用这两张图三角测量时，最弱方向对 1 像素误差的局部灵敏度为 138–209 mm。整体衣形的轮廓交并比下降，仿真外形仍需改进。下摆、背面、净裁剪线和缝份仍需同件实物的独立测量。

## 复现

```sh
python -m examples.evaluate_auckland_archive --case netball --reprojection-limit-mm 25 -o build/auckland-netball
python -m examples.evaluate_auckland_stand build/auckland-netball -o build/auckland-stand
```

使用 Blender 4.3.2。第二条命令保存 `cameras.json`、`study.json` 和两套逐点投影报告。报告记录了纸样、GLB、照片和标注的 SHA-256。标尺和衣物标点来自馆藏实拍，按同一版本重新生成交付件后即可核对实测指标。
