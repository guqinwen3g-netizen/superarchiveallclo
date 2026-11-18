# 从实物三维扫描核对成衣开口

GarmentLab 使用商业激光和光学设备扫描实物，并公开研究基准中的处理后网格和部件标注。已下载并逐项校验 `10_WhiteShortSleeved` 与 `47_duanxT` 的网格和三类标注。在本实验中，发布物只提供直接扫描证据，未确认各件对应的原始正背照和单位标定。数字保留发布文件的原始坐标单位。扫描与标注文件均从作者发布入口获取。[论文和扫描方法](https://proceedings.neurips.cc/paper_files/paper/2024/hash/15f80ec0fed53885d2ca6272edb96ede-Abstract-Conference.html) · [作者发布说明](https://garmentlab.readthedocs.io/en/latest/tutorial/realworldbenchmark/index.html) · [文件哈希与结果](garmentlab-opening-study.json)

程序回读三角网格，检查布面连通性，再把只有一个相邻三角形的边串成闭环，以原有点标注确认领口、袖口和下摆。在编号 47 上，708 个衣身标注点保留为可配准证据，共检查了 798 个衣身标注点。两件都得到 **1 个领口、2 个袖口和 1 个下摆**。开口沿网格边逐段相加，另将路径等距分成 64 段，以直线连接各取点，记录长度对边界细度的敏感性。

| 开口 | 编号 10 原始边界 / 64 弦段 | 编号 47 原始边界 / 64 弦段 |
| --- | ---: | ---: |
| 较短袖口 | 208.56 / 196.14 | 277.34 / 248.33 |
| 较长袖口 | 296.92 / 218.75 | 409.79 / 306.57 |
| 领口 | 451.73 / 364.77 | 578.03 / 500.18 |
| 下摆 | 848.44 / 670.85 | 927.75 / 775.55 |

直接沿扫描网格逐边量出的长度受局部折痕、边缘锯齿和网格密度影响。编号 10 的较长袖口在等距重采样后缩短约 26%。这些结果可以支持开口结构核对。袖口生产尺寸仍需同件独立标定和实物尺量。

以奥克兰馆藏 `2017.66.22` 的正背照片重跑毫米纸样与同版仿真时，输出的布层中面原始周长为领口 **421.75 mm**、两袖口 **327.71 mm 和 327.86 mm**、下摆 **1255.76 mm**。该归档仍通过五片毫米 DXF、十五条接缝和四处开口的检查。其开口实测需要该件自己的三维参考。[奥克兰馆藏样本和独立尺寸](body-profile-study.md)

Smithsonian 藏品 `2020.0091.01` 有真实正背照、馆方未安装和已安装量尺，并公开三维扫描。核对原始照片后，确认该短袖球衣为 V 领，正背照拍摄时使用人台。它可用于其他领型的后续研究。馆方将三维下载用途限定在非商业、教育和个人使用范围。[藏品与使用条件](https://americanhistory.si.edu/collections/object/nmah_2000307) · [馆方正背原照与扫描](https://avpreservation.si.edu/edan-record/3d_package%3Abdea64ee-e17f-490a-bf78-c6815c28c88d)

## 复现

```sh
python -m examples.evaluate_garmentlab_openings -o build/garmentlab-openings
python -m examples.evaluate_auckland_tees --case netball --body-profile -o build/auckland-profile
python -m superarchiveallclo.capture build/auckland-profile/netball/data/paired.json \
  --model models/tee-vision.npz --blind --form display_torso -o build/auckland-opening-archive
```

有同件、已标定的扫描参考时，可运行 `python -m superarchiveallclo.scan_openings --help` 查看输入。两袖的误差按周长排序配对。要逐侧分析，仍需采集端提供左右标识。
