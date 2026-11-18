# superarchiveallclo

**Super Archive All Clone** 为真实服装建立数字档案。从实物拍摄、测量或扫描开始，为每件衣服保存原始证据、可编辑的二维纸样，以及由同一套裁片缝合并模拟出来的 3D 模型。长期目标是覆盖各类服装，并交付经实物打样和目标 CAD 验证的制板文件。

**当前模型覆盖按标准拍摄的单层圆领 T 恤。** 从带比例尺的正背照片预测制板尺寸，并生成同版纸样和 3D。首个风格方向为 **Clean Fit / Minimalism 极简男装穿搭**。

首款提议用一件微落肩圆领 T 恤，把真实照片、纸样、仿真和试穿走通。随后加入直筒长裤组成第一套穿搭，再用外搭衬衫验证叠穿。开发从可自己生成、自己测量的样本开始。

## 跑通一份归档

安装 Python 依赖，确保 Blender 4.3+ 可通过 `blender` 调用。内置样例只提供合成正背照片。预测尺寸、制板和仿真由程序自动完成。

```sh
python -m pip install -r requirements.txt
python -m superarchiveallclo.capture examples/vision-demo/session.json \
  --model models/tee-vision.npz -o build/archive
```

归档包括原图、比例尺识别记录、预测尺寸及来源、DXF / SVG / A4 纸样、GLB 和 Blender 场景。替换照片可处理新的标准采集档案。训练和评估采用不同的合成样本。实物误差需要独立样衣测量。[图像模型与可复现训练](docs/vision-model.md) · [实物采集与验收](docs/capture.md)

运行 `python -m superarchiveallclo.check_archive build/archive` 可回读 DXF 毫米轮廓，核对原图和纸样哈希，并检查 GLB 容器。独立盲测样本可用 `superarchiveallclo.review` 一次生成归档、尺寸和轮廓报告。[实物 MVP 目标与数据格式](docs/mvp-acceptance.md)

程序也能从毫米尺度的实物点云重建单面候选纸样和 3D。已在 RGBench 的真实圆领 T 恤上运行，保存原始点云、量测依据、DXF 和 GLB，并用同件复拍检查平铺三维，用配套数字参考量化各裁片误差。[袖山与袖窿的纸样结果](docs/sleeve-join-study.md) · [平铺网格应变和接缝核对](docs/flat-mesh-relaxation.md)

同一套纸样也已用于双点悬挂的重力仿真。六帧实物扫描用于核查正视轮廓及三维距离。[悬挂对照与复现](docs/suspension-study.md)

悬挂对照也支持从已标定的相机逐射线量出预测的布面深度，并用早期扫描预测短时摆动。[实物点云的深度和摆动评测](docs/measured-depth-study.md)

已注册的彩照与深度图可用 `superarchiveallclo.rgbd` 校正俯拍透视并生成归档。原始 RGB、深度和可选分割图随归档保存。[输入与标定说明](docs/capture.md#已注册的深度照片) · [真实 RGB-D 样本实验](docs/rgbd-study.md)

斜拍的普通照片可用 `superarchiveallclo.rectify` 和四个共面的实测矩形标记恢复毫米俯视图，再运行同一套照片制板和 3D 流程。[采集及运行](docs/capture.md#斜拍与四点平面标定) · [馆藏正背照倾斜实验](docs/planar-rectification-study.md)

深度入口也能读入与彩照逐像素对应的 XYZ 点云并恢复毫米纸样。已用另一帧实物 Kinect 点云核对同版平铺 3D。[IRI UPC-CSIC 实物复扫](docs/iri-pcd-study.md)

设备只保留衣物深度时，可以用同一相机的外部实测平面生成毫米俯视图。留出的深度帧可直接按相机射线核对同版 GLB。[表面深度评测与复现](docs/registered-depth-review.md)

真实正背照已用于候选纸样归档、留出的馆方尺寸评分及同版平铺 GLB 的投影对照。[两件奥克兰博物馆藏品实验](docs/paired-photo-study.md)

归档时可以把实拍衣身的颜色、文字和图案映射到由纸样生成的 3D，并随 GLB 和 Blender 工程保存。[两件实物的正背印花和渲染对照](docs/photo-appearance-study.md)

同一件馆藏圆领 T 恤另用标定挂拍构造三维外包络，在留出照片及三维体素上评估同版 GLB，并量化人台手臂支撑的影响。[人台三维对照与复现](docs/mannequin-study.md)

一条命令可从奥克兰馆藏正背原照重跑盲测、毫米 DXF、同版 3D、独立挂拍外包络和局部三维点对照。[完整实物实验与复现](docs/photo-landmark-study.md)

第二件馆藏圆领 T 恤的挂拍评测支持显式标记人台遮挡，并按高度报告袖部和衣身的宽度与深度偏差。[带臂人台的实物对照](docs/stellenbosch-drape-study.md)

照片三维对照可从独立实测标记求出相机投影，也可读取已有相机矩阵。馆藏挂拍用留出标记核对相机误差，真实 Kinect 数据验证原生彩图和三维坐标的配准。[标定格式、结果和复现](docs/camera-calibration-study.md)

双面采集可分别恢复腋下胸宽与下摆宽，把侧缝收窄或放宽的幅度写入毫米纸样及同版仿真。不同宽松度的实物已用馆方胸宽和肩宽做盲测。[衣身宽度对照](docs/body-profile-study.md) · [馆藏正背照与尺寸](docs/paired-photo-study.md)

窄身和较短袖口横跨的采集可选新训练的双面权重。EPFL 正背实拍已生成毫米纸样和同版 3D，并用两帧实测深度复核。[窄身模型和留出评测](docs/varied-tee-model.md)

展台仿真可以使用无臂人台的接触支撑。对照报告记录模型随高度变化的前后深度，以及导出时布料的位移。[同版垂坠对照](docs/drape-support-study.md)

成衣仿真归档可读取领口、两只袖口和下摆的三维周长。带部件标注的实物扫描也可以执行同样的开口检查。[两件短袖实物扫描的构造核对](docs/garmentlab-opening-study.md)

## 生成首款 T 恤纸样

在仓库目录使用 Python 3.10+ 和 ReportLab。样例为虚构的 960 mm 净胸围体型，配 560 mm 成衣半胸宽和 680 mm 后中衣长。

```sh
python -m pip install -r requirements.txt
python -m superarchiveallclo examples/tee.json -o build/tee
python -m superarchiveallclo examples/tee.json -o build/tee-variants --variants
```

第二条生成基准款。第三条另生成基准款和三个单变量变体，分别缩短衣长 20 mm、放宽半胸 20 mm、增加每侧落肩 10 mm。每次指定一个新的输出目录。

| 文件 | 用途 |
| --- | --- |
| `pattern.json` | 毫米级制板参数、五片裁片的曲线控制点、裁剪线、配缝关系、布纹方向、剪口和领条折线 |
| `pattern.svg` | 可在矢量编辑器中查看和编辑的纸样 |
| `pattern-a4.pdf` | A4 拼页原型，第一页附 100 mm 校准方框和裁片页序 |
| `measurements.csv` 和 `seams.csv` | 成衣测量值、每段缝线长度、领条与领口长度比及配缝残差 |

编辑样例 JSON 可重生成。落肩量是每侧相对净肩宽的水平延伸。袖山尺寸按前后袖窿弧长求解。领条样例使用领口净缝长度的 85%，实际松紧需要按面料回弹和实样校准。组装领条时先合两端，再沿折线对折，一同缝入两个长边的毛口。[样例尺寸](examples/tee.json)

打印时选 **实际大小 / 100%**。先量首页方框，再按各裁片的行和列对齐重叠区十字。实线为裁剪线，蓝色虚线为净缝线，点划线为折线，短刻线为剪口。左右袖各裁一片，保留镜像关系。[基准款 A4 纸样](examples/tee-a4.pdf)

当前验收覆盖数字纸样和尺寸。样衣、面料回弹与试穿仍需实物或三维验证。

## 项目规划

| 从这里开始 | 内容 |
| --- | --- |
| [产品与首款设计](docs/product-plan.md) | 第一套穿搭、四类输入、用户怎么改版、每个交付状态代表什么 |
| [模型与系统设计](docs/model-design.md) | 模型学什么、怎样表示衣服、如何让纸样和试穿对应、从哪里得到第一批标签 |
| [开发路线与验收](docs/roadmap.md) | 第一轮动手做什么、每个阶段交什么、如何用真实衣服检验进步 |
