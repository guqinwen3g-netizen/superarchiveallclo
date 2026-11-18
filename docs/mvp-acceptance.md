# 实物归档 MVP 验收

首件目标是一件有使用权、独立于训练的单层圆领 T 恤。从有可信尺度的正背面照片或扫描运行模型，生成候选纸样和同版 3D。用原件的实测尺寸核对纸样，用结构近照或原始版片核对缝合，用同姿态和尺度的参考图对照 3D。已用公开实物点云验证单面采集和候选出档。[单面实验与复拍差异](real-scan-study.md)

| 工作 | 验收证据 |
| --- | --- |
| 样本采集 | 同一件衣服的正背原始采集、可信尺度、卷尺实测和结构证据。记录使用范围与拍摄条件 |
| 盲测 | 保存实测标签和模型预测。每个尺寸报告毫米误差，对所有拒绝项计数。以款式或实物为组隔离训练、验证和盲测 |
| 版片 | 报告裁片闭合及配缝。回读 DXF 的毫米单位、裁片与裁剪线。依据独立原件证据核对袖窿、领口、下摆、领条等构造 |
| 3D | 用同版纸样哈希核对来源，确认 GLB 可读。以相同视角对照实物形态，注明材质先验、轮廓误差和未确认的细节 |

当前仓库的照片样例和基准权重基于合成样本，可用于回归入口。实物验收需要一份单独的采集档案。最终报告记录归档版本、样衣标识、已核对部位和剩余误差。

馆藏 `2017.66.22` 的实物实验可用 `python -m examples.evaluate_auckland_archive -o build/auckland-end-to-end` 复现。归档保留两项独立尺寸、袖口近照、同件挂拍三维外包络以及分视角估计的局部布面点。直接布面扫描与打样仍待核对。[完整实物证据](photo-landmark-study.md)

## 一次运行独立样衣验收

将样衣登记到下节的数据清单和 `test` 集合。准备独立轮廓参考后，运行下面的命令。还没有轮廓参考时，可先省略 `--shape-reference`，生成尺寸和归档检查结果。

```sh
python -m superarchiveallclo.review data/dataset.json --item tee-01 \
  --model models/tee-vision.npz --shape-reference data/tee-01/shape-reference.json \
  -o build/tee-01-review
```

一次运行得到 `review.json`、`archive/` 和可用时的 `shape/`。程序对评分和重建使用同一份权重快照，核对照片哈希、预测尺寸和归档状态。`review.json` 记录已实测尺寸与误差、DXF 回读和 3D 轮廓证据。领条、袖窿和下摆的结构及面料仍需用实物核对。

新生成的权重记录参与训练和调参的款式组与照片哈希，评分时会拦截重叠样本。旧权重若缺少这项记录，评分报告会标明，并需核对原始训练记录。

含有实测标签的验收档案须用盲测模式生成候选归档。该模式让原始标签只参与后续评分。生成记录中的 `reference_measurements_withheld` 可核对此设置。

```sh
python -m superarchiveallclo.capture data/tee-01/session.json \
  --model models/tee-vision.npz --blind -o build/tee-01
```

## 接入配对数据

用 [采集文件格式](capture.md) 记录照片和实测。真实原件设 `kind` 为 `physical`，合成样本设为 `synthetic_fixture`。盲测可填写已有的独立实测项。要参与本轮训练或正则化选择，样本应包含模型的全部 11 项目标。项目会保留缺测和拒绝记录，按实际有实测的项目计算误差。

数据清单引用每件衣服的采集文件。相对路径相对于清单。相同原件的重复拍摄、同纸样族不同尺码与颜色使用同一个 `group`，把整组放入同一个集合。每件实物有稳定且唯一的 `item_id`。将使用许可、授权范围或来源记录到 `rights`。

```json
{
  "schema_version": 1,
  "samples": [
    {
      "session": "tee-01/session.json",
      "group": "crew-neck-family-01",
      "split": "test",
      "rights": "原件所有者授权用于内部评测，附来源记录"
    }
  ]
}
```

对已有权重盲测。报告自动按采集类型统计，包含全部样本的处理状态。

```sh
python -m superarchiveallclo.dataset data/dataset.json \
  --model models/tee-vision.npz --report build/real-evaluation.json
```

有独立训练集和验证集后，可重新训练，并让新模型和现有模型在同一盲测集上评分。成果保存在新的权重文件。

```sh
python -m superarchiveallclo.train_vision --dataset data/dataset.json \
  --compare-model models/tee-vision.npz \
  --weights build/next-tee-vision.npz --report build/next-tee-evaluation.json
```

## 可复现的工程验收

```sh
python -m unittest discover -s tests
python -m superarchiveallclo.capture examples/vision-demo/session.json \
  --model models/tee-vision.npz -o build/smoke-archive
python -m superarchiveallclo.check_archive build/smoke-archive
```

采集命令在发布完整归档前运行格式与来源校验。独立检查命令也可对已有目录重跑。回读覆盖原图校验和、DXF 毫米单位和裁剪线、纸样与三维源哈希及 GLB 容器。

## 量化实物与 3D 的轮廓差异

先用实物进行独立的同姿态拍摄。正面、背面和侧面都尽量使用远距离、视线垂直的取景。用与观察平面平行的实物标尺得到每像素毫米数。参考图保留原始证据，并提供白色衣身、黑色背景的轮廓图。轮廓边缘保留黑色边距，避免把遮挡或裁切当成衣形。

对照清单中的 `kind` 和 `item_id` 与归档一致。`setup` 记录人台、手臂姿势、取景、比例恢复方式，以及轮廓图来自哪张原始照片。视角可填 `front`、`back`、`left` 或 `right`。左右从穿着者角度定义。

```json
{
  "schema_version": 1,
  "kind": "physical",
  "item_id": "tee-01",
  "setup": "记录原图路径、参考人台、手臂姿势和取景标尺",
  "views": {
    "front": {"mask": "front-mask.png", "mm_per_pixel": 0.8},
    "back": {"mask": "back-mask.png", "mm_per_pixel": 0.8}
  }
}
```

```sh
python -m superarchiveallclo.compare_shape build/tee-01 \
  data/tee-01/shape-reference.json -o build/tee-01-shape
```

工具会用 Blender 回读归档中的实际 GLB，按正交视角投影。参考和模型在每像素 2 mm 的网格上仅做平移对齐。报告列出宽高毫米差、轮廓重合度、双向边界距离的中位数和 P90，并保存参考图、GLB 和纸样哈希。叠图用蓝色显示参考独有区域，用橙色显示模型独有区域，灰黑色为重合区域。

轮廓只衡量对应拍摄视角可见的形状。实物的材质、内部构造和不可见的表面仍需独立核对。已用相同合成轮廓校验重合度为 1.0，并用故意放大 10% 的侧视参考验证尺度误差会保留。

有标定平铺点云时，可用 `superarchiveallclo.compare_flat` 从同版纸样生成对应姿态的 GLB，对独立复拍计算可见点到表面和轮廓边缘距离。配套数字网格带有展开 UV 时，`superarchiveallclo.compare_pattern` 可量出净线裁片误差并核对跨片连接。[真实 T 恤的对照结果](real-reference-comparison.md)

有标定抓取点和独立悬挂点云时，`superarchiveallclo.compare_hang` 可评估同版网格在两个绝对支撑点约束下的重力仿真。配套相机外参可逐射线检查可见布面深度。还可用提前拍摄的点云预测短时摆动。[实物悬挂对照](suspension-study.md) · [逐射线深度与短时摆动](measured-depth-study.md)

有同一姿态的近正交多视角挂拍时，`superarchiveallclo.visual_hull` 可以结合每帧独立尺度和同一人体基准生成毫米三维外包络。报告包含留出照片的轮廓误差、GLB 超出外包络的面积与距离，以及各高度区间的前后深度。可见布面深度由带标定的三维设备独立核对。[馆藏实物三维外包络](mannequin-study.md) · [同版垂坠和分段深度](drape-support-study.md)

## 公开数据核对

- GarmentCodeData 提供参数化纸样及模拟服装。可用作制板和渲染对照，真实 T 恤验收仍需同一件实物的照片与尺寸。[GarmentCode 官方资料](https://github.com/maria-korosteleva/GarmentCode)
- DeepFashion2 收录商家及消费者照片，并提供遮挡、关键点和像素掩码。公开标注说明中的尺度是图像内尺度分档，没有对应的实物毫米制板数据。[DeepFashion2 标注与访问说明](https://github.com/switchablenorms/DeepFashion2)
- SAFT 的真实评测提供布料的 RGB、深度及点云。可用于研究布料形变，现有说明没有可直接用于本项目盲测的同款 T 恤纸样和实测尺寸。[SAFT 论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Stotko_SAFT_Shape_and_Appearance_of_Fabrics_from_Template_via_Differentiable_ICCV_2025_paper.pdf)
- LiDAR 服装测量研究的两个公开实验各列出 330 张 RGB 图及深度、PLY 和 COCO 标注。每份 COCO 标注包含 100 条短袖上衣记录，各自对应不同的 `pair_id`。图像字段未给出正背面，标注未提供实测毫米尺寸或版片，两份 JSON 的许可列表也为空。该来源可继续用于核对单视图测量方法，实物归档验收仍需要同件正背照、独立尺寸、结构证据及使用授权。[研究说明](https://github.com/ZinoStudio931/Automatic-Garments-Size-Measurement-using-HRNet-and-Point-Cloud) · [公开实验目录](https://drive.google.com/drive/folders/1ePJE7KybnQlHRa-4QxiX-jPfpIYKf_Oa)
- Cloth2Tex 提供把参数化网格配准到商品图并补全纹理的方法。其项目说明提供外观对照方向，尚未核实可用于本项目实物验收的同件实测尺寸和对应版片。[Cloth2Tex 官方实现](https://github.com/HumanAIGC/Cloth2Tex)
- Kitanesque 的旧衣数据提供 558 件商品的卷尺记录和商品链接，表格许可为 CC BY 4.0。实测表没有配套的标定正背照，商品照片的使用范围也需单独查证。[实测数据表](https://www.kaggle.com/datasets/andradalcaran/vintage-clothing-sizing-label-vs-tape)
- Trossen Robotics 的 T 恤折叠数据提供 Apache-2.0 许可的机器人拍摄视频，数据卡未列出同件成衣的毫米实测、纸样或标准正背面采集。[数据卡](https://huggingface.co/datasets/TrossenRoboticsCommunity/aloha_fold_tshirt)
- V&A 馆藏抽查有单件 T 恤的毫米可换算尺寸。O1782036 记录高 66 cm、肩宽 35.7 cm。三张馆藏图为正面和印花细节，未见背照和可用标尺。馆藏元数据与图像用途依馆方条款核对。[馆藏对象记录](https://api.vam.ac.uk/v2/object/O1782036) · [馆方使用条款](https://www.vam.ac.uk/info/va-websites-terms-conditions)
- RGBench 的公开 Sim2Real 数据提供同件 T 恤的标定实物点云和 Style3D 数字参考。已量出平铺三维、裁片净线、连接关系与实物袖口的误差。扫描未提供带正背标签的平铺配对及卷尺记录。[RGBench 最新结果](sleeve-join-study.md)
- CloPeMa Clothes 提供真实 T 恤的 RGB、深度和分割图，许可为 CC BY 4.0。已在两件圆领 T 恤上运行注册深度通道并生成候选归档。下载的 Kinect 归档未附该相机的独立内参或服装实测值，焦距敏感性和数据范围有单独记录。[深度实验与复现](rgbd-study.md) · [格拉斯哥大学数据档案](https://researchdata.gla.ac.uk/270/)
- PatternedClothing 提供同一研究 T 恤的裁片模型和实拍注册网格，许可为 CC BY-NC 4.0。Open+Pattern 提供 CC BY 4.0 的 T 恤 DXF、CLO 文件和 GLB，其公开产品页列出的图像为插图。两者分别提供动态几何和独立制板参考。[PatternedClothing](https://github.com/facebookresearch/PatternedClothing) · [Open+Pattern](https://openpattern.io/product/standard-fit-t-shirt/)
- Smithsonian 藏品 `nmah_460141` 有馆方标注的同件正背照片和整体尺寸 63.50 × 68.58 × 5.08 cm。该页面的 CC0 适用于目录元数据，照片另列使用条件。已核查目录和照片标注。[Smithsonian 藏品记录](https://www.si.edu/object/t-shirt:nmah_460141)
- Wikimedia Commons 的鹿儿岛 OpenStreetMap 调查 T 恤提供同件正背图，许可为 CC BY-SA 4.0。已验证可见衣身的相对比例，并检查到袖缘贴图边。该对照还需要可信尺度和独立原件尺寸。[正背照报告](sleeve-join-study.md#有授权的同件正背照)
- 奥克兰博物馆的两件圆领 T 恤提供 CC BY 4.0 的同件正背照。一件带独立标尺，另一件有馆方尺寸、独立内部结构照和额外的近正交挂拍。两件均已生成候选归档。挂拍支持毫米外包络对照，原版纸样和直接三维布面测量仍待核对。[正背照盲测](paired-photo-study.md) · [人台三维对照](mannequin-study.md)
- Deep Fashion3D V2 提供真实服装点云、注册网格和领口等边缘标注，许可为 CC BY-NC 4.0。归档的官方下载页面要求申请解压密码，发布说明未列出独立卷尺尺寸或原厂纸样。[发布说明](https://kv2000.github.io/2023/06/20/deepFashion3DV2/) · [下载与许可](https://github.com/GAP-LAB-CUHK-SZ/deepFashion3D)
- 犹他大学的双层标记服装采集项目已下载核对服装模板和长袖 T 恤的 16 视角同步样本。该帧标定文件及独立测得的布面有待取得。CTU Spread Garments 的说明提供短袖正背标签和深度，完整发布入口本轮暂不可用。[两个实拍来源的核查记录](drape-support-study.md#其他实拍来源)
