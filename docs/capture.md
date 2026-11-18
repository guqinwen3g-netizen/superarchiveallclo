# 从实物到数字档案

第一个闭环以一件单层圆领 T 恤为对象。店员拍摄和补测，程序归档原始证据，从已知数据重建候选纸样，再用该纸样的净缝线和配缝关系驱动 Blender Cloth。摄影和卷尺可完成初始采集。扫描、夹克等复杂品类进入后续扩展。

标准正背照可以直接走模型入口。相机尺度可由程序识别品红色标定卡，也可以来自同平面直尺或已标定的正投影。制板尺寸由训练权重预测。以下量法用于补充可信实测和独立验收。[照片拍摄与模型运行](vision-model.md)

## 在店里采集

1. **平铺正背面。** 面料自然展开，袖子摆正。手机镜头尽量垂直于衣物，用柔和均匀的光，拍全衣服和同平面的直尺。每面各保留一张原图。图上的尺段尽量长，分别记录两个端点和对应毫米数。
2. **补结构和面料证据。** 拍领口、袖口、下摆、内侧缝、面料标签。再拍一张挂起的正面和侧面，留给仿真对照。可记录成分、克重、厚度和双向伸长实测值。缺少实测时，候选模型使用参考针织参数。
3. **量关键尺寸。** 用卷尺填毫米数，或者给标定照片中的两个像素点。水平或垂直尺寸分别用 `axis: "x"` 或 `axis: "y"`。像素从按 EXIF 方向摆正的图像左上角开始。弧线、遮挡部位和弹性最好用实物补测。
4. **保留验证照片。** 挂拍或试穿参考可以附在归档里。实样比对时用相同视角、比例和悬挂条件，区分外观差异和拍摄差异。

## 尺寸定义

| 字段 | 量法 |
| --- | --- |
| `flat_chest_mm` | 平铺从一侧腋下到另一侧腋下 |
| `flat_hem_mm` | 平铺下摆折线两端的水平宽度 |
| `back_length_mm` | 后领净缝中心到下摆折线，垂直 |
| `garment_shoulder_width_mm` | 左右肩缝与袖窿交点的水平距离 |
| `shoulder_slope_mm` | 侧颈点到肩袖交点的垂直落差，取左右平均 |
| `armhole_depth_mm` | 侧颈点水平线到腋下水平线的垂直距离 |
| `sleeve_length_mm` | 肩袖交点到袖口折线，沿袖子中心方向 |
| `sleeve_bicep_flat_mm` | 袖展开后的上臂宽度，可用成衣此处一整圈周长 |
| `sleeve_opening_flat_mm` | 袖展开后的袖口宽度，可用袖口一整圈周长 |
| `neck_width_mm` | 左右领肩交点间的水平距离 |
| `front_neck_depth_mm` / `back_neck_depth_mm` | 从侧颈点水平线到对应领口净缝中心点 |
| `collar_finished_width_mm` | 领条折好后的宽度，缺省 18 mm |
| `collar_length_ratio` | 未拉伸领条净长除以领口净缝全长，缺省 0.85 |

## 采集文件

运行 `python -m examples.make_capture_fixture build/capture-fixture` 可查看完整例子。实物档案把 `kind` 设为 `physical`，将 JPEG 或 PNG 和 `session.json` 放在同一目录。正背面分别放在 `views.front` 和 `views.back`，其余视角按需要附上。每张图的 `file` 相对于采集文件。标定格式为 `"scale": {"mm": 200, "points": [[80, 450], [380, 450]]}`。

正投影可以直接填写 `"calibration": {"mm_per_pixel": 2, "source": "记录相机、深度或校准方法"}`。同一图选择一种标定方式，像素尺度以按 EXIF 扶正的原图为准。直尺必须与衣服同平面，长度独立于待估计的衣服尺寸。斜拍、有透视变化的图像需要先校正。已有分割好的平铺点云时可以用 `superarchiveallclo.scan` 生成带比例的正投影和采集记录。[真实点云输入与复现](real-scan-study.md)

已有馆方尺寸和档案照片时，单次研究可以把可见的实测线段用作定尺度依据。在 `scale` 下写 `source` 和 `basis_field`，例如用 `back_length_mm` 标出已用的后中衣长。标识过的定尺度尺寸会从独立测试评分排除。请保留测点定义和拍摄条件。[真实正背照与目录尺寸对照](paired-photo-study.md)

### 已注册的深度照片

记录在 `rgbd.json`。每个视角的颜色、深度和可选分割图须在同一个像素网格上，与扶正后的颜色图和输入内参对应。内参以实际标定记录为准。先用设备校正畸变和注册，深度值采用相机光轴方向的距离，单位可为 `mm` 或 `m`。支持 16 位 PNG 和数值 NPY。提供多标签 PNG 时可以用 `mask_label` 选中服装。

```json
{
  "schema_version": 1,
  "item_id": "tee-001",
  "source": "采集设备、标定日期及内参来源",
  "views": {
    "front": {
      "rgb": "front.jpg",
      "depth": "front-depth.png",
      "depth_unit": "mm",
      "intrinsics": {"fx": 600, "fy": 600, "cx": 319.5, "cy": 239.5},
      "mask": "front-mask.png"
    }
  }
}
```

示例中的内参仅用于展示字段。拍摄时让衣物朝上摆正并保留四周可见台面，程序用背景深度确定台面方向，再按毫米生成俯视投影。拍到同件背面时，在 `views` 下加 `back`，使用双面模型。只有单面时使用单面模型。省略 `--model` 可先生成投影及采集记录。启用模型时，原始彩照、深度图和分割证据会随候选归档保存，校验命令可复核哈希。

```sh
python -m superarchiveallclo.rgbd rgbd.json \
  --model models/tee-vision-front.npz -o build/registered-tee
```

### 尺寸与材料

尺寸键放在 `measurements` 下。卷尺实测可填 `{"value": 560}`。馆方实测可加 `"method": "catalogue_measurement"` 和 `"source"` 来记录原始目录。照片标注可填 `{"view": "front", "points": [[120, 300], [960, 301]], "axis": "x"}`。材料观察放在 `material`，随原始记录保留。每个字段的来源和推断项会写入输出档案。

对侧缝近似直线、腋下与下摆都可见的照片，可在采集文件顶层填 `"body_profile": "paired_silhouette"`，由双面模型分别估计胸宽和下摆宽。两面估计相符并通过配缝时，纸样会使用两种宽度。归档保留轮廓取点和来源。证据不足时使用基础模型。也可直接提供可信的 `flat_hem_mm`。[衣身轮廓实物实验](body-profile-study.md)

## 交付与验收

- **原件证据**存放在 `photos/` 和 `capture.json`。包括原图 SHA-256、标定点、尺寸证据及参考值。
- **二维纸样**存放在 `pattern/`。DXF 为毫米单位，按裁片分块，带裁剪、净缝、定位和折线图层，净缝保留曲线。配缝方向和边标识记录在 JSON。A4 PDF 可 1∶1 拼页。专用服装 CAD 的 DXF-AAMA/ASTM 往返需要用目标软件另行验证。
- **三维模拟**存放在 `model/`。对应的纸样边先匹配并共享网格顶点，再以平面边长拟合缝合状态，最后交给 Blender Cloth 在参考人体上计算垂坠。程序校验缝合面保留领口、两个袖口和下摆四个开口，并用纸样哈希关联 GLB 和可编辑 Blender 项目。报告包含相对纸样的网格边长偏差，以及求解及导出阶段的参考人体顶点穿入距离。渲染使用参考棉针织、罗纹领条和 1 mm 布层厚度。

对照无手臂人台的挂拍时，可给 `superarchiveallclo.capture` 加 `--form torso_only`。默认使用带上臂支撑的参考人台。所选人台记录在 `model/simulation.json`。[两种人台的同版对照](mannequin-study.md)

`--form display_torso` 使用无臂人台的接触支撑，并让袖子从侧下方起算。可以用 `--drape-frames` 调整计算时长。归档报告末段顶点位移，帮助复核导出时的垂坠状态。[接触支撑、时长及实物误差](drape-support-study.md)

给 `superarchiveallclo.capture` 加 `--photo-appearance`，可从两张不同的平铺正背照提取衣身颜色、文字和图案。程序将图像映射到纸样坐标，再随缝合和垂坠后的模型导出。袖片和领条使用取样颜色。原图照明和部分折痕也会留在贴图上。归档中的 `model/appearance.json` 记录两张照片哈希、衣身取点和贴图区，`check_archive` 会回读 GLB 并校验嵌入图像。[实物示例与复现](photo-appearance-study.md)

真实闭环通过的依据是同一件衣服的正背和侧面，纸样按原件测量复核，3D 在同样视角和尺度下对照挂拍或试穿，再记录版型误差、结构差异和材质差异。合成样例用于验证文件、尺度、配缝与运行通道。摆放角度、未见的构造与布料物性仍需实物对照。
