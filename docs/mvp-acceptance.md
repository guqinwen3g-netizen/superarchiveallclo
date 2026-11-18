# 实物归档 MVP 验收

首件目标是一件有使用权、独立于训练的单层圆领 T 恤。从带标定卡的实拍正背照运行模型，生成候选纸样和同版 3D。用原件的实测尺寸核对纸样，用结构近照或原始版片核对缝合，用同姿态和尺度的参考图对照 3D。

| 工作 | 验收证据 |
| --- | --- |
| 样本采集 | 同一件衣服的正背原图、比例卡、卷尺实测和结构证据。记录使用范围与拍摄条件 |
| 盲测 | 保存实测标签和模型预测。每个尺寸报告毫米误差，对所有拒绝项计数。以款式或实物为组隔离训练、验证和盲测 |
| 版片 | 报告裁片闭合及配缝。回读 DXF 的毫米单位、裁片与裁剪线。依据独立原件证据核对袖窿、领口、下摆、领条等构造 |
| 3D | 用同版纸样哈希核对来源，确认 GLB 可读。以相同视角对照实物形态，注明材质先验、轮廓误差和未确认的细节 |

当前仓库的照片样例和基准权重基于合成样本，可用于回归入口。实物验收需要一份单独的采集档案。最终报告记录归档版本、样衣标识、已核对部位和剩余误差。

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

## 公开数据核对

- GarmentCodeData 提供参数化纸样及模拟服装。可用作制板和渲染对照，真实 T 恤验收仍需同一件实物的照片与尺寸。[GarmentCode 官方资料](https://github.com/maria-korosteleva/GarmentCode)
- DeepFashion2 收录商家及消费者照片，并提供遮挡、关键点和像素掩码。公开标注说明中的尺度是图像内尺度分档，没有对应的实物毫米制板数据。[DeepFashion2 标注与访问说明](https://github.com/switchablenorms/DeepFashion2)
- SAFT 的真实评测提供布料的 RGB、深度及点云。可用于研究布料形变，现有说明没有可直接用于本项目盲测的同款 T 恤纸样和实测尺寸。[SAFT 论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Stotko_SAFT_Shape_and_Appearance_of_Fabrics_from_Template_via_Differentiable_ICCV_2025_paper.pdf)
- LiDAR 服装测量研究公布了 330 组 RGB、深度、点云和关键点标注。该方法从单张平铺照测量，公开目录尚未确认同一原件的正背照、实测纸样和明确的数据再使用条款。可继续核对它是否适合单视图测量对照。[研究仓库与数据说明](https://github.com/ZinoStudio931/Automatic-Garments-Size-Measurement-using-HRNet-and-Point-Cloud)
