# superarchiveallclo

**Super Archive All Clone** 为真实服装建立数字档案。从实物拍摄、测量或扫描开始，为每件衣服保存原始证据、可编辑的二维纸样，以及由同一套裁片缝合并模拟出来的 3D 模型。长期目标是覆盖各类服装，并交付经实物打样和目标 CAD 验证的制板文件。

**当前开发范围为单层圆领 T 恤的归档闭环。** 先让带尺寸证据的采集档案生成纸样和同版 3D 模型，再用真实样衣核对尺寸、结构和视觉。首个风格方向为 **Clean Fit / Minimalism 极简男装穿搭**。

首款提议用一件微落肩圆领 T 恤，把真实照片、纸样、仿真和试穿走通。随后加入直筒长裤组成第一套穿搭，再用外搭衬衫验证叠穿。开发从可自己生成、自己测量的样本开始。

## 跑通一份归档

安装 Python 依赖，确保 Blender 4.3+ 可通过 `blender` 调用。以下命令会生成带标签的合成采集素材，并使用和实物采集相同的入口执行制板和布料求解。

```sh
python -m pip install -r requirements.txt
python -m examples.make_capture_fixture build/capture-fixture
python -m superarchiveallclo.capture build/capture-fixture/session.json -o build/archive
```

归档包括带来源的原图和尺寸、DXF / SVG / A4 纸样、配缝表、GLB、可编辑 Blender 场景与三张视角图。用实物采集目录替换合成样例即可生成候选档案。[拍摄步骤、量法与验收方式](docs/capture.md) · [合成样例的实际仿真图](examples/archive-preview.jpg)

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
