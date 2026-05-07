# GC10-DET 钢材表面缺陷检测数据整理

## 基本信息

- 姓名：施哲旭
- 数据集：GC10-DET
- 原始标注格式：Pascal VOC XML
- 输出标注格式：YOLO txt
- 原始数据目录：`data/raw/GC10-DET/`
- 处理后数据目录：`data/processed/`
- 固定随机种子：`20260430`
- 划分比例：train / val / test = 70% / 15% / 15%

## 数据来源

数据来自 GC10-DET。原始文件保留在 `data/raw/GC10-DET/`，图片位于 `1/` 到 `10/` 类别目录，VOC XML 标注位于 `lable/`。原始压缩包保留在 `data/raw/GC10-DET.zip`。

- 下载来源：https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets
- 论文来源：https://doi.org/10.3390/s20061562
- 授权与引用：原作者说明可用于研究，使用时应引用对应论文；这里仅整理本地提供版本。

原始数据统计：图片文件 2312 个，XML 文件 2294 个，原始标注框 3564 个。类别 10 的拼写变体已统一，未知标签已作为脏标签剔除。

## 处理流程

1. 读取 `data/raw/GC10-DET/` 原始目录。
2. 解析 VOC XML，输出 `data/processed/meta/raw_annotation_summary.csv`。
3. 校验图片是否可打开，删除损坏样本。
4. 基于 MD5 与 dHash 删除重复和近似重复样本。
5. 删除未知类别、越界框、零面积框，以及原图宽或高小于 5 px 的过小框。
6. 删除清洗后没有有效标注的图片。
7. 按采集键分组，再用固定随机种子分层划分 train / val / test。
8. 使用灰色 padding 转为正方形，并缩放到 1024x1024，同步更新标注。
9. 仅对训练集生成增强样本，并再次校验增强后标签。

## 数据增强

增强方法包括水平翻转、亮度/对比度扰动、随机缩放平移和轻微旋转。成功生成训练增强图 1489 张，增强前后对比图见 `figures/augmentation_examples/examples.jpg`。

## 输出结构

```text
data/processed/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── meta/
```

最终 split 统计：

| split | image_count | annotation_count | avg_annotations_per_image |
| --- | --- | --- | --- |
| train | 2980 | 4540 | 1.523 |
| val | 338 | 537 | 1.589 |
| test | 370 | 644 | 1.741 |

## 报告与复查文件

- 数据处理报告：`dataset_report.md`
- 数据质量审计报告：`audit_report.md`
- 原始标注解析表：`data/processed/meta/raw_annotation_summary.csv`
- 处理后标注明细：`data/processed/meta/processed_annotations.csv`
- 删除记录：`data/processed/meta/removed_records.csv`
- 每类带框样本：`figures/samples/`
- 清洗前后对比：`figures/clean/before_after.jpg`
- 正方形化对比：`figures/square/before_after.jpg`
- 增强对比：`figures/augmentation_examples/examples.jpg`
- 审计证据图：`figures/audit/audit_findings.jpg`
- 统计图表：`figures/chart/`

## 复现命令

运行环境为本地 conda 环境 `SZX-project`，主要依赖包括 `Pillow`、`pandas`、`matplotlib`。

```powershell
conda activate SZX-project
python scripts/main.py
```
