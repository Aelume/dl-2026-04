# GC10-DET 数据处理报告

## 数据源与标注格式

数据使用 GC10-DET 钢材表面缺陷检测数据集。原始文件保留在 `data/raw/GC10-DET/`，图片按缺陷类别目录存放，标注文件在 `lable/`，格式为 Pascal VOC XML。

- 下载来源：原作者公开仓库 `https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets`
- 论文来源：Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network, Sensors 2020, `https://doi.org/10.3390/s20061562`
- 授权与引用：原作者说明可用于研究，使用时应引用对应论文；本项目仅对本地提供版本做格式整理与质量审计。
- 原始图片文件数：2312
- 原始 XML 文件数：2294
- 原始可匹配有标注图片记录数：2294
- 原始无 XML 图片文件估计数：18
- 原始标注框数：3564
- 输出格式：YOLO txt，方便直接喂给 YOLO 训练，也便于逐图抽查标签。

类别 10 的 `10_yaozhed` 被视为 `10_yaozhe` 的原始拼写变体并统一为 `waist_folding`；孤立标签 `d` 不在 GC10 类别定义中，按脏标签删除。

## 清洗规则

- 损坏图片：使用 Pillow 完整打开和 `verify()` 校验，无法解码的图片删除。本次 XML 可匹配图片均可正常解码；`2312` 到 `2294` 的差异主要来自原始目录中估计 `18` 张无 XML 标注图片未进入有标注数据集。
- 重复图片：先按文件 MD5 删除完全重复记录，再按 16x16 dHash 删除汉明距离不超过 2 的近似重复记录。
- 无有效标注：未知类别、越界框、宽高非正和过小框过滤后，如果图片不再包含有效框，则删除图片。
- 过小框阈值：原始坐标中宽或高小于 5 px 的框删除。GC10 原图为 2048x1000，该阈值主要剔除标注噪声，同时保留肉眼可辨的小缺陷。
- 随机划分：固定随机种子 `20260430`，先按采集键分组，再按组内主类别分层划分 train / val / test = 70% / 15% / 15%。
- 正方形化：使用灰色 padding 到正方形，再缩放到 1024x1024；所有框同步执行 `(x + pad_x) * scale` 和 `(y + pad_y) * scale`。

## 数据增强

增强只用于训练集。每张训练图最多生成 1 张增强图，四种方法按固定顺序轮换：水平翻转、亮度/对比度扰动、随机缩放平移、轻微旋转。几何增强后重新计算框坐标并裁剪到图像边界，空标签文件不写出。

- 增强方法：flip, light, scale, rotate
- 成功增强图片数：1489
- 跳过增强计数：scale 2 张

选水平翻转、亮度扰动和小幅几何变化，是因为这些接近实际拍摄时相机方向、曝光和钢板位置的差异。旋转和裁剪幅度不能太大，不然容易把长条状缺陷切断，所以只用了轻量版本。

## 多维统计表

### 按 split 统计

| split | image_count | annotation_count | avg_annotations_per_image |
| --- | --- | --- | --- |
| train | 2980 | 4540 | 1.523 |
| val | 338 | 537 | 1.589 |
| test | 370 | 644 | 1.741 |

### 按类别统计

| class_id | class_name | class_name_cn | image_count | annotation_count | annotation_ratio |
| --- | --- | --- | --- | --- | --- |
| 0 | punching_hole | 冲孔 | 543 | 543 | 0.0949 |
| 1 | welding_line | 焊缝 | 858 | 860 | 0.1503 |
| 2 | crescent_gap | 月牙弯 | 443 | 445 | 0.0778 |
| 3 | water_spot | 水斑 | 538 | 605 | 0.1058 |
| 4 | oil_spot | 油斑 | 392 | 868 | 0.1517 |
| 5 | silk_spot | 丝斑 | 1109 | 1377 | 0.2407 |
| 6 | inclusion | 异物 | 313 | 524 | 0.0916 |
| 7 | rolled_pit | 压痕 | 76 | 146 | 0.0255 |
| 8 | crease | 折痕 | 85 | 118 | 0.0206 |
| 9 | waist_folding | 腰折 | 229 | 235 | 0.0411 |

### 按缺陷尺寸统计

尺寸分桶采用 COCO 常用面积尺度并应用在 1024x1024 输出图上：small 为面积小于 32^2 px，medium 为 32^2 到 96^2 px，large 为不小于 96^2 px。

| size_bucket | annotation_count | threshold |
| --- | --- | --- |
| small | 477 | area < 32^2 px |
| medium | 1647 | 32^2 <= area < 96^2 px |
| large | 3597 | area >= 96^2 px |

### 按清洗阶段统计

| stage | image_count | annotation_count | note |
| --- | --- | --- | --- |
| raw | 2312 | 3564 | 原始图片文件与 VOC 标注框 |
| after_corrupt_check | 2294 | 3564 | 完成 XML 可匹配图片的可读性检查；无 XML 图片不进入后续有标注处理 |
| after_duplicate_check | 2201 | 3564 | 删除完全重复与近似重复记录 |
| after_invalid_label_filter | 2199 | 3457 | 删除未知、越界、零面积与过小标注 |
| after_square_transform | 2199 | 3457 | padding 为正方形并缩放到 1024 |
| after_train_augmentation | 3688 | 5721 | 对符合条件的训练图生成增强样本 |

### 按图像粒度统计

| defects_per_image | image_count |
| --- | --- |
| 1 | 2371 |
| 2-3 | 1162 |
| 4+ | 155 |

## 数据问题观察

1. 原始数据中有 `88` 个 XML 需要通过文件名回补图片路径，说明原始 `folder` 字段不能完全信任。
2. 原始类别名存在 `10_yaozhe` / `10_yaozhed` 两种写法，已统一；未知标签 `d` 已删除。
3. 类别分布不均衡，`silk_spot`、`oil_spot` 明显多于 `rolled_pit`、`crease`，训练时可考虑类别重采样或损失权重。
4. `waist_folding` 的框普遍较大，包含较多背景，后续训练前建议人工复查该类边界是否需要收紧。

## 可视化文件

- 每类带框样本：`figures/samples/`
- 清洗前后对比：`figures/clean/before_after.jpg`
- 正方形化前后对比：`figures/square/before_after.jpg`
- 增强前后对比：`figures/augmentation_examples/examples.jpg`
- 审计证据图：`figures/audit/audit_findings.jpg`
- 图表：`figures/chart/`

## 复查表

- 原始标注解析：`data/processed/meta/raw_annotation_summary.csv`
- 处理后标注明细：`data/processed/meta/processed_annotations.csv`
- 删除记录：`data/processed/meta/removed_records.csv`
- 分层统计表：`data/processed/meta/*.csv`

## 复现方法

运行环境为本地 conda 环境 `SZX-project`，主要依赖包括 `Pillow`、`pandas`、`matplotlib`。复现时将原始 GC10-DET 文件放在 `data/raw/GC10-DET/` 后执行：

```powershell
conda activate SZX-project
python scripts/main.py
```
