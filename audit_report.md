# GC10-DET 数据质量审计报告

## 审计方法

1. 数据泄漏：对处理后的 train / val / test 图片计算 MD5 和 16x16 dHash。MD5 相同视为完全重复；同一原始采集键下 dHash 汉明距离不超过 2 视为高度相似。
2. 脏标签：复查原始类别名，检查未知类别、类别名变体和 XML 图片路径不一致问题。
3. 标注质量：统计过小、越界、面积异常大的框，抽取贴边框和大框样本用于人工复核。
4. 捷径特征：检查处理后文件名、目录、图像尺寸是否泄漏原始类别或采集批次信息。

## 发现的问题与证据

### 数据泄漏

| check | severity | sample_a | sample_b | detail |
| --- | --- | --- | --- | --- |
| leakage | pass |  |  | 未发现跨最终 train/val/test 的完全重复、同源近似重复或采集序列泄漏。 |

清洗阶段已在划分前删除 MD5 重复和 dHash 近似重复记录，并按原始采集键分组划分，因此最终干净 split 之间未发现同源序列泄漏。训练集增强图只由训练集样本生成，不参与 val / test。

### 脏标签与类别混淆

原始标注中 `10_yaozhe` 与 `10_yaozhed` 指向同一类腰折，已合并为 class 9；孤立标签 `d` 不属于 GC10 十类定义，清洗时删除。另有 XML 的 `folder` 字段与实际图片目录不完全一致，解析时优先精确匹配，失败后才使用唯一文件名回补，并把该问题记录在 `data/processed/meta/raw_annotation_summary.csv`。

### 标注框质量

| check | severity | sample | detail |
| --- | --- | --- | --- |
| dirty_label | medium | data/raw/GC10-DET/4/img_02_4406772100_00175.jpg | 原始标签未知或清洗后无有效标签，已在清洗阶段删除。 |
| metadata_mismatch | medium | data/processed/meta/raw_annotation_summary.csv | 88 条 XML 需要通过文件名回补图片路径。 |
| large_or_loose_box | low | data/processed/images/train/01530_rotate.jpg | waist_folding 标注框面积占比 0.474，建议人工复查边界。 |
| edge_touching_box | low | data/processed/images/train/00004.jpg | welding_line 标注框贴近输出图边界，需复查缺陷是否被截断。 |
| edge_touching_box | low | data/processed/images/train/00006.jpg | crescent_gap 标注框贴近输出图边界，需复查缺陷是否被截断。 |
| edge_touching_box | low | data/processed/images/test/00008.jpg | crease 标注框贴近输出图边界，需复查缺陷是否被截断。 |
| edge_touching_box | low | data/processed/images/train/00010.jpg | welding_line 标注框贴近输出图边界，需复查缺陷是否被截断。 |
| edge_touching_box | low | data/processed/images/val/00019.jpg | welding_line 标注框贴近输出图边界，需复查缺陷是否被截断。 |
| shortcut_feature | fixed | data/processed/images/ | 处理后文件名不保留原始类别目录或采集式命名，所有输出图统一为 1024x1024。 |

带框对照样图见 `figures/audit/audit_findings.jpg`，按大框、贴边和脏标签三类分块展示证据图。

其中 `low` 级别条目作为人工复查候选保留，不直接判定为错误标注；只有未知类别、越界框、零面积框和过小噪声框在自动清洗阶段被剔除。

过小阈值设为原图宽或高小于 5 px，越界框、零面积框和未知类别框都不写入最终标签。面积比例过大的 `waist_folding` 等样本保留但列入人工复查，因为此类缺陷本身可能覆盖较长钢板区域，直接删除会引入类别偏差。

### 捷径特征

原始图片按类别目录存放，如果直接用原始路径训练，模型或数据加载流程可能间接获得类别提示。处理后统一使用 `00001.jpg` 这类中性文件名，所有图片尺寸统一为 1024x1024，目录只保留 split，不保留原始类别目录。

## 修复策略

- 在划分前删除重复和近似重复样本，避免 train / test 泄漏。
- 统一类别名，删除未知类别，并保留原始标注解析表便于复核。
- 对疑似宽松的大框只做审计标记，不盲目删除；后续若人工确认边界过松，可在 XML 或 YOLO 标签上收紧后重跑脚本。
- 处理后文件名和尺寸统一，降低文件名、原始目录和尺寸带来的捷径特征。

## 修复前后数据量变化

| stage | image_count | annotation_count | note |
| --- | --- | --- | --- |
| raw | 2312 | 3564 | 原始图片文件与 VOC 标注框 |
| after_corrupt_check | 2294 | 3564 | 完成 XML 可匹配图片的可读性检查；无 XML 图片不进入后续有标注处理 |
| after_duplicate_check | 2201 | 3564 | 删除完全重复与近似重复记录 |
| after_invalid_label_filter | 2199 | 3457 | 删除未知、越界、零面积与过小标注 |
| after_square_transform | 2199 | 3457 | padding 为正方形并缩放到 1024 |
| after_train_augmentation | 3688 | 5721 | 对符合条件的训练图生成增强样本 |

## 对 mAP 的影响

去掉泄漏和脏标注后，val mAP 数值可能比用脏数据时低，但更能反映模型真实水平。清掉无效框有助于训练收敛；大框样本没有直接删，是为了不丢少数类的召回。后续训练重点看 mAP50 和 mAP50-95，尤其是样本少的 `rolled_pit`、`crease`、`waist_folding`。

## 复查文件

- 泄漏审计：`data/processed/meta/leakage_audit.csv`
- 标注质量审计：`data/processed/meta/quality_audit.csv`
- 汇总审计表：`data/processed/meta/audit_findings.csv`
- 清洗删除记录：`data/processed/meta/removed_records.csv`
- 审计证据图：`figures/audit/audit_findings.jpg`
