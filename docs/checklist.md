# 验收清单

- [ ] 选择并下载 GC10-DET 或 Severstal。
- [ ] 保留 `data/raw/`。
- [ ] 输出 `raw_annotation_summary.csv`。
- [ ] 删除损坏图片、重复图片、无有效标注图片。
- [ ] 过滤过小标注、越界框、面积为 0 的框。
- [ ] 将图像改为正方形，并同步更新标注。
- [ ] 输出 YOLO txt 或 COCO json。
- [ ] 使用固定随机种子划分 train / val / test。
- [ ] 至少实现 4 类数据增强。
- [ ] 增强后仍保证每张图像至少 1 个有效标注。
- [ ] 保存增强前后对比图到 `figures/augmentation_examples/`。
- [ ] 提交 `dataset_report.md`，包含分散粒度表。
- [ ] 提交 `audit_report.md`，包含泄漏、脏标签和标注质量审计。
