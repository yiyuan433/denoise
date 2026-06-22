#!/bin/bash
# 完整实验自动化脚本
# 用法：bash run_all_experiments.sh

set -e  # 有错误时立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================="
echo "DAS 降噪论文 - 完整实验流程"
echo "=================================="
echo "开始时间: $(date)"
echo ""

# ============ 第1阶段：训练我们的模型 ============
echo "========== 第1阶段：训练 Wavelet-Transformer 模型 =========="
for dataset in eq-36 eq-68 slice_german slice_german_1; do
  echo ">>> 训练 Ours 模型 on $dataset ..."
  python train_2d.py --model ours --dataset $dataset --epochs 100 || exit 1
done
echo "✓ 我们的模型训练完成"
echo ""

# ============ 第2阶段：训练基线模型（可选）============
echo "========== 第2阶段：训练基线模型 DnCNN =========="
for dataset in eq-36 eq-68 slice_german slice_german_1; do
  echo ">>> 训练 DnCNN on $dataset ..."
  python train_2d.py --model dncnn --dataset $dataset --epochs 100 || exit 1
done
echo "✓ DnCNN 训练完成"
echo ""

echo "========== 第2阶段：训练基线模型 U-Net =========="
for dataset in eq-36 eq-68 slice_german slice_german_1; do
  echo ">>> 训练 U-Net on $dataset ..."
  python train_2d.py --model unet --dataset $dataset --epochs 100 || exit 1
done
echo "✓ U-Net 训练完成"
echo ""

# ============ 第3阶段：推理对比实验 ============
echo "========== 第3阶段：运行推理对比实验 =========="
echo ">>> 运行合成噪声 + 现场数据对比 ..."
python paper_pipeline.py --mode all --plot-robustness || exit 1
echo "✓ 推理对比完成"
echo ""

# ============ 第4阶段：消融实验 ============
echo "========== 第4阶段：运行消融实验 =========="
for dataset in eq-36 eq-68 slice_german slice_german_1; do
  echo ">>> 消融实验 on $dataset ..."
  python ablation_runner.py --dataset $dataset --noise 0.1 --epochs 80 || exit 1
done
echo "✓ 消融实验完成"
echo ""

# ============ 第5阶段：收集结果表格 ============
echo "========== 第5阶段：收集论文用表格 =========="
python paper_pipeline.py --mode collect --collect-ablation || exit 1
echo "✓ 表格收集完成"
echo ""

# ============ 总结 ============
echo "========== ✓ 所有实验完成 =========="
echo "结束时间: $(date)"
echo ""
echo "📋 重要输出位置："
echo "  1. 论文用表格：        paper_outputs/tables_for_paper.md"
echo "  2. 图表清单：          paper_outputs/figure_manifest.md"
echo "  3. 原始实验数据：      results/ 目录"
echo "  4. 可视化图表：        figures/ 目录"
echo "  5. 训练好的模型：      checkpoints/ 目录"
echo ""
echo "下一步："
echo "  1. 查看 paper_outputs/tables_for_paper.md"
echo "  2. 复制表格到 paper_v2.md"
echo "  3. 插入图表生成完整论文"
echo ""
