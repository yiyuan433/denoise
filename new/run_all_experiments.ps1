# 完整实验自动化脚本（Windows PowerShell 版本）
# 用法：.\run_all_experiments.ps1

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "DAS 降噪论文 - 完整实验流程" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "开始时间: $(Get-Date)" -ForegroundColor Yellow
Write-Host ""

$ErrorActionPreference = "Stop"

# ============ 第1阶段：训练我们的模型 ============
Write-Host "========== 第1阶段：训练 Wavelet-Transformer 模型 ==========" -ForegroundColor Green
foreach ($dataset in @("eq-36", "eq-68", "slice_german", "slice_german_1")) {
    Write-Host ">>> 训练 Ours 模型 on $dataset ..." -ForegroundColor Yellow
    python train_2d.py --model ours --dataset $dataset --epochs 100
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
Write-Host "✓ 我们的模型训练完成" -ForegroundColor Green
Write-Host ""

# ============ 第2阶段：训练基线模型 ============
Write-Host "========== 第2阶段：训练基线模型 DnCNN ==========" -ForegroundColor Green
foreach ($dataset in @("eq-36", "eq-68", "slice_german", "slice_german_1")) {
    Write-Host ">>> 训练 DnCNN on $dataset ..." -ForegroundColor Yellow
    python train_2d.py --model dncnn --dataset $dataset --epochs 100
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
Write-Host "✓ DnCNN 训练完成" -ForegroundColor Green
Write-Host ""

Write-Host "========== 第2阶段：训练基线模型 U-Net ==========" -ForegroundColor Green
foreach ($dataset in @("eq-36", "eq-68", "slice_german", "slice_german_1")) {
    Write-Host ">>> 训练 U-Net on $dataset ..." -ForegroundColor Yellow
    python train_2d.py --model unet --dataset $dataset --epochs 100
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
Write-Host "✓ U-Net 训练完成" -ForegroundColor Green
Write-Host ""

# ============ 第3阶段：推理对比实验 ============
Write-Host "========== 第3阶段：运行推理对比实验 ==========" -ForegroundColor Green
Write-Host ">>> 运行合成噪声 + 现场数据对比 ..." -ForegroundColor Yellow
python paper_pipeline.py --mode all --plot-robustness
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "✓ 推理对比完成" -ForegroundColor Green
Write-Host ""

# ============ 第4阶段：消融实验 ============
Write-Host "========== 第4阶段：运行消融实验 ==========" -ForegroundColor Green
foreach ($dataset in @("eq-36", "eq-68", "slice_german", "slice_german_1")) {
    Write-Host ">>> 消融实验 on $dataset ..." -ForegroundColor Yellow
    python ablation_runner.py --dataset $dataset --noise 0.1 --epochs 80
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
Write-Host "✓ 消融实验完成" -ForegroundColor Green
Write-Host ""

# ============ 第5阶段：收集结果表格 ============
Write-Host "========== 第5阶段：收集论文用表格 ==========" -ForegroundColor Green
python paper_pipeline.py --mode collect --collect-ablation
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "✓ 表格收集完成" -ForegroundColor Green
Write-Host ""

# ============ 总结 ============
Write-Host "========== ✓ 所有实验完成 ==========" -ForegroundColor Green
Write-Host "结束时间: $(Get-Date)" -ForegroundColor Yellow
Write-Host ""
Write-Host "📋 重要输出位置：" -ForegroundColor Cyan
Write-Host "  1. 论文用表格：        paper_outputs/tables_for_paper.md" -ForegroundColor White
Write-Host "  2. 图表清单：          paper_outputs/figure_manifest.md" -ForegroundColor White
Write-Host "  3. 原始实验数据：      results/ 目录" -ForegroundColor White
Write-Host "  4. 可视化图表：        figures/ 目录" -ForegroundColor White
Write-Host "  5. 训练好的模型：      checkpoints/ 目录" -ForegroundColor White
Write-Host ""
Write-Host "下一步：" -ForegroundColor Cyan
Write-Host "  1. 查看 paper_outputs/tables_for_paper.md" -ForegroundColor White
Write-Host "  2. 复制表格到 paper_v2.md" -ForegroundColor White
Write-Host "  3. 插入图表生成完整论文" -ForegroundColor White
Write-Host ""
