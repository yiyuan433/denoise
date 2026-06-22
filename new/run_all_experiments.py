#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整实验执行管理器

一键运行所有实验流程：训练 → 对比 → 消融

使用方法：
    python run_all_experiments.py --stage all
    python run_all_experiments.py --stage train
    python run_all_experiments.py --stage eval
    python run_all_experiments.py --stage ablation
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import argparse

class ExperimentManager:
    
    def __init__(self, output_log="experiment_log.txt"):
        self.output_log = output_log
        self.log_file = open(output_log, 'w', encoding='utf-8')
        self.train_datasets = ["eq-36", "eq-68"]
        self.eval_datasets = ["eq-36", "eq-68"]
        self.paper_datasets = ["eq-36", "eq-68"]
        self.ablation_datasets = ["eq-36"]
        self.models = ["ours", "dncnn", "unet"]
        self.start_time = datetime.now()
    
    def log(self, message, print_console=True):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        
        if print_console:
            print(log_msg)
        
        self.log_file.write(log_msg + "\n")
        self.log_file.flush()
    
    def run_command(self, cmd, description=""):
        """运行命令并记录"""
        
        self.log(f"\n{'='*80}")
        if description:
            self.log(f"🚀 {description}")
        self.log(f"   命令: {cmd}")
        self.log(f"{'='*80}\n")
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                self.log(f"✅ 成功：{description}")
                return True
            else:
                self.log(f"❌ 失败：{description} (返回码: {result.returncode})")
                return False
        
        except Exception as e:
            self.log(f"❌ 异常：{description} - {str(e)}")
            return False

    def _checkpoint_path(self, model_name, dataset):
        return Path("checkpoints") / f"best_{model_name}_{dataset}_ss.pth"

    def _train_command(self, model_name, dataset):
        if model_name == "ours":
            return f"python train_self_supervised.py --model ours --dataset {dataset} --epochs 100 --mask-ratio 0.05"
        return f"python train_self_supervised.py --model {model_name} --dataset {dataset} --epochs 100"

    def ensure_checkpoint(self, model_name, dataset):
        """若检查点不存在则先训练，存在则直接复用。"""
        ckpt_path = self._checkpoint_path(model_name, dataset)
        if ckpt_path.exists():
            self.log(f"♻️  已存在权重，跳过训练：{ckpt_path}")
            return True

        desc = f"{model_name.upper()} on {dataset}"
        cmd = self._train_command(model_name, dataset)
        return self.run_command(cmd, desc)

    def _artifact_exists(self, relative_path):
        return Path(relative_path).exists()

    def _run_paper_pipeline(self):
        datasets = ",".join(self.paper_datasets)
        cmd = (
            f"python paper_pipeline.py --mode all --datasets {datasets} "
            f"--comparison-noise 0.1 --collect-ablation --plot-robustness"
        )
        return self.run_command(cmd, "Paper pipeline generation")

    def _run_paper_results_extractor(self):
        return self.run_command("python paper_results_extractor.py --mode all", "Paper results extraction")

    def _run_sweep(self, mode, dataset):
        cmd = f"python paper_sweeps.py --mode {mode} --dataset {dataset} --plot"
        return self.run_command(cmd, f"Paper sweep: {mode} on {dataset}")

    def _verify_required_artifacts(self):
        required = [
            Path("paper_outputs") / "tables_for_paper.md",
            Path("paper_outputs") / "figure_manifest.md",
            Path("paper_outputs") / "run_summary.json",
            Path("paper_comparison_tables.md"),
            Path("paper_ablation_tables.md"),
            Path("results") / "eq-36_comparison_no_ref.json",
            Path("results") / "eq-68_comparison_no_ref.json",
            Path("results") / "ablation_eq-36.json",
            Path("results") / "eq-36_tv_sweep_noise0.1.json",
            Path("results") / "eq-36_patch_sweep_noise0.1.json",
            Path("figures") / "eq-36_tv_sweep_noise0.1.png",
            Path("figures") / "eq-36_patch_sweep_noise0.1.png",
        ]

        missing = [str(path) for path in required if not path.exists()]
        if missing:
            self.log("\n⚠️  关键论文产物缺失：")
            for item in missing:
                self.log(f"    - {item}")
            return False

        self.log("\n✅ 论文所需关键图表与数据已全部生成")
        return True
    
    def stage_train(self):
        """阶段 1：训练所有模型"""
        
        self.log("\n\n" + "="*80)
        self.log("📋 第一阶段：模型训练")
        self.log("="*80)
        
        success_count = 0
        total_count = 0
        
        for model_name in self.models:
            self.log(f"\n--- 训练 {model_name.upper()} ---")
            for dataset in self.train_datasets:
                total_count += 1
                if self.ensure_checkpoint(model_name, dataset):
                    success_count += 1
        
        self.log(f"\n📊 训练完成：{success_count}/{total_count} 个模型训练成功")
        return success_count == total_count
    
    def stage_eval(self):
        """阶段 2：对比实验与推理"""
        
        self.log("\n\n" + "="*80)
        self.log("📋 第二阶段：对比实验")
        self.log("="*80)
        
        success_count = 0
        total_count = 0
        
        for dataset in self.eval_datasets:
            for model_name in self.models:
                total_count += 1
                if self.ensure_checkpoint(model_name, dataset):
                    success_count += 1

            total_count += 1
            desc = f"Comparison experiment on {dataset}"
            cmd = f"python test_no_reference.py --model ours --dataset {dataset} --compare-all --include-baselines"
            
            if self.run_command(cmd, desc):
                success_count += 1
        
        self.log(f"\n📊 对比实验完成：{success_count}/{total_count} 个数据集处理成功")
        return success_count == total_count
    
    def stage_ablation(self):
        """阶段 3：消融实验"""
        
        self.log("\n\n" + "="*80)
        self.log("📋 第三阶段：消融实验")
        self.log("="*80)
        
        success_count = 0
        total_count = 0
        
        for dataset in ["eq-36"]:
            total_count += 1
            desc = f"Ablation study on {dataset}"
            cmd = f"python ablation_runner.py --dataset {dataset} --noise 0.1 --epochs 80"
            
            if self.run_command(cmd, desc):
                success_count += 1
        
        self.log(f"\n📊 消融实验完成：{success_count}/{total_count} 个数据集处理成功")
        return success_count == total_count
    
    def verify_outputs(self):
        """验证所有输出文件是否生成"""
        
        self.log("\n\n" + "="*80)
        self.log("🔍 输出文件验证")
        self.log("="*80)
        
        checks = {
            "模型权重": self._check_checkpoints(),
            "对比结果": self._check_comparison_results(),
            "消融结果": self._check_ablation_results(),
            "可视化图表": self._check_figures()
        }
        
        self.log("\n📋 验证总结：")
        for check_name, passed in checks.items():
            status = "✅ 通过" if passed else "⚠️  缺失"
            self.log(f"  {status} - {check_name}")
        
        all_passed = all(checks.values())
        if all_passed:
            self.log("\n✅ 所有输出文件已生成！")
        else:
            self.log("\n⚠️  部分输出文件缺失，请检查实验是否正确完成")
        
        return all_passed
    
    def _check_checkpoints(self):
        """检查模型权重文件"""
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            return False
        
        ss_checkpoints = list(checkpoint_dir.glob("best_*_ss.pth"))
        expected = 6  # 3 models * 2 datasets
        found = len(ss_checkpoints)
        
        self.log(f"    模型权重：找到 {found}/{expected} 个文件")
        return found > 0
    
    def _check_comparison_results(self):
        """检查对比实验结果"""
        results_dir = Path("results")
        if not results_dir.exists():
            return False
        
        comparison_files = list(results_dir.glob("*_comparison_no_ref.json"))
        expected = len(self.eval_datasets)
        found = len(comparison_files)
        
        self.log(f"    对比结果：找到 {found}/{expected} 个文件")
        return found > 0
    
    def _check_ablation_results(self):
        """检查消融实验结果"""
        results_dir = Path("results")
        if not results_dir.exists():
            return False
        
        ablation_files = list(results_dir.glob("ablation_*.json"))
        expected = len(self.ablation_datasets)
        found = len(ablation_files)
        
        self.log(f"    消融结果：找到 {found}/{expected} 个文件")
        return found > 0
    
    def _check_figures(self):
        """检查可视化图表"""
        figures_dir = Path("figures")
        if not figures_dir.exists():
            return False
        
        png_files = list(figures_dir.glob("*.png"))
        found = len(png_files)
        
        self.log(f"    可视化图表：找到 {found} 个 PNG 文件")
        return found > 0
    
    def generate_report(self):
        """生成最终报告"""
        
        self.log("\n\n" + "="*80)
        self.log("📄 实验完成报告")
        self.log("="*80)
        
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.log(f"\n⏱️  总耗时：{int(hours)}h {int(minutes)}m {int(seconds)}s")
        self.log(f"\n📊 生成的表格文件：")
        
        # 检查是否生成了论文表格
        if Path("paper_comparison_tables.md").exists():
            self.log("    ✅ paper_comparison_tables.md")
        if Path("paper_ablation_tables.md").exists():
            self.log("    ✅ paper_ablation_tables.md")
        
        self.log(f"\n📝 下一步建议：")
        self.log(f"    1. 运行：python paper_results_extractor.py --mode all")
        self.log(f"    2. 查看生成的论文表格（paper_*_tables.md）")
        self.log(f"    3. 将表格复制到 paper_v2.md 中的对应位置")
        self.log(f"    4. 使用 PAPER_PROJECT_INTEGRATION.md 中的指南填充论文")
        
        self.log(f"\n📚 完整指南请参考：PAPER_PROJECT_INTEGRATION.md")
    
    def run(self, stages=['train', 'eval', 'ablation']):
        """执行指定阶段"""
        
        self.log(f"🎯 实验开始时间：{self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"📋 执行阶段：{', '.join(stages)}")
        
        stage_functions = {
            'train': self.stage_train,
            'eval': self.stage_eval,
            'ablation': self.stage_ablation
        }
        
        results = {}
        for stage in stages:
            if stage in stage_functions:
                results[stage] = stage_functions[stage]()
            else:
                self.log(f"❌ 未知阶段：{stage}")
        
        # 验证输出
        self.verify_outputs()
        
        # 生成报告
        self.generate_report()
        
        # 生成论文图表、数据与汇总文件
        if results.get('eval', False) or results.get('ablation', False):
            self.log("\n🔄 生成论文图表与数据...")
            self._run_paper_pipeline()
            self._run_paper_results_extractor()
            self._run_sweep("tv", "eq-36")
            self._run_sweep("patch", "eq-36")
            self._verify_required_artifacts()
        
        self.log_file.close()
        
        print(f"\n📋 完整日志已保存到：{self.output_log}")
        return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description='完整实验执行管理器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例：
  python run_all_experiments.py --stage all          # 执行所有阶段
  python run_all_experiments.py --stage train        # 仅训练
  python run_all_experiments.py --stage eval         # 仅对比实验
  python run_all_experiments.py --stage ablation     # 仅消融实验
  python run_all_experiments.py --stage train eval   # 训练和对比
        '''
    )
    
    parser.add_argument(
        '--stage',
        nargs='+',
        choices=['train', 'eval', 'ablation', 'all'],
        default=['all'],
        help='执行的实验阶段'
    )
    
    parser.add_argument(
        '--log',
        default='experiment_log.txt',
        help='日志文件路径'
    )
    
    args = parser.parse_args()
    
    # 处理 'all' 选项
    if 'all' in args.stage:
        stages = ['train', 'eval', 'ablation']
    else:
        stages = args.stage
    
    manager = ExperimentManager(output_log=args.log)
    success = manager.run(stages=stages)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
