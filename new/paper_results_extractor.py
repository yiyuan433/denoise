#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动从项目实验结果生成论文表格和摘要

使用方法：
    python paper_results_extractor.py --mode all
    python paper_results_extractor.py --mode comparison
    python paper_results_extractor.py --mode ablation
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

class PaperResultsExtractor:
    
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        self.datasets = ["eq-36", "eq-68"]
        self.ablation_datasets = ["eq-36"]
    
    def extract_comparison_results(self, save_md=True) -> Dict:
        """提取所有对比实验结果"""
        
        print("\n" + "="*80)
        print("对比实验结果提取")
        print("="*80)
        
        all_results = {}
        
        for dataset in self.datasets:
            comparison_file = Path(self.results_dir) / f"{dataset}_comparison_no_ref.json"
            
            if not comparison_file.exists():
                print(f"⚠️  {comparison_file} 不存在，跳过")
                continue
            
            print(f"\n📊 读取 {dataset} 对比结果...")
            
            with open(comparison_file) as f:
                comparison = json.load(f)
            
            all_results[dataset] = comparison
            
            # 打印顶部方法
            print(f"\n  排名 | 方法 | no_ref_score | 残差能量比 | 信号相关性 | 平滑度增益")
            print("  " + "-"*70)
            
            for i, item in enumerate(comparison[:10], 1):
                method = item.get('method', 'Unknown')[:20]
                score = item.get('no_ref_score', 0)
                residual = item.get('residual_energy_ratio', 0)
                corr = item.get('signal_corr_with_raw', 0)
                smooth = item.get('smoothness_gain', 0)
                
                marker = "🥇" if i == 1 else f"  {i}"
                print(f"  {marker:3} | {method:<20} | {score:>12.4f} | {residual:>10.4f} | {corr:>10.4f} | {smooth:>10.4f}")
        
        # 生成 Markdown 表格文件
        if save_md:
            self._save_comparison_tables(all_results)
        
        return all_results
    
    def extract_ablation_results(self, save_md=True) -> Dict:
        """提取所有消融实验结果"""
        
        print("\n" + "="*80)
        print("消融实验结果提取")
        print("="*80)
        
        all_results = {}
        
        for dataset in self.ablation_datasets:
            ablation_file = Path(self.results_dir) / f"ablation_{dataset}.json"
            
            if not ablation_file.exists():
                print(f"⚠️  {ablation_file} 不存在，跳过")
                continue
            
            print(f"\n📊 读取 {dataset} 消融结果...")
            
            with open(ablation_file) as f:
                ablations = json.load(f)
            
            all_results[dataset] = ablations
            
            # 打印消融配置结果
            print(f"\n  配置 | SNR (dB) | PSNR (dB) | SSIM | 相关性")
            print("  " + "-"*50)
            
            full_snr = None
            for item in ablations:
                metrics = item.get('metrics', {})
                ablation_name = item.get('ablation', 'Unknown')[:20]
                snr = metrics.get('snr', 0)
                psnr = metrics.get('psnr', 0)
                ssim = metrics.get('ssim', 0)
                corr = metrics.get('correlation', 0)
                
                if ablation_name == "full":
                    full_snr = snr
                    marker = "✓"
                else:
                    if full_snr:
                        delta = snr - full_snr
                        marker = f"(-{abs(delta):.2f})"
                    else:
                        marker = " "
                
                print(f"  {ablation_name:<20} | {snr:>8.2f} | {psnr:>9.2f} | {ssim:>6.4f} | {corr:>8.4f} {marker}")
        
        # 生成 Markdown 表格文件
        if save_md:
            self._save_ablation_tables(all_results)
        
        return all_results
    
    def _save_comparison_tables(self, results: Dict):
        """保存对比实验 Markdown 表格"""
        
        output_file = "paper_comparison_tables.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 论文对比实验表格\n\n")
            f.write("*自动生成，用于直接复制到论文中*\n\n")
            
            for dataset in self.ablation_datasets:
                if dataset not in results:
                    continue
                
                comparison = results[dataset]
                
                f.write(f"\n## 表：{dataset.upper()} 数据集对比结果\n\n")
                f.write("| 排名 | 方法 | no_ref_score | 残差能量比 | 信号相关性 | 平滑度增益 |\n")
                f.write("|------|------|--------------|----------|----------|----------|\n")
                
                for i, item in enumerate(comparison[:15], 1):
                    method = item.get('method', 'Unknown')
                    score = item.get('no_ref_score', 0)
                    residual = item.get('residual_energy_ratio', 0)
                    corr = item.get('signal_corr_with_raw', 0)
                    smooth = item.get('smoothness_gain', 0)
                    
                    # 标记最优方法
                    if i == 1:
                        method = f"**{method}**"
                        score = f"**{score:.4f}**"
                    
                    f.write(f"| {i} | {method} | {score} | {residual:.4f} | {corr:.4f} | {smooth:.4f} |\n")
        
        print(f"\n✅ 对比实验表格已保存到 {output_file}")
    
    def _save_ablation_tables(self, results: Dict):
        """保存消融实验 Markdown 表格"""
        
        output_file = "paper_ablation_tables.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 论文消融实验表格\n\n")
            f.write("*自动生成，用于直接复制到论文中*\n\n")
            
            for dataset in self.datasets:
                if dataset not in results:
                    continue
                
                ablations = results[dataset]
                
                f.write(f"\n## 表：{dataset.upper()} 消融实验结果\n\n")
                f.write("| 消融配置 | SNR (dB) | PSNR (dB) | SSIM | 相关性 | 性能降幅 |\n")
                f.write("|---------|---------|----------|------|--------|----------|\n")
                
                full_snr = None
                full_psnr = None
                
                for item in ablations:
                    metrics = item.get('metrics', {})
                    ablation_name = item.get('ablation', 'Unknown')
                    snr = metrics.get('snr', 0)
                    psnr = metrics.get('psnr', 0)
                    ssim = metrics.get('ssim', 0)
                    corr = metrics.get('correlation', 0)
                    
                    if ablation_name == "full":
                        full_snr = snr
                        full_psnr = psnr
                        delta_str = "-"
                        ablation_name = f"**{ablation_name}**"
                        snr_str = f"**{snr:.2f}**"
                        psnr_str = f"**{psnr:.2f}**"
                    else:
                        if full_snr:
                            delta = snr - full_snr
                            delta_str = f"-{abs(delta):.2f}" if delta < 0 else f"+{delta:.2f}"
                        else:
                            delta_str = "N/A"
                        snr_str = f"{snr:.2f}"
                        psnr_str = f"{psnr:.2f}"
                    
                    f.write(f"| {ablation_name} | {snr_str} | {psnr_str} | {ssim:.4f} | {corr:.4f} | {delta_str} |\n")
        
        print(f"✅ 消融实验表格已保存到 {output_file}")
    
    def generate_summary(self) -> str:
        """生成论文摘要数据总结"""
        
        print("\n" + "="*80)
        print("实验数据总结")
        print("="*80)
        
        summary = []
        
        # 读取对比结果
        comparison_results = {}
        for dataset in self.datasets:
            comparison_file = Path(self.results_dir) / f"{dataset}_comparison_no_ref.json"
            if comparison_file.exists():
                with open(comparison_file) as f:
                    comparison_results[dataset] = json.load(f)
        
        # 读取消融结果
        ablation_results = {}
        for dataset in self.ablation_datasets:
            ablation_file = Path(self.results_dir) / f"ablation_{dataset}.json"
            if ablation_file.exists():
                with open(ablation_file) as f:
                    ablation_results[dataset] = json.load(f)
        
        # 计算总结数据
        print("\n📈 统计数据：")
        
        if comparison_results:
            print(f"\n  ✓ 对比实验：{len(comparison_results)} 个数据集")
            all_methods = set()
            best_scores = []
            
            for dataset, comparison in comparison_results.items():
                num_methods = len(comparison)
                best_score = comparison[0].get('no_ref_score', 0) if comparison else 0
                
                all_methods.update([item.get('method') for item in comparison])
                best_scores.append(best_score)
                
                print(f"    - {dataset}: {num_methods} 种方法，最佳方法得分 {best_score:.4f}")
            
            summary.append(f"对比了 {len(all_methods)} 种去噪方法")
            summary.append(f"在 {len(comparison_results)} 组实测数据上进行验证")
            summary.append(f"综合评分范围：{min(best_scores):.4f}-{max(best_scores):.4f}")
        
        if ablation_results:
            print(f"\n  ✓ 消融实验：{len(ablation_results)} 个数据集")
            
            for dataset, ablations in ablation_results.items():
                num_configs = len(ablations)
                print(f"    - {dataset}: {num_configs} 个消融配置")
            
            summary.append(f"进行了 {num_configs} 项消融实验验证模块贡献")
        
        print("\n📝 建议在论文摘要中包含：")
        for s in summary:
            print(f"    • {s}")
        
        return " ".join(summary)
    
    def run_all(self):
        """执行所有提取操作"""
        
        print("\n" + "="*80)
        print("🚀 论文结果自动提取工具")
        print("="*80)
        
        # 检查结果目录
        if not Path(self.results_dir).exists():
            print(f"❌ 错误：{self.results_dir} 目录不存在")
            print("   请先运行实验脚本生成结果")
            return False
        
        # 提取对比结果
        comparison_results = self.extract_comparison_results()
        
        # 提取消融结果
        ablation_results = self.extract_ablation_results()
        
        # 生成总结
        summary = self.generate_summary()
        
        print("\n" + "="*80)
        print("✅ 完成！")
        print("="*80)
        print(f"\n生成的文件：")
        print(f"  • paper_comparison_tables.md - 对比实验表格")
        print(f"  • paper_ablation_tables.md - 消融实验表格")
        print(f"\n这些文件可直接复制到论文中使用。")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='从项目实验结果自动生成论文表格',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例：
  python paper_results_extractor.py --mode all
  python paper_results_extractor.py --mode comparison
  python paper_results_extractor.py --mode ablation
        '''
    )
    
    parser.add_argument(
        '--mode',
        choices=['all', 'comparison', 'ablation', 'summary'],
        default='all',
        help='提取模式'
    )
    
    parser.add_argument(
        '--results-dir',
        default='results',
        help='实验结果目录'
    )
    
    args = parser.parse_args()
    
    extractor = PaperResultsExtractor(args.results_dir)
    
    if args.mode == 'all':
        extractor.run_all()
    elif args.mode == 'comparison':
        extractor.extract_comparison_results()
    elif args.mode == 'ablation':
        extractor.extract_ablation_results()
    elif args.mode == 'summary':
        extractor.generate_summary()


if __name__ == "__main__":
    main()
