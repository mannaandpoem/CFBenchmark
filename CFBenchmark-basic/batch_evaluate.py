#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量CSV文件评估脚本
用于批量评估CFBenchmark保存的CSV文件，生成汇总报告
"""

import os
import pandas as pd
import numpy as np
import argparse
import json
from typing import Dict, List, Any, Optional
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 导入单个文件评估器
from evaluate_csv import CSVEvaluator

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BatchEvaluator:
    """批量CSV文件评估器"""
    
    def __init__(self, input_dir: str, output_dir: str = None):
        """
        初始化批量评估器
        
        Args:
            input_dir: 包含CSV文件的目录
            output_dir: 输出目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir or os.path.join(input_dir, "batch_evaluation_results")
        self.results = {}
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
    def find_csv_files(self) -> List[str]:
        """查找所有CSV文件"""
        csv_files = []
        
        # 查找所有output.csv文件
        patterns = [
            os.path.join(self.input_dir, "**/*-output.csv"),
            os.path.join(self.input_dir, "**/*output.csv"),
            os.path.join(self.input_dir, "*.csv")
        ]
        
        for pattern in patterns:
            csv_files.extend(glob.glob(pattern, recursive=True))
        
        # 去重并排序
        csv_files = sorted(list(set(csv_files)))
        
        print(f"找到 {len(csv_files)} 个CSV文件")
        for csv_file in csv_files:
            print(f"  - {csv_file}")
        
        return csv_files
    
    def evaluate_all_files(self, csv_files: List[str]) -> Dict[str, Any]:
        """评估所有CSV文件"""
        all_results = {}
        
        for csv_file in csv_files:
            print(f"\n正在评估: {csv_file}")
            try:
                # 创建评估器
                evaluator = CSVEvaluator(csv_file)
                
                # 执行评估
                result = evaluator.evaluate()
                
                # 保存结果
                file_key = self._get_file_key(csv_file)
                all_results[file_key] = {
                    'file_path': csv_file,
                    'result': result
                }
                
                print(f"✓ 评估完成: {file_key}")
                
            except Exception as e:
                print(f"✗ 评估失败: {csv_file} - {str(e)}")
                continue
        
        return all_results
    
    def _get_file_key(self, csv_file: str) -> str:
        """生成文件键名"""
        # 提取相对路径和文件名信息
        rel_path = os.path.relpath(csv_file, self.input_dir)
        
        # 提取模型名和任务类型
        parts = rel_path.split(os.sep)
        
        if len(parts) >= 3:
            # 格式: test_type/model_name/task-output.csv
            test_type = parts[-3] if parts[-3] in ['few-shot', 'zero-shot'] else 'unknown'
            model_name = parts[-2]
            task_name = os.path.splitext(parts[-1])[0].replace('-output', '')
            return f"{test_type}_{model_name}_{task_name}"
        else:
            # 简单格式
            return os.path.splitext(os.path.basename(csv_file))[0]
    
    def generate_summary_report(self, all_results: Dict[str, Any]) -> str:
        """生成汇总报告"""
        report_path = os.path.join(self.output_dir, "batch_evaluation_summary.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CFBenchmark 批量评估汇总报告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入目录: {self.input_dir}\n")
            f.write(f"输出目录: {self.output_dir}\n")
            f.write(f"评估文件数: {len(all_results)}\n\n")
            
            # 按任务类型分组
            task_groups = self._group_by_task_type(all_results)
            
            for task_type, files in task_groups.items():
                f.write(f"\n{task_type.upper()} 任务评估结果:\n")
                f.write("-" * 60 + "\n")
                
                if task_type in ['summary', 'risk', 'suggestion']:
                    # 生成任务
                    self._write_generation_summary(f, files)
                else:
                    # 分类任务
                    self._write_classification_summary(f, files)
            
            # 整体统计
            f.write(f"\n\n整体统计:\n")
            f.write("-" * 30 + "\n")
            self._write_overall_statistics(f, all_results)
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("报告生成完成\n")
        
        print(f"✓ 汇总报告已保存到: {report_path}")
        return report_path
    
    def _group_by_task_type(self, all_results: Dict[str, Any]) -> Dict[str, List]:
        """按任务类型分组"""
        task_groups = {}
        
        for file_key, data in all_results.items():
            task_type = data['result']['classification_type']
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append((file_key, data))
        
        return task_groups
    
    def _write_classification_summary(self, f, files: List):
        """写入分类任务汇总"""
        f.write(f"{'文件名':<40} {'F1分数':<10} {'精确率':<10} {'召回率':<10} {'样本数':<8}\n")
        f.write("-" * 80 + "\n")
        
        total_f1 = []
        total_precision = []
        total_recall = []
        total_samples = 0
        
        for file_key, data in files:
            result = data['result']
            metrics = result['metrics']
            
            f1 = metrics['average_f1']
            precision = metrics['average_precision']
            recall = metrics['average_recall']
            samples = result['total_samples']
            
            f.write(f"{file_key:<40} {f1:<10.4f} {precision:<10.4f} {recall:<10.4f} {samples:<8d}\n")
            
            total_f1.append(f1)
            total_precision.append(precision)
            total_recall.append(recall)
            total_samples += samples
        
        # 计算平均值
        if total_f1:
            avg_f1 = np.mean(total_f1)
            avg_precision = np.mean(total_precision)
            avg_recall = np.mean(total_recall)
            
            f.write("-" * 80 + "\n")
            f.write(f"{'平均值':<40} {avg_f1:<10.4f} {avg_precision:<10.4f} {avg_recall:<10.4f} {total_samples:<8d}\n")
    
    def _write_generation_summary(self, f, files: List):
        """写入生成任务汇总"""
        f.write(f"{'文件名':<40} {'余弦相似度':<12} {'回答长度':<10} {'输出长度':<10} {'样本数':<8}\n")
        f.write("-" * 80 + "\n")
        
        total_cosine = []
        total_response_len = []
        total_output_len = []
        total_samples = 0
        
        for file_key, data in files:
            result = data['result']
            metrics = result['metrics']
            
            cosine = metrics['average_cosine_similarity']
            response_len = metrics['average_response_length']
            output_len = metrics['average_output_length']
            samples = result['total_samples']
            
            f.write(f"{file_key:<40} {cosine:<12.4f} {response_len:<10.1f} {output_len:<10.1f} {samples:<8d}\n")
            
            total_cosine.append(cosine)
            total_response_len.append(response_len)
            total_output_len.append(output_len)
            total_samples += samples
        
        # 计算平均值
        if total_cosine:
            avg_cosine = np.mean(total_cosine)
            avg_response_len = np.mean(total_response_len)
            avg_output_len = np.mean(total_output_len)
            
            f.write("-" * 80 + "\n")
            f.write(f"{'平均值':<40} {avg_cosine:<12.4f} {avg_response_len:<10.1f} {avg_output_len:<10.1f} {total_samples:<8d}\n")
    
    def _write_overall_statistics(self, f, all_results: Dict[str, Any]):
        """写入整体统计"""
        # 统计任务类型分布
        task_counts = {}
        model_counts = {}
        test_type_counts = {}
        
        for file_key, data in all_results.items():
            # 任务类型
            task_type = data['result']['classification_type']
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
            
            # 模型类型（从文件键提取）
            parts = file_key.split('_')
            if len(parts) >= 2:
                model = parts[1]
                model_counts[model] = model_counts.get(model, 0) + 1
            
            # 测试类型
            if len(parts) >= 1:
                test_type = parts[0]
                test_type_counts[test_type] = test_type_counts.get(test_type, 0) + 1
        
        f.write(f"任务类型分布:\n")
        for task, count in sorted(task_counts.items()):
            f.write(f"  {task}: {count} 个文件\n")
        
        f.write(f"\n模型分布:\n")
        for model, count in sorted(model_counts.items()):
            f.write(f"  {model}: {count} 个文件\n")
        
        f.write(f"\n测试类型分布:\n")
        for test_type, count in sorted(test_type_counts.items()):
            f.write(f"  {test_type}: {count} 个文件\n")
    
    def create_comparison_visualizations(self, all_results: Dict[str, Any]):
        """创建对比可视化图表"""
        # 按任务类型分组
        task_groups = self._group_by_task_type(all_results)
        
        for task_type, files in task_groups.items():
            if len(files) < 2:
                continue  # 至少需要2个文件才能对比
            
            if task_type in ['summary', 'risk', 'suggestion']:
                self._create_generation_comparison(task_type, files)
            else:
                self._create_classification_comparison(task_type, files)
    
    def _create_classification_comparison(self, task_type: str, files: List):
        """创建分类任务对比图表"""
        file_names = []
        f1_scores = []
        precisions = []
        recalls = []
        
        for file_key, data in files:
            result = data['result']
            metrics = result['metrics']
            
            file_names.append(file_key.replace(f"{task_type}_", "").replace("_", "\n"))
            f1_scores.append(metrics['average_f1'])
            precisions.append(metrics['average_precision'])
            recalls.append(metrics['average_recall'])
        
        # 创建对比图
        x = np.arange(len(file_names))
        width = 0.25
        
        plt.figure(figsize=(max(12, len(file_names) * 2), 8))
        plt.bar(x - width, f1_scores, width, label='F1分数', alpha=0.8)
        plt.bar(x, precisions, width, label='精确率', alpha=0.8)
        plt.bar(x + width, recalls, width, label='召回率', alpha=0.8)
        
        plt.xlabel('模型/配置')
        plt.ylabel('分数')
        plt.title(f'{task_type.title()} 任务 - 模型性能对比')
        plt.xticks(x, file_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, f"{task_type}_classification_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ {task_type} 分类对比图已保存到: {plot_path}")
    
    def _create_generation_comparison(self, task_type: str, files: List):
        """创建生成任务对比图表"""
        file_names = []
        cosine_similarities = []
        
        for file_key, data in files:
            result = data['result']
            metrics = result['metrics']
            
            file_names.append(file_key.replace(f"{task_type}_", "").replace("_", "\n"))
            cosine_similarities.append(metrics['average_cosine_similarity'])
        
        # 创建对比图
        plt.figure(figsize=(max(10, len(file_names) * 1.5), 6))
        bars = plt.bar(file_names, cosine_similarities, alpha=0.8, color='lightgreen')
        
        # 添加数值标签
        for bar, score in zip(bars, cosine_similarities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.xlabel('模型/配置')
        plt.ylabel('余弦相似度')
        plt.title(f'{task_type.title()} 任务 - 模型性能对比')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, f"{task_type}_generation_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ {task_type} 生成对比图已保存到: {plot_path}")
    
    def save_detailed_results(self, all_results: Dict[str, Any]) -> str:
        """保存详细结果到JSON文件"""
        json_path = os.path.join(self.output_dir, "batch_detailed_results.json")
        
        # 转换numpy类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        results_converted = recursive_convert(all_results)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 详细结果已保存到: {json_path}")
        return json_path
    
    def run_batch_evaluation(self):
        """运行批量评估"""
        print(f"开始批量评估...")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        
        # 查找CSV文件
        csv_files = self.find_csv_files()
        
        if not csv_files:
            print("未找到CSV文件")
            return
        
        # 评估所有文件
        all_results = self.evaluate_all_files(csv_files)
        
        if not all_results:
            print("没有成功评估的文件")
            return
        
        # 生成汇总报告
        report_path = self.generate_summary_report(all_results)
        
        # 保存详细结果
        json_path = self.save_detailed_results(all_results)
        
        # 创建对比可视化
        self.create_comparison_visualizations(all_results)
        
        print(f"\n批量评估完成!")
        print(f"成功评估 {len(all_results)} 个文件")
        print(f"结果保存在: {self.output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CFBenchmark CSV文件批量评估工具')
    parser.add_argument('input_dir', help='包含CSV文件的目录')
    parser.add_argument('--output_dir', '-o', help='输出目录（默认为input_dir/batch_evaluation_results）')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在 - {args.input_dir}")
        return
    
    try:
        # 创建批量评估器
        batch_evaluator = BatchEvaluator(args.input_dir, args.output_dir)
        
        # 运行批量评估
        batch_evaluator.run_batch_evaluation()
        
    except Exception as e:
        print(f"批量评估过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 