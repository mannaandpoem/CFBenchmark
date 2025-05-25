#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV文件评估脚本
用于评估CFBenchmark保存的CSV文件，计算F1分数、余弦相似度等指标
"""

import os
import pandas as pd
import numpy as np
import argparse
import json
from typing import Dict, List, Any, Optional, Tuple
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CSVEvaluator:
    """CSV文件评估器"""
    
    def __init__(self, csv_path: str, classification_type: str = None):
        """
        初始化评估器
        
        Args:
            csv_path: CSV文件路径
            classification_type: 分类类型 (company, product, sector, event, sentiment, summary, risk, suggestion)
        """
        self.csv_path = csv_path
        self.classification_type = classification_type or self._infer_classification_type()
        self.df = pd.read_csv(csv_path)
        
        # 验证CSV文件格式
        self._validate_csv_format()
        
        # 预处理数据
        self._preprocess_data()
        
    def _infer_classification_type(self) -> str:
        """从文件名推断分类类型"""
        filename = os.path.basename(self.csv_path).lower()
        
        classification_types = [
            'company', 'product', 'sector', 'event',
            'sentiment', 'summary', 'risk', 'suggestion'
        ]
        
        for cls_type in classification_types:
            if cls_type in filename:
                return cls_type
        
        return 'unknown'
    
    def _validate_csv_format(self):
        """验证CSV文件格式"""
        required_columns = ['input', 'response', 'output']
        
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"CSV文件缺少必需的列: {col}")
        
        print(f"✓ CSV文件格式验证通过，共有 {len(self.df)} 行数据")
    
    def _preprocess_data(self):
        """预处理数据"""
        # 处理空值
        self.df['response'] = self.df['response'].fillna('')
        self.df['output'] = self.df['output'].fillna('')
        
        # 转换为字符串
        self.df['response'] = self.df['response'].astype(str)
        self.df['output'] = self.df['output'].astype(str)
        
        print(f"✓ 数据预处理完成")
    
    def extract_labels(self, text: str) -> List[str]:
        """从文本中提取标签"""
        if pd.isna(text) or text == '' or text == 'nan':
            return []
        
        # 处理中文逗号和英文逗号
        text = str(text).replace('，', ',')
        
        # 分割并清理标签
        labels = [label.strip() for label in text.split(',') if label.strip()]
        
        return labels
    
    def calculate_f1_score(self, response_labels: List[str], output_labels: List[str]) -> Dict[str, float]:
        """计算F1分数"""
        response_set = set(response_labels)
        output_set = set(output_labels)
        
        # 计算交集
        intersection = response_set.intersection(output_set)
        
        # 计算精确率、召回率和F1分数
        precision = len(intersection) / len(output_set) if len(output_set) > 0 else 0.0
        recall = len(intersection) / len(response_set) if len(response_set) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': len(intersection),
            'predicted_positives': len(output_set),
            'actual_positives': len(response_set)
        }
    
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """计算余弦相似度"""
        def preprocess_text(text):
            """文本预处理"""
            text = text.lower()
            # 替换中文标点
            text = text.replace('，', ' ').replace('。', ' ').replace('！', ' ').replace('？', ' ')
            text = text.replace('、', ' ').replace('；', ' ').replace('：', ' ').replace('"', ' ')
            text = text.replace('"', ' ').replace(''', ' ').replace(''', ' ').replace('（', ' ').replace('）', ' ')
            # 替换英文标点
            text = text.replace(',', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ')
            text = text.replace(';', ' ').replace(':', ' ').replace('"', ' ').replace("'", ' ')
            text = text.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ')
            # 移除多余空格
            text = ' '.join(text.split())
            return text
        
        def get_tokens(text):
            """获取词汇tokens"""
            words = text.split()
            tokens = []
            for word in words:
                # 如果包含中文字符，按字符分割
                if any('\u4e00' <= char <= '\u9fff' for char in word):
                    tokens.extend(list(word))
                else:
                    tokens.append(word)
            return tokens
        
        # 预处理文本
        text1 = preprocess_text(str(text1))
        text2 = preprocess_text(str(text2))
        
        # 处理空文本
        if not text1.strip() and not text2.strip():
            return 1.0
        if not text1.strip() or not text2.strip():
            return 0.0
        
        # 获取tokens
        tokens1 = get_tokens(text1)
        tokens2 = get_tokens(text2)
        
        # 创建词汇表
        vocab = list(set(tokens1 + tokens2))
        if not vocab:
            return 0.0
        
        # 计算TF向量
        def calculate_tf_vector(tokens, vocab):
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            
            vector = []
            for token in vocab:
                vector.append(tf.get(token, 0))
            
            return np.array(vector, dtype=float)
        
        vector1 = calculate_tf_vector(tokens1, vocab)
        vector2 = calculate_tf_vector(tokens2, vocab)
        
        # 计算余弦相似度
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, cosine_sim))
    
    def evaluate_classification_tasks(self) -> Dict[str, Any]:
        """评估分类任务（company, product, sector, event, sentiment）"""
        results = []
        all_labels = set()
        
        for idx, row in self.df.iterrows():
            response_labels = self.extract_labels(row['response'])
            output_labels = self.extract_labels(row['output'])
            
            # 收集所有标签
            all_labels.update(response_labels)
            all_labels.update(output_labels)
            
            # 计算F1分数
            f1_metrics = self.calculate_f1_score(response_labels, output_labels)
            
            results.append({
                'index': idx,
                'response_labels': response_labels,
                'output_labels': output_labels,
                **f1_metrics
            })
        
        # 计算总体指标
        total_f1 = np.mean([r['f1'] for r in results])
        total_precision = np.mean([r['precision'] for r in results])
        total_recall = np.mean([r['recall'] for r in results])
        
        # 计算标签级别的统计
        label_stats = self._calculate_label_statistics(results, all_labels)
        
        return {
            'task_type': 'classification',
            'classification_type': self.classification_type,
            'total_samples': len(results),
            'metrics': {
                'average_f1': total_f1,
                'average_precision': total_precision,
                'average_recall': total_recall
            },
            'label_statistics': label_stats,
            'detailed_results': results
        }
    
    def evaluate_generation_tasks(self) -> Dict[str, Any]:
        """评估生成任务（summary, risk, suggestion）"""
        results = []
        
        for idx, row in self.df.iterrows():
            response_text = str(row['response'])
            output_text = str(row['output'])
            
            # 计算余弦相似度
            cosine_sim = self.calculate_cosine_similarity(response_text, output_text)
            
            results.append({
                'index': idx,
                'response_text': response_text,
                'output_text': output_text,
                'cosine_similarity': cosine_sim,
                'response_length': len(response_text),
                'output_length': len(output_text)
            })
        
        # 计算总体指标
        avg_cosine_similarity = np.mean([r['cosine_similarity'] for r in results])
        avg_response_length = np.mean([r['response_length'] for r in results])
        avg_output_length = np.mean([r['output_length'] for r in results])
        
        return {
            'task_type': 'generation',
            'classification_type': self.classification_type,
            'total_samples': len(results),
            'metrics': {
                'average_cosine_similarity': avg_cosine_similarity,
                'average_response_length': avg_response_length,
                'average_output_length': avg_output_length
            },
            'detailed_results': results
        }
    
    def _calculate_label_statistics(self, results: List[Dict], all_labels: set) -> Dict[str, Any]:
        """计算标签级别的统计信息"""
        label_counts = defaultdict(int)
        label_correct = defaultdict(int)
        label_predicted = defaultdict(int)
        label_actual = defaultdict(int)
        
        for result in results:
            response_labels = set(result['response_labels'])
            output_labels = set(result['output_labels'])
            
            # 统计每个标签的出现次数
            for label in response_labels:
                label_actual[label] += 1
            
            for label in output_labels:
                label_predicted[label] += 1
            
            # 统计正确预测的标签
            for label in response_labels.intersection(output_labels):
                label_correct[label] += 1
        
        # 计算每个标签的精确率、召回率和F1分数
        label_metrics = {}
        for label in all_labels:
            precision = label_correct[label] / label_predicted[label] if label_predicted[label] > 0 else 0.0
            recall = label_correct[label] / label_actual[label] if label_actual[label] > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            label_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'actual_count': label_actual[label],
                'predicted_count': label_predicted[label],
                'correct_count': label_correct[label]
            }
        
        return {
            'total_unique_labels': len(all_labels),
            'label_metrics': label_metrics
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """执行评估"""
        print(f"开始评估 {self.classification_type} 任务...")
        
        # 根据任务类型选择评估方法
        if self.classification_type in ['summary', 'risk', 'suggestion']:
            results = self.evaluate_generation_tasks()
        else:
            results = self.evaluate_classification_tasks()
        
        print(f"✓ 评估完成")
        return results
    
    def generate_report(self, results: Dict[str, Any], output_dir: str = None) -> str:
        """生成评估报告"""
        if output_dir is None:
            output_dir = os.path.dirname(self.csv_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成报告文件名
        base_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        report_path = os.path.join(output_dir, f"{base_name}_evaluation_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"CFBenchmark CSV评估报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"文件路径: {self.csv_path}\n")
            f.write(f"任务类型: {results['task_type']}\n")
            f.write(f"分类类型: {results['classification_type']}\n")
            f.write(f"样本数量: {results['total_samples']}\n\n")
            
            f.write("主要指标:\n")
            f.write("-" * 30 + "\n")
            
            if results['task_type'] == 'classification':
                metrics = results['metrics']
                f.write(f"平均F1分数: {metrics['average_f1']:.4f}\n")
                f.write(f"平均精确率: {metrics['average_precision']:.4f}\n")
                f.write(f"平均召回率: {metrics['average_recall']:.4f}\n\n")
                
                # 标签统计
                label_stats = results['label_statistics']
                f.write(f"标签统计:\n")
                f.write(f"唯一标签数量: {label_stats['total_unique_labels']}\n\n")
                
                # 前10个表现最好的标签
                label_metrics = label_stats['label_metrics']
                sorted_labels = sorted(label_metrics.items(), 
                                     key=lambda x: x[1]['f1'], reverse=True)
                
                f.write("标签性能排行 (前10个):\n")
                f.write("-" * 50 + "\n")
                for i, (label, metrics) in enumerate(sorted_labels[:10]):
                    f.write(f"{i+1:2d}. {label:20s} F1: {metrics['f1']:.4f} "
                           f"P: {metrics['precision']:.4f} R: {metrics['recall']:.4f}\n")
                
            else:  # generation task
                metrics = results['metrics']
                f.write(f"平均余弦相似度: {metrics['average_cosine_similarity']:.4f}\n")
                f.write(f"平均回答长度: {metrics['average_response_length']:.1f}\n")
                f.write(f"平均输出长度: {metrics['average_output_length']:.1f}\n\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("报告生成完成\n")
        
        print(f"✓ 评估报告已保存到: {report_path}")
        return report_path
    
    def save_detailed_results(self, results: Dict[str, Any], output_dir: str = None) -> str:
        """保存详细结果到JSON文件"""
        if output_dir is None:
            output_dir = os.path.dirname(self.csv_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        json_path = os.path.join(output_dir, f"{base_name}_detailed_results.json")
        
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 递归转换所有numpy类型
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        results_converted = recursive_convert(results)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 详细结果已保存到: {json_path}")
        return json_path
    
    def create_visualizations(self, results: Dict[str, Any], output_dir: str = None):
        """创建可视化图表"""
        if output_dir is None:
            output_dir = os.path.dirname(self.csv_path)
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        
        if results['task_type'] == 'classification':
            self._create_classification_plots(results, output_dir, base_name)
        else:
            self._create_generation_plots(results, output_dir, base_name)
    
    def _create_classification_plots(self, results: Dict[str, Any], output_dir: str, base_name: str):
        """创建分类任务的可视化图表"""
        detailed_results = results['detailed_results']
        
        # 1. F1分数分布直方图
        f1_scores = [r['f1'] for r in detailed_results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(f1_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'{self.classification_type.title()} - F1分数分布')
        plt.xlabel('F1分数')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{base_name}_f1_distribution.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 标签性能图表
        if 'label_statistics' in results:
            label_metrics = results['label_statistics']['label_metrics']
            
            # 选择前15个最常见的标签
            sorted_labels = sorted(label_metrics.items(), 
                                 key=lambda x: x[1]['actual_count'], reverse=True)[:15]
            
            labels = [item[0] for item in sorted_labels]
            f1_scores = [item[1]['f1'] for item in sorted_labels]
            precisions = [item[1]['precision'] for item in sorted_labels]
            recalls = [item[1]['recall'] for item in sorted_labels]
            
            # 创建标签性能对比图
            x = np.arange(len(labels))
            width = 0.25
            
            plt.figure(figsize=(15, 8))
            plt.bar(x - width, f1_scores, width, label='F1分数', alpha=0.8)
            plt.bar(x, precisions, width, label='精确率', alpha=0.8)
            plt.bar(x + width, recalls, width, label='召回率', alpha=0.8)
            
            plt.xlabel('标签')
            plt.ylabel('分数')
            plt.title(f'{self.classification_type.title()} - 标签性能对比 (前15个)')
            plt.xticks(x, labels, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{base_name}_label_performance.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ 分类任务可视化图表已保存到: {output_dir}")
    
    def _create_generation_plots(self, results: Dict[str, Any], output_dir: str, base_name: str):
        """创建生成任务的可视化图表"""
        detailed_results = results['detailed_results']
        
        # 1. 余弦相似度分布
        cosine_similarities = [r['cosine_similarity'] for r in detailed_results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(cosine_similarities, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title(f'{self.classification_type.title()} - 余弦相似度分布')
        plt.xlabel('余弦相似度')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{base_name}_cosine_similarity_distribution.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 文本长度对比
        response_lengths = [r['response_length'] for r in detailed_results]
        output_lengths = [r['output_length'] for r in detailed_results]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(response_lengths, output_lengths, alpha=0.6, color='coral')
        plt.xlabel('标准答案长度')
        plt.ylabel('模型输出长度')
        plt.title('文本长度对比')
        plt.grid(True, alpha=0.3)
        
        # 添加对角线
        max_len = max(max(response_lengths), max(output_lengths))
        plt.plot([0, max_len], [0, max_len], 'r--', alpha=0.5)
        
        plt.subplot(1, 2, 2)
        plt.hist(response_lengths, bins=20, alpha=0.7, label='标准答案', color='skyblue')
        plt.hist(output_lengths, bins=20, alpha=0.7, label='模型输出', color='orange')
        plt.xlabel('文本长度')
        plt.ylabel('频次')
        plt.title('文本长度分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_text_length_analysis.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 生成任务可视化图表已保存到: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CFBenchmark CSV文件评估工具')
    parser.add_argument('csv_path', help='CSV文件路径')
    parser.add_argument('--classification_type', '-t', 
                       choices=['company', 'product', 'sector', 'event', 
                               'sentiment', 'summary', 'risk', 'suggestion'],
                       help='分类类型（可选，会自动从文件名推断）')
    parser.add_argument('--output_dir', '-o', help='输出目录（默认为CSV文件所在目录）')
    parser.add_argument('--no_plots', action='store_true', help='不生成可视化图表')
    parser.add_argument('--no_detailed', action='store_true', help='不保存详细结果JSON文件')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.csv_path):
        print(f"错误: 文件不存在 - {args.csv_path}")
        return
    
    try:
        # 创建评估器
        evaluator = CSVEvaluator(args.csv_path, args.classification_type)
        
        # 执行评估
        results = evaluator.evaluate()
        
        # 生成报告
        report_path = evaluator.generate_report(results, args.output_dir)
        
        # 保存详细结果
        if not args.no_detailed:
            json_path = evaluator.save_detailed_results(results, args.output_dir)
        
        # 创建可视化图表
        if not args.no_plots:
            evaluator.create_visualizations(results, args.output_dir)
        
        # 打印主要结果
        print("\n" + "=" * 50)
        print("评估结果摘要:")
        print("=" * 50)
        print(f"任务类型: {results['task_type']}")
        print(f"分类类型: {results['classification_type']}")
        print(f"样本数量: {results['total_samples']}")
        
        if results['task_type'] == 'classification':
            metrics = results['metrics']
            print(f"平均F1分数: {metrics['average_f1']:.4f}")
            print(f"平均精确率: {metrics['average_precision']:.4f}")
            print(f"平均召回率: {metrics['average_recall']:.4f}")
        else:
            metrics = results['metrics']
            print(f"平均余弦相似度: {metrics['average_cosine_similarity']:.4f}")
        
        print(f"\n详细报告已保存到: {report_path}")
        
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 