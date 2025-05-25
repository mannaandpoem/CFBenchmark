# CFBenchmark CSV评估工具使用说明

本文档介绍如何使用CFBenchmark的CSV评估工具来评估模型输出结果。

## 概述

CFBenchmark提供了两个主要的评估脚本：

1. **`evaluate_csv.py`** - 单个CSV文件评估工具
2. **`batch_evaluate.py`** - 批量CSV文件评估工具

这些工具可以自动计算各种评估指标，包括F1分数、精确率、召回率（用于分类任务）和余弦相似度（用于生成任务）。

## 安装依赖

在使用评估工具之前，请确保安装了必要的依赖：

```bash
pip install pandas numpy matplotlib seaborn
```

## 单个文件评估 (`evaluate_csv.py`)

### 基本用法

```bash
python evaluate_csv.py <csv_file_path>
```

### 参数说明

- `csv_path`: CSV文件路径（必需）
- `--classification_type, -t`: 分类类型，可选值：company, product, sector, event, sentiment, summary, risk, suggestion
- `--output_dir, -o`: 输出目录（默认为CSV文件所在目录）
- `--no_plots`: 不生成可视化图表
- `--no_detailed`: 不保存详细结果JSON文件

### 使用示例

```bash
# 基本评估
python evaluate_csv.py cfbenchmark-response/few-shot/openai/gpt-4o-mini/company-output.csv

# 指定分类类型和输出目录
python evaluate_csv.py company-output.csv -t company -o ./results

# 只生成报告，不生成图表和详细结果
python evaluate_csv.py company-output.csv --no_plots --no_detailed
```

### 输出文件

单个文件评估会生成以下文件：

1. **`{filename}_evaluation_report.txt`** - 文本格式的评估报告
2. **`{filename}_detailed_results.json`** - 详细结果的JSON文件
3. **可视化图表**（PNG格式）：
   - 分类任务：F1分数分布图、标签性能对比图
   - 生成任务：余弦相似度分布图、文本长度分析图

## 批量评估 (`batch_evaluate.py`)

### 基本用法

```bash
python batch_evaluate.py <input_directory>
```

### 参数说明

- `input_dir`: 包含CSV文件的目录（必需）
- `--output_dir, -o`: 输出目录（默认为input_dir/batch_evaluation_results）

### 使用示例

```bash
# 批量评估cfbenchmark-response目录下的所有CSV文件
python batch_evaluate.py cfbenchmark-response

# 指定输出目录
python batch_evaluate.py cfbenchmark-response -o ./batch_results
```

### 输出文件

批量评估会生成以下文件：

1. **`batch_evaluation_summary.txt`** - 批量评估汇总报告
2. **`batch_detailed_results.json`** - 所有文件的详细结果
3. **对比可视化图表**：
   - `{task_type}_classification_comparison.png` - 分类任务模型对比
   - `{task_type}_generation_comparison.png` - 生成任务模型对比

## CSV文件格式要求

评估工具要求CSV文件包含以下三列：

- **`input`**: 输入文本
- **`response`**: 标准答案/期望输出
- **`output`**: 模型实际输出

示例CSV格式：
```csv
input,response,output
这是一个示例输入,标准答案,模型输出
```

## 评估指标说明

### 分类任务指标

适用于：company, product, sector, event, sentiment

- **F1分数**: 精确率和召回率的调和平均数
- **精确率**: 正确预测的正例数 / 预测为正例的总数
- **召回率**: 正确预测的正例数 / 实际正例的总数

### 生成任务指标

适用于：summary, risk, suggestion

- **余弦相似度**: 基于TF-IDF向量的文本相似度
- **文本长度**: 标准答案和模型输出的字符长度

## 任务类型自动识别

评估工具会根据文件名自动识别任务类型：

- `company-output.csv` → company任务
- `summary-output.csv` → summary任务
- 等等...

如果无法自动识别，可以使用`-t`参数手动指定。

## 可视化图表

### 分类任务图表

1. **F1分数分布直方图**: 显示所有样本的F1分数分布
2. **标签性能对比图**: 显示前15个最常见标签的精确率、召回率和F1分数

### 生成任务图表

1. **余弦相似度分布直方图**: 显示所有样本的相似度分布
2. **文本长度分析图**: 
   - 散点图：标准答案长度 vs 模型输出长度
   - 直方图：两者的长度分布对比

### 批量对比图表

- **模型性能对比**: 横向对比不同模型在同一任务上的表现
- **任务间对比**: 显示同一模型在不同任务上的表现

## 报告内容

### 单个文件报告

```
============================================================
CFBenchmark CSV评估报告
============================================================

文件路径: company-output.csv
任务类型: classification
分类类型: company
样本数量: 51

主要指标:
------------------------------
平均F1分数: 0.8235
平均精确率: 0.8431
平均召回率: 0.8039

标签统计:
唯一标签数量: 45

标签性能排行 (前10个):
--------------------------------------------------
 1. 天康生物              F1: 1.0000 P: 1.0000 R: 1.0000
 2. 洪都航空              F1: 1.0000 P: 1.0000 R: 1.0000
 ...
```

### 批量评估报告

```
================================================================================
CFBenchmark 批量评估汇总报告
================================================================================

评估时间: 2024-01-15 10:30:45
输入目录: cfbenchmark-response
输出目录: cfbenchmark-response/batch_evaluation_results
评估文件数: 16

COMPANY 任务评估结果:
------------------------------------------------------------
文件名                                   F1分数     精确率     召回率     样本数
--------------------------------------------------------------------------------
few-shot_gpt-4o-mini_company            0.8235    0.8431    0.8039       51
zero-shot_gpt-4o-mini_company           0.7892    0.8123    0.7654       51
--------------------------------------------------------------------------------
平均值                                   0.8064    0.8277    0.7847      102
```

## 常见问题

### Q: 如何处理中文文本？
A: 评估工具已经内置了中文文本处理功能，包括中文标点符号的处理和中文字符的分词。

### Q: 如何自定义评估指标？
A: 可以修改`CSVEvaluator`类中的相关方法来添加自定义指标。

### Q: 批量评估时如何跳过某些文件？
A: 可以将不需要评估的文件移到其他目录，或者修改文件名使其不包含"-output"。

### Q: 如何处理空值或缺失数据？
A: 评估工具会自动处理空值，将其转换为空字符串进行处理。

## 高级用法

### 自定义标签提取

如果需要自定义标签提取逻辑，可以继承`CSVEvaluator`类并重写`extract_labels`方法：

```python
class CustomEvaluator(CSVEvaluator):
    def extract_labels(self, text: str) -> List[str]:
        # 自定义标签提取逻辑
        return custom_label_extraction(text)
```

### 添加新的评估指标

```python
def custom_metric(self, response_labels, output_labels):
    # 实现自定义指标
    return metric_value

# 在evaluate方法中调用
custom_score = self.custom_metric(response_labels, output_labels)
```

## 性能优化

- 对于大型数据集，建议使用批量评估以提高效率
- 可以使用`--no_plots`参数跳过图表生成以节省时间
- 对于内存受限的环境，可以分批处理大文件

## 故障排除

### 常见错误及解决方案

1. **"CSV文件缺少必需的列"**
   - 检查CSV文件是否包含input, response, output三列

2. **"文件不存在"**
   - 检查文件路径是否正确
   - 确保文件具有读取权限

3. **"matplotlib相关错误"**
   - 安装matplotlib: `pip install matplotlib`
   - 如果是服务器环境，可能需要设置DISPLAY环境变量

4. **"中文显示乱码"**
   - 确保系统安装了中文字体
   - 可以修改脚本中的字体设置

## 联系支持

如果遇到问题或需要功能建议，请：

1. 检查本文档的故障排除部分
2. 查看脚本的错误输出信息
3. 提交issue到项目仓库

---

*最后更新: 2024-01-15* 