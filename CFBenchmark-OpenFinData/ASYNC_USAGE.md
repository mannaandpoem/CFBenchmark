# 异步并行执行使用指南

本文档介绍如何使用新增的异步并行执行功能来提高 CFBenchmark 的处理效率。

## 主要改进

### 1. 异步 API 调用
- `gpt_api` 函数已转换为协程，支持异步调用
- 使用 `AsyncOpenAI` 客户端进行非阻塞 API 请求
- 支持并发处理多个 API 请求

### 2. 并行评估处理
- `exec_fineva_main.py` 支持并行处理多个评估任务
- `get_score.py` 支持并行计算合规性和生成质量分数
- 使用 `asyncio.Semaphore` 控制并发数量

### 3. 评估器异步支持
- `ChatGPTEvaluator` 新增 `answer_async` 方法
- 向后兼容原有的同步 `answer` 方法

## 使用方法

### 运行主评估脚本

```bash
# 基本用法
python exec_fineva_main.py \
    --model_name "gpt-3.5-turbo" \
    --model_path "gpt-3.5-turbo" \
    --save_path "./results" \
    --api_key "your_api_key" \
    --max_workers 10

# 使用自定义 API 端点
python exec_fineva_main.py \
    --model_name "gpt-3.5-turbo" \
    --model_path "gpt-3.5-turbo" \
    --save_path "./results" \
    --api_key "your_api_key" \
    --base_url "https://your-custom-endpoint.com/v1" \
    --max_workers 5
```

### 运行评分脚本

```bash
# 基本用法
python get_score.py \
    --model_name "gpt-3.5-turbo" \
    --result_path "./results" \
    --api_key "your_api_key" \
    --max_workers 8

# 使用自定义 API 端点
python get_score.py \
    --model_name "gpt-3.5-turbo" \
    --result_path "./results" \
    --api_key "your_api_key" \
    --base_url "https://your-custom-endpoint.com/v1" \
    --max_workers 5
```

## 参数说明

### 新增参数

- `--max_workers`: 最大并发工作线程数（默认值：5）
  - 建议根据 API 限制和系统性能调整
  - 对于 OpenAI API，建议设置为 5-10
  - 对于自定义端点，可根据服务器性能调整

### 环境变量

```bash
# 设置 OpenAI API 密钥
export OPENAI_API_KEY="your_api_key_here"
```

## 性能优化建议

### 1. 并发数量调优
- **OpenAI API**: 建议 `max_workers=5-10`
- **自定义端点**: 根据服务器性能调整
- **本地模型**: 可以设置更高的并发数

### 2. API 限制考虑
- 注意 API 的速率限制（RPM/TPM）
- 使用 `asyncio.sleep()` 进行适当的延迟
- 监控 API 使用量和成本

### 3. 错误处理
- 自动重试机制（指数退避）
- 异常隔离，单个请求失败不影响其他请求
- 详细的错误日志记录

## 测试异步功能

运行测试脚本验证异步功能：

```bash
python test_async.py
```

该脚本将测试：
- 异步 `gpt_api` 函数
- 异步 `ChatGPTEvaluator`
- 并发请求性能

## 兼容性说明

### 向后兼容
- 原有的同步方法仍然可用
- 现有脚本无需修改即可运行
- 新的异步功能是可选的

### 依赖要求
- Python 3.7+（支持 asyncio）
- openai >= 1.0.0（支持 AsyncOpenAI）
- 其他依赖保持不变

## 故障排除

### 常见问题

1. **API 密钥错误**
   ```
   ValueError: OpenAI API key not found
   ```
   解决方案：设置 `OPENAI_API_KEY` 环境变量或使用 `--api_key` 参数

2. **并发限制错误**
   ```
   Rate limit exceeded
   ```
   解决方案：降低 `max_workers` 值或增加延迟时间

3. **异步运行时错误**
   ```
   RuntimeError: asyncio.run() cannot be called from a running event loop
   ```
   解决方案：确保在正确的上下文中调用异步函数

### 调试技巧

1. 启用详细日志记录
2. 使用较小的 `max_workers` 值进行测试
3. 监控 API 使用情况和响应时间

## 示例代码

### 异步评估器使用

```python
import asyncio
from evaluator.chatgpt_evaluator import ChatGPTEvaluator

async def main():
    evaluator = ChatGPTEvaluator(
        model_type="gpt-3.5-turbo",
        api_key="your_api_key"
    )
    
    # 并行处理多个查询
    queries = [
        "解释机器学习",
        "什么是人工智能？",
        "深度学习的定义"
    ]
    
    tasks = [evaluator.answer_async(query) for query in queries]
    results = await asyncio.gather(*tasks)
    
    for query, result in zip(queries, results):
        print(f"Query: {query}")
        print(f"Result: {result[:100]}...")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
```

### 自定义并发控制

```python
import asyncio
from utils.gpt_utils import gpt_api

async def controlled_batch_processing(queries, max_concurrent=5):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_query(query):
        async with semaphore:
            return await gpt_api(query)
    
    tasks = [process_query(query) for query in queries]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

## 性能对比

使用异步并行执行后的性能提升：

- **单线程处理**: 10个请求 ≈ 50-100秒
- **异步并行处理**: 10个请求 ≈ 10-20秒（5个并发）
- **性能提升**: 约 3-5倍

具体性能提升取决于：
- API 响应时间
- 网络延迟
- 并发数量设置
- 系统资源 