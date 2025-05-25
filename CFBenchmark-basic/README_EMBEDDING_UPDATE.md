# CFBenchmark Embedding Update

## 更新说明

本次更新将CFBenchmark的embedding逻辑从`sentence-transformers`库迁移到了`transformers`库，参考了Old代码的实现模式。

## 主要变更

### 1. 依赖库变更
- **之前**: 使用 `sentence-transformers` 库
- **现在**: 使用 `transformers` 库中的 `AutoModel` 和 `AutoTokenizer`

### 2. 初始化参数新增
```python
CFBenchmark(
    # ... 其他参数
    embedding_model_path: str = "BAAI/bge-large-zh-v1.5",  # 新增参数
    # ... 其他参数
)
```

### 3. Embedding模型加载
```python
# 新的加载方式 (参考Old代码)
self.t2v_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_path)
self.t2v_model = AutoModel.from_pretrained(
    self.embedding_model_path,
    load_in_8bit=False,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
self.t2v_model.eval()
```

### 4. 余弦相似度计算
完全按照Old代码的模式重写了`get_cosine_similarities`方法：

```python
def get_cosine_similarities(self, row):
    sentences_1 = str(row['output'])
    sentences_2 = str(row['response'])
    
    # 使用transformers模型计算embedding
    encoded_input = self.t2v_tokenizer([sentences_1, sentences_2], 
                                      padding=True, truncation=True, 
                                      return_tensors='pt', max_length=512)
    
    with torch.no_grad():
        model_output = self.t2v_model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]  # 使用CLS token
    
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    cosine_sim = torch.nn.functional.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0)
    return cosine_sim.item()
```

## 安装依赖

### 方法1: 使用内置方法
```python
from CFBenchmark import CFBenchmark
CFBenchmark.install_optional_dependencies()
```

### 方法2: 手动安装
```bash
pip install transformers accelerate torch datasets
```

## 使用示例

```python
from CFBenchmark import CFBenchmark

# 初始化 (现在包含embedding_model_path参数)
benchmark = CFBenchmark(
    api_key="your_api_key",
    embedding_model_path="BAAI/bge-large-zh-v1.5",  # 可以更换为其他模型
    max_workers=4
)

# 运行基准测试
benchmark.get_model_results()
benchmark.get_test_scores()
```

## 测试

运行测试脚本验证embedding功能：

```bash
cd CFBenchmark-basic
python test_embedding.py
```

## 兼容性

- 如果transformers模型加载失败，系统会自动回退到增强的TF-IDF余弦相似度计算
- 保持了与原有API的兼容性
- 支持自定义embedding模型路径

## 优势

1. **更好的控制**: 直接使用transformers库，可以更精细地控制模型加载和推理过程
2. **内存效率**: 可以通过`device_map`和`torch_dtype`参数优化内存使用
3. **模型灵活性**: 可以轻松切换不同的预训练模型
4. **与Old代码一致**: 完全按照Old代码的模式实现，保证了一致性

## 注意事项

- 首次运行时会下载模型文件，请确保网络连接正常
- 建议使用GPU以获得更好的性能
- 如果遇到CUDA内存不足，可以调整`torch_dtype`或使用CPU模式 