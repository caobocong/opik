# Opik Python SDK 使用指南

本指南将介绍Opik Python SDK的核心功能和使用方法，包括Client、Traces、Spans、Threads、Metrics、Online evaluation、prompt library、Datasets、Annotation queues、Optimization等功能。

## 1. 安装

```bash
pip install opik
```

## 2. 核心功能介绍

### 2.1 Client

Opik Client是SDK的主要入口，用于初始化配置、管理资源和执行各种操作。

**使用示例：**

```python
from opik import Opik

# 初始化Opik Client
opik_client = Opik(project_name="your-project-name")

# 获取客户端配置
print(f"项目名称: {opik_client.project_name}")

# 关闭客户端
opik_client.end()
```

### 2.2 Traces

Trace用于跟踪和记录完整的请求生命周期，包含多个Spans。

**使用示例：**

```python
# 创建Trace
trace = opik_client.trace(
    name="demo-trace",
    input={"question": "What is the capital of France?"},
    metadata={"env": "development", "model": "gpt-4"},
    tags=["demo", "test"]
)

# 搜索Traces
traces = opik_client.search_traces(
    project_name="your-project-name",
    filter_string="name contains 'demo'",
    max_results=5
)
```

### 2.3 Spans

Span代表Trace中的一个具体操作，如LLM调用、数据处理等。

**使用示例：**

```python
# 创建Span
span = opik_client.span(
    trace_id=trace.id,
    name="llm-call",
    type="llm",
    input={"prompt": "What is the capital of France?"},
    output={"response": "Paris"},
    model="gpt-4",
    provider="openai",
    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
)

# 创建子Span
child_span = opik_client.span(
    trace_id=trace.id,
    parent_span_id=span.id,
    name="post-processing",
    type="general",
    input={"raw_response": "Paris"},
    output={"processed_response": "The capital of France is Paris."}
)
```

### 2.4 Threads

Threads用于将多个Traces分组到一个会话中，便于跟踪完整的用户交互。

**使用示例：**

```python
thread_id = "user-session-123"

# 创建带Thread ID的Trace
trace = opik_client.trace(
    name="user-query-1",
    thread_id=thread_id,
    input={"question": "What is the capital of France?"}
)
```

### 2.5 Metrics & Feedback Scores

用于记录和跟踪模型性能指标和用户反馈。

**使用示例：**

```python
# 为Trace添加反馈分数
opik_client.log_traces_feedback_scores([
    {
        "id": trace.id,
        "name": "accuracy",
        "value": 1.0,
        "project_name": "your-project-name"
    }
])

# 为Span添加反馈分数
opik_client.log_spans_feedback_scores([
    {
        "id": span.id,
        "name": "response_quality",
        "value": 0.95,
        "project_name": "your-project-name"
    }
])
```

### 2.6 Online Evaluation

用于在线评估模型的性能，支持多种评估指标。

**使用示例：**

```python
from opik.evaluation import evaluate
from opik.evaluation.metrics import accuracy, f1_score

# 在线评估
eval_result = evaluate(
    task_type="text-classification",
    predictions=["Paris"],
    references=["Paris"],
    metrics=[accuracy, f1_score]
)

print(f"评估结果: {eval_result}")
```

### 2.7 Prompt Library

用于管理和使用提示词模板，支持变量替换。

**使用示例：**

```python
from opik.api_objects.prompt import Prompt

# 创建Prompt
prompt = Prompt(
    name="capital-question",
    prompt_template="What is the capital of {{country}}?",
    variables=["country"]
)

# 格式化Prompt
formatted_prompt = prompt.format(country="France")
print(f"格式化后的Prompt: {formatted_prompt}")
```

### 2.8 Datasets

用于管理和使用数据集，支持数据插入、更新和查询。

**使用示例：**

```python
# 创建或获取数据集
dataset = opik_client.get_or_create_dataset(
    name="capital-questions-dataset",
    description="Dataset of capital city questions"
)

# 插入数据项
dataset_item = {
    "input": {"country": "France"},
    "expected_output": "Paris"
}
dataset.insert(dataset_item)

# 获取数据集项
items = dataset.list()
print(f"数据集包含 {len(items)} 个项")
```

### 2.9 Annotation Queues

用于管理需要人工标注的数据队列。

**注意：** 完整功能需要Opik服务器支持。

### 2.10 Optimization

用于优化模型和提示词，提高模型性能。

**注意：** 完整功能需要Opik服务器支持。

## 3. 完整示例

以下是一个完整的示例，展示了Opik SDK的主要功能：

```python
#!/usr/bin/env python3
from opik import Opik
from opik.api_objects.prompt import Prompt
from opik.evaluation import evaluate
from opik.evaluation.metrics import accuracy, f1_score
import time
import random

# 1. 初始化Opik Client
opik_client = Opik(project_name="opik-demo-project")

# 2. 创建Trace
trace = opik_client.trace(
    name="demo-trace",
    input={"question": "What is the capital of France?"},
    metadata={"env": "development", "model": "gpt-4"},
    tags=["demo", "test"]
)

# 3. 创建Span
span = opik_client.span(
    trace_id=trace.id,
    name="llm-call",
    type="llm",
    input={"prompt": "What is the capital of France?"},
    output={"response": "Paris"},
    model="gpt-4",
    provider="openai",
    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
)

# 4. 使用Threads
thread_id = f"thread-{random.randint(1000, 9999)}"
thread_trace = opik_client.trace(
    name="thread-trace-1",
    thread_id=thread_id,
    input={"question": "What is the capital of France?"}
)

# 5. 记录Feedback Scores
opik_client.log_traces_feedback_scores([
    {
        "id": trace.id,
        "name": "accuracy",
        "value": 1.0,
        "project_name": "opik-demo-project"
    }
])

# 6. 在线评估
eval_result = evaluate(
    task_type="text-classification",
    predictions=["Paris"],
    references=["Paris"],
    metrics=[accuracy, f1_score]
)

# 7. 使用Prompt Library
prompt = Prompt(
    name="capital-question",
    prompt_template="What is the capital of {{country}}?",
    variables=["country"]
)
formatted_prompt = prompt.format(country="France")

# 8. 使用Datasets
dataset = opik_client.get_or_create_dataset(
    name="capital-questions-dataset",
    description="Dataset of capital city questions"
)
dataset_item = {
    "input": {"country": "France"},
    "expected_output": "Paris"
}
dataset.insert(dataset_item)

# 9. 搜索Traces
time.sleep(1)
traces = opik_client.search_traces(
    project_name="opik-demo-project",
    filter_string="name contains 'demo'",
    max_results=5
)

# 10. 关闭客户端
opik_client.flush()
opik_client.end()
```

## 4. 高级功能

### 4.1 搜索功能

Opik支持强大的搜索功能，可以根据各种条件搜索Traces和Spans。

```python
# 搜索Traces
traces = opik_client.search_traces(
    project_name="your-project-name",
    filter_string="start_time >= '2024-01-01T00:00:00Z' AND usage.total_tokens > 1000",
    max_results=10
)

# 搜索Spans
spans = opik_client.search_spans(
    project_name="your-project-name",
    trace_id=trace.id,
    filter_string="type = 'llm'",
    max_results=5
)
```

### 4.2 批量操作

Opik支持批量操作，提高性能。

```python
# 批量记录反馈分数
scores = [
    {"id": trace1.id, "name": "accuracy", "value": 0.9, "project_name": "your-project-name"},
    {"id": trace2.id, "name": "accuracy", "value": 0.85, "project_name": "your-project-name"}
]
opik_client.log_traces_feedback_scores(scores=scores)
```

## 5. 最佳实践

1. **始终关闭客户端**：使用`opik_client.end()`关闭客户端，确保所有数据被发送到服务器。
2. **使用适当的Trace和Span名称**：清晰的名称有助于后续的搜索和分析。
3. **添加有意义的metadata和tags**：有助于分类和过滤数据。
4. **记录完整的input和output**：便于调试和分析。
5. **使用Threads管理会话**：对于多轮对话，使用Thread ID将相关Traces分组。

## 6. 故障排除

### 6.1 数据未显示在Opik控制台

- 确保调用了`opik_client.flush()`或`opik_client.end()`
- 检查网络连接
- 查看日志获取更多信息

### 6.2 搜索无结果

- 确保使用了正确的项目名称
- 检查过滤条件是否正确
- 等待数据被处理（可能需要几秒钟）

## 7. 相关资源

- [Opik官方文档](https://docs.opik.com/)
- [GitHub仓库](https://github.com/comet-ml/opik)
- [API参考](https://docs.opik.com/api-reference/)

## 8. 版本历史

- v1.0.0: 初始版本
- v1.1.0: 添加了Annotation Queues支持
- v1.2.0: 优化了搜索功能

## 9. 贡献

欢迎提交Issue和Pull Request！

## 10. 许可证

Opik SDK使用MIT许可证。
