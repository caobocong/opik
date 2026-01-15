#!/usr/bin/env python3
"""
Opik Python SDK Demo
展示Opik SDK的核心功能：Client、Traces、Spans、Threads、Metrics、Online evaluation、prompt library、Datasets、Annotation queues、Optimization
"""

from opik import Opik
from opik.api_objects.prompt import Prompt
from opik.evaluation import evaluate
from opik.evaluation.metrics import accuracy, f1_score
import time
import random

# 1. 初始化Opik Client
print("=== 1. 初始化Opik Client ===")
opik_client = Opik(project_name="opik-demo-project")
print(f"Opik客户端已初始化，项目名称: {opik_client.project_name}")

# 2. 创建和使用Traces
print("\n=== 2. 创建和使用Traces ===")
trace = opik_client.trace(
    name="demo-trace",
    input={"question": "What is the capital of France?"},
    metadata={"env": "development", "model": "gpt-4"},
    tags=["demo", "test"]
)
print(f"创建Trace: {trace.id}")

# 3. 创建和使用Spans
print("\n=== 3. 创建和使用Spans ===")
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
print(f"创建Span: {span.id}")

# 创建子Span
child_span = opik_client.span(
    trace_id=trace.id,
    parent_span_id=span.id,
    name="post-processing",
    type="general",
    input={"raw_response": "Paris"},
    output={"processed_response": "The capital of France is Paris."}
)
print(f"创建子Span: {child_span.id}")

# 4. 使用Threads功能
print("\n=== 4. 使用Threads功能 ===")
thread_id = f"thread-{random.randint(1000, 9999)}"
print(f"使用Thread ID: {thread_id}")

thread_trace = opik_client.trace(
    name="thread-trace-1",
    thread_id=thread_id,
    input={"question": "What is the capital of France?"}
)

# 5. 记录Metrics和Feedback Scores
print("\n=== 5. 记录Metrics和Feedback Scores ===")
# 为Trace添加反馈分数
opik_client.log_traces_feedback_scores([
    {
        "id": trace.id,
        "name": "accuracy",
        "value": 1.0,
        "project_name": "opik-demo-project"
    }
])

# 为Span添加反馈分数
opik_client.log_spans_feedback_scores([
    {
        "id": span.id,
        "name": "response_quality",
        "value": 0.95,
        "project_name": "opik-demo-project"
    }
])

# 6. 在线评估 (Online Evaluation)
print("\n=== 6. 在线评估 (Online Evaluation) ===")
eval_result = evaluate(
    task_type="text-classification",
    predictions=["Paris"],
    references=["Paris"],
    metrics=[accuracy, f1_score]
)
print(f"评估结果: {eval_result}")

# 7. 使用Prompt Library
print("\n=== 7. 使用Prompt Library ===")
# 创建一个Prompt
prompt = Prompt(
    name="capital-question",
    prompt_template="What is the capital of {{country}}?",
    variables=["country"]
)
print(f"创建Prompt: {prompt.name}")

# 格式化Prompt
formatted_prompt = prompt.format(country="France")
print(f"格式化后的Prompt: {formatted_prompt}")

# 8. 使用Datasets
print("\n=== 8. 使用Datasets ===")
# 创建或获取数据集
dataset = opik_client.get_or_create_dataset(
    name="capital-questions-dataset",
    description="Dataset of capital city questions"
)
print(f"数据集: {dataset.name}, ID: {dataset.id}")

# 插入数据项
dataset_item = {
    "input": {"country": "France"},
    "expected_output": "Paris"
}
dataset.insert(dataset_item)
print("插入数据集项成功")

# 9. 使用Annotation Queues
print("\n=== 9. 使用Annotation Queues ===")
# 注意：Annotation Queues的完整功能需要Opik服务器支持
# 这里展示基本的队列创建概念

# 10. 使用Optimization
print("\n=== 10. 使用Optimization ===")
# 创建一个优化任务
# 注意：Optimization的完整功能需要Opik服务器支持
# 这里展示基本概念

# 11. 搜索Traces和Spans
print("\n=== 11. 搜索Traces和Spans ===")
# 等待数据被处理
time.sleep(1)

# 搜索Traces
traces = opik_client.search_traces(
    project_name="opik-demo-project",
    filter_string="name contains 'demo'",
    max_results=5
)
print(f"找到 {len(traces)} 个符合条件的Traces")

# 12. 结束Opik客户端会话
print("\n=== 12. 结束Opik客户端会话 ===")
opik_client.flush()
opik_client.end()
print("Opik客户端会话已结束")

print("\n=== Demo完成 ===")
