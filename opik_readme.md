# Opik 核心概念 Demo 测试

## 1. 概述

本文件展示了 Opik 框架中核心概念的使用方法，包括：
- Traces & Spans
- Threads
- Experiments
- Online Evaluation
- Metrics
- Annotation Queues
- Optimizer
- Traces & Spans

## 2. 环境设置

```python
import opik
import datetime
import uuid
from typing import Dict, List, Any

# 初始化 Opik 客户端
client = opik.Opik(
    project_name="opik-core-concepts-demo",
    host="http://localhost:5173/api",
    # api_key="your-api-key",  # 如果需要认证
    _use_batching=True
)
```

## 3. Traces & Spans 基础

### 3.1 创建 Trace 和 Span

```python
# 创建一个新的 trace
with client.trace(
    name="user-query-processing",
    input={"query": "What is Opik?"},
    metadata={"user_id": "demo-user-123"}
) as trace:
    
    # 在 trace 中创建 span
    with trace.span(
        name="document-retrieval",
        type="retrieval",
        input={"query": "What is Opik?"}
    ) as retrieval_span:
        # 模拟文档检索
        retrieved_docs = [
            {"content": "Opik is an AI monitoring and evaluation platform...", "score": 0.95},
            {"content": "Opik provides trace-based monitoring...", "score": 0.88}
        ]
        retrieval_span.update(
            output={"documents": retrieved_docs},
            metadata={"doc_count": len(retrieved_docs)}
        )
    
    # 创建 LLM 调用 span
    with trace.span(
        name="llm-generation",
        type="llm",
        input={"prompt": "Summarize: Opik is an AI monitoring...", "model": "gpt-3.5-turbo"},
        metadata={"temperature": 0.7}
    ) as llm_span:
        # 模拟 LLM 生成
        generated_response = "Opik is an AI monitoring and evaluation platform that provides trace-based monitoring, experiment management, and evaluation tools for AI applications."
        llm_span.update(
            output={"response": generated_response},
            usage={"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200},
            total_cost=0.0004  # 模拟成本
        )
    
    # 更新 trace 输出
    trace.update(
        output={"response": generated_response},
        metadata={"processing_time": 0.85}  # 模拟处理时间
    )
```

### 3.2 手动创建 Trace 和 Span

```python
# 手动创建 trace
trace_id = str(uuid.uuid4())
client.trace(
    id=trace_id,
    name="manual-trace",
    start_time=datetime.datetime.now(),
    input={"test": "data"}
)

# 手动创建 span
span_id = str(uuid.uuid4())
client.span(
    id=span_id,
    trace_id=trace_id,
    name="manual-span",
    type="general",
    start_time=datetime.datetime.now(),
    input={"span_input": "test"},
    output={"span_output": "result"}
)

# 结束 trace
client.trace_end(
    trace_id=trace_id,
    end_time=datetime.datetime.now(),
    output={"final_result": "success"}
)
```

## 4. Threads 线程管理

```python
# 创建一个线程 ID 来关联多个 traces
thread_id = f"user-session-{uuid.uuid4()}"

# 第一个 trace
with client.trace(
    name="user-query-1",
    input={"query": "What is Opik?"},
    thread_id=thread_id
) as trace:
    # 处理逻辑...
    trace.update(output={"response": "Opik is an AI monitoring platform."})

# 第二个 trace，同一线程
with client.trace(
    name="user-query-2",
    input={"query": "How does Opik work?"},
    thread_id=thread_id
) as trace:
    # 处理逻辑...
    trace.update(output={"response": "Opik uses traces and spans to monitor AI applications."})

# 第三个 trace，同一线程
with client.trace(
    name="user-query-3",
    input={"query": "What features does Opik have?"},
    thread_id=thread_id
) as trace:
    # 处理逻辑...
    trace.update(output={"response": "Opik has tracing, experiments, evaluation, and metrics."})
```

## 5. Dataset & Experiments

### 5.1 创建 Dataset

```python
# 创建或获取数据集
dataset = client.get_or_create_dataset(name="qa-eval-dataset")

# 插入数据项
dataset.insert([
    {"input": "What is Opik?", "expected_output": "Opik is an AI monitoring and evaluation platform."},
    {"input": "How does Opik work?", "expected_output": "Opik uses traces and spans to monitor AI applications."},
    {"input": "What features does Opik have?", "expected_output": "Opik has tracing, experiments, evaluation, and metrics."},
    {"input": "Is Opik open source?", "expected_output": "Yes, Opik is open source and available on GitHub."},
    {"input": "How to get started with Opik?", "expected_output": "You can get started with Opik by installing the SDK and following the documentation."}
])

# 获取数据集项
items = dataset.get_items()
dataset_item_map = {item["input"]: item["id"] for item in items}
```

### 5.2 创建 Experiment

```python
# 创建实验
experiment = client.create_experiment(
    name="llm-qa-performance",
    dataset_name="qa-eval-dataset"
)

# 运行实验 - 为每个数据集项创建 trace
for item in items:
    input_query = item["input"]
    expected_output = item["expected_output"]
    
    with client.trace(
        name="experiment-run",
        input={"query": input_query, "expected": expected_output},
        metadata={"dataset_item_id": item["id"]}
    ) as trace:
        
        # 模拟 LLM 响应
        simulated_response = f"{expected_output} (simulated response)"
        
        # 创建评估 span
        with trace.span(
            name="qa-evaluation",
            type="evaluation",
            input={
                "query": input_query,
                "expected": expected_output,
                "actual": simulated_response
            }
        ) as eval_span:
            # 计算简单的相似度分数
            similarity = 0.9 if expected_output in simulated_response else 0.5
            eval_span.update(
                output={
                    "similarity_score": similarity,
                    "pass": similarity > 0.8
                },
                metadata={"evaluation_method": "simple-similarity"}
            )
        
        trace.update(
            output={"response": simulated_response},
            metadata={"experiment_name": "llm-qa-performance"}
        )

# 将 traces 关联到实验
experiment_items = []
for item in items:
    # 这里需要实际的 trace_id，可以从之前的 trace 中获取
    # 这里简化处理
    experiment_items.append(
        opik.api_objects.experiment.experiment_item.ExperimentItemReferences(
            dataset_item_id=item["id"],
            trace_id="sample-trace-id"  # 替换为实际 trace_id
        )
    )

experiment.insert(experiment_items)
```

## 6. Online Evaluation

### 6.1 评估单个 Prompt

```python
# 定义评估函数
def evaluate_response(actual: str, expected: str) -> Dict[str, Any]:
    """简单的评估函数，计算相似度和准确性"""
    similarity = 1.0 if expected.lower() in actual.lower() else 0.0
    accuracy = 1.0 if actual.strip() == expected.strip() else 0.0
    return {
        "similarity": similarity,
        "accuracy": accuracy,
        "pass": similarity >= 0.8
    }

# 评估 prompt
prompt = "What is Opik?"
expected = "Opik is an AI monitoring and evaluation platform."
actual = "Opik is an AI monitoring and evaluation platform for LLM applications."

# 记录评估结果
with client.trace(
    name="online-evaluation",
    input={"prompt": prompt, "expected": expected}
) as trace:
    
    # 执行评估
    evaluation_results = evaluate_response(actual, expected)
    
    # 记录评估 span
    trace.span(
        name="response-evaluation",
        type="evaluation",
        input={"actual": actual, "expected": expected},
        output=evaluation_results,
        metadata={"evaluation_method": "custom-similarity"}
    )
    
    # 更新 trace 输出
    trace.update(
        output={"response": actual, "evaluation": evaluation_results}
    )
```

### 6.2 使用评估装饰器

```python
# 使用 opik 的评估装饰器
@opik.evaluate(
    dataset_name="qa-eval-dataset",
    metrics=["accuracy", "similarity"]
)
def qa_pipeline(query: str) -> str:
    """模拟 QA 管道"""
    # 实际应用中，这里会调用 LLM 和其他组件
    return f"Answer to: {query} (simulated)"

# 运行评估
qa_pipeline("What is Opik?")
```

## 7. Metrics 指标管理

### 7.1 记录自定义指标

```python
# 创建 trace 并记录指标
with client.trace(
    name="metric-demo-trace",
    input={"query": "What is Opik?"}
) as trace:
    
    # 模拟处理
    processing_time = 0.75
    tokens_used = 250
    cost = 0.0005
    
    # 记录指标
    trace.update(
        output={"response": "Opik is an AI monitoring platform."},
        metadata={
            "processing_time": processing_time,
            "tokens_used": tokens_used,
            "cost": cost,
            "model_used": "gpt-3.5-turbo",
            "success": True
        }
    )
    
    # 添加反馈分数
    trace.log_feedback_score(
        name="user-satisfaction",
        value=4.5,
        category_name="feedback",
        reason="Helpful response"
    )
```

### 7.2 记录 LLM 使用指标

```python
with client.trace(
    name="llm-metrics-demo",
    input={"prompt": "Explain Opik in simple terms."}
) as trace:
    
    with trace.span(
        name="llm-call",
        type="llm",
        model="gpt-4",
        provider="openai"
    ) as llm_span:
        
        # 模拟 LLM 响应
        response = "Opik is a tool that helps you monitor and improve your AI applications by tracking how they work."
        
        # 记录 LLM 使用指标
        llm_span.update(
            output={"response": response},
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "total_tokens": 130
            },
            total_cost=0.0026  # GPT-4 成本：$0.03/1K tokens
        )
    
    trace.update(output={"response": response})
```

## 8. Annotation Queues

```python
# 创建 trace 并添加到注释队列
with client.trace(
    name="annotation-queue-demo",
    input={"query": "Complex query that needs human review"},
    metadata={"needs_annotation": True}
) as trace:
    
    # 模拟复杂响应
    complex_response = "This is a complex response that requires human annotation to verify accuracy and quality."
    
    trace.update(
        output={"response": complex_response},
        tags=["needs-annotation", "complex-query"]
    )
    
    # 记录需要注释的原因
    trace.span(
        name="annotation-flag",
        type="annotation",
        input={"query": "Complex query that needs human review"},
        output={"needs_annotation": True},
        metadata={
            "annotation_reason": "complex-response",
            "priority": "high"
        }
    )
```

## 9. Optimizer

### 9.1 使用 Optimizer 优化 Prompt

```python
# 定义基础 prompt
base_prompt = """Answer the following question about Opik:
Question: {{query}}
Answer:"""

# 定义优化目标
optimizer_config = {
    "max_tokens": 100,
    "temperature": 0.7,
    "metrics": ["conciseness", "accuracy", "relevance"]
}

# 创建优化任务
optimizer = client.create_optimizer(
    name="prompt-optimization-demo",
    prompt_template=base_prompt,
    config=optimizer_config
)

# 运行优化
optimizer.run(
    dataset_name="qa-eval-dataset",
    iterations=3
)

# 获取优化结果
optimized_prompts = optimizer.get_optimized_prompts()
print(f"Optimized prompts: {optimized_prompts}")
```

## 10. 完整的端到端示例

```python
def run_end_to_end_demo():
    """完整的端到端示例，展示所有核心概念"""
    
    # 创建数据集
    dataset = client.get_or_create_dataset(name="end-to-end-demo-dataset")
    dataset.insert([
        {"input": "What is Opik?", "expected": "Opik is an AI monitoring platform."},
        {"input": "How to use Opik?", "expected": "Install the SDK and start tracking traces."}
    ])
    
    # 创建实验
    experiment = client.create_experiment(
        name="end-to-end-demo-experiment",
        dataset_name="end-to-end-demo-dataset"
    )
    
    # 处理每个数据集项
    items = dataset.get_items()
    experiment_items = []
    
    for item in items:
        query = item["input"]
        expected = item["expected"]
        thread_id = f"demo-thread-{uuid.uuid4()}"
        
        # 创建 trace
        with client.trace(
            name="end-to-end-demo-trace",
            input={"query": query, "expected": expected},
            thread_id=thread_id,
            metadata={"demo_name": "end-to-end"}
        ) as trace:
            
            # 1. 文档检索
            with trace.span(
                name="document-retrieval",
                type="retrieval",
                input={"query": query}
            ) as retrieval_span:
                retrieved_docs = [
                    {"content": expected, "score": 0.95}
                ]
                retrieval_span.update(
                    output={"documents": retrieved_docs},
                    metadata={"doc_count": 1}
                )
            
            # 2. LLM 生成
            with trace.span(
                name="llm-generation",
                type="llm",
                model="gpt-3.5-turbo",
                provider="openai",
                input={"prompt": f"Answer: {query}"}
            ) as llm_span:
                response = f"{expected} (generated by LLM)"
                llm_span.update(
                    output={"response": response},
                    usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
                    total_cost=0.00014
                )
            
            # 3. 评估
            with trace.span(
                name="response-evaluation",
                type="evaluation",
                input={
                    "query": query,
                    "expected": expected,
                    "actual": response
                }
            ) as eval_span:
                similarity = 1.0 if expected in response else 0.5
                eval_span.update(
                    output={
                        "similarity": similarity,
                        "accuracy": similarity,
                        "pass": similarity > 0.8
                    }
                )
            
            # 4. 更新 trace
            trace.update(
                output={"response": response},
                metadata={"experiment_name": "end-to-end-demo-experiment"}
            )
            
            # 5. 添加到实验
            experiment_items.append(
                opik.api_objects.experiment.experiment_item.ExperimentItemReferences(
                    dataset_item_id=item["id"],
                    trace_id=trace.id
                )
            )
    
    # 将项目添加到实验
    experiment.insert(experiment_items)
    
    print("End-to-end demo completed successfully!")

# 运行端到端示例
run_end_to_end_demo()

# 刷新所有数据
client.flush()
```

## 11. 高级用法：Spans 嵌套和类型

```python
with client.trace(
    name="advanced-spans-demo",
    input={"query": "Advanced Opik features"}
) as trace:
    
    # 嵌套 span 示例
    with trace.span(
        name="multi-step-processing",
        type="pipeline"
    ) as pipeline_span:
        
        # 第一步：数据预处理
        with pipeline_span.span(
            name="data-preprocessing",
            type="preprocessing",
            input={"raw_input": "Advanced Opik features"}
        ) as preprocessing_span:
            processed_input = "Opik advanced features"
            preprocessing_span.update(output={"processed": processed_input})
        
        # 第二步：特征提取
        with pipeline_span.span(
            name="feature-extraction",
            type="extraction",
            input={"processed_input": processed_input}
        ) as extraction_span:
            features = ["tracing", "evaluation", "experiments", "metrics"]
            extraction_span.update(output={"features": features})
        
        # 第三步：最终处理
        with pipeline_span.span(
            name="final-processing",
            type="processing",
            input={"features": features}
        ) as final_span:
            result = f"Opik advanced features: {', '.join(features)}"
            final_span.update(output={"result": result})
        
        pipeline_span.update(output={"final_result": result})
    
    trace.update(output={"response": result})
```

## 12. 总结

本示例展示了 Opik 框架中核心概念的使用方法，包括：

1. **Traces & Spans**：用于跟踪和监控 AI 应用的执行流程
2. **Threads**：用于将相关的 traces 分组，方便分析用户会话
3. **Experiments**：用于比较不同模型或配置的性能
4. **Online Evaluation**：用于实时评估模型输出
5. **Metrics**：用于衡量模型性能和成本
6. **Annotation Queues**：用于标记需要人工注释的内容
7. **Optimizer**：用于优化 prompts 和模型配置

通过这些概念的组合使用，您可以全面监控、评估和优化您的 AI 应用。

## 13. 运行示例

```bash
# 安装依赖
pip install opik

# 运行示例
python opik_readme.md  # 注意：需要将示例代码保存为 .py 文件
```

## 14. 进一步学习

- 查看 [Opik 文档](https://docs.opik.com/) 了解更多详情
- 探索 [Opik GitHub 仓库](https://github.com/comet-ml/opik) 获取源代码
- 参加 [Opik 社区](https://community.opik.com/) 与其他用户交流

---

**注意**：本示例中的 API 调用可能需要根据您的 Opik 版本进行调整。请参考最新的 Opik 文档获取准确的 API 信息。