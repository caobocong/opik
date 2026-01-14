import opik
from typing import Dict, List, Any
# opik.configure(use_local=True)
# 初始化 Opik 客户端
client = opik.Opik(
    project_name="advisor",
    host="http://localhost:5173/api",
    _use_batching=True
)

# 直接创建 trace 对象
trace = client.trace(
    name="user-query-processing",
    input={"query": "What is Opik?"},
    metadata={"user_id": "demo-user-123"}
)

# 创建文档检索 span
retrieval_span = trace.span(
    name="document-retrieval",
    type="general",  # 修复：改为支持的类型：general, tool, llm, guardrail
    input={"query": "What is Opik?"}
)

# 模拟文档检索并更新 span
retrieved_docs = [
    {"content": "Opik is an AI monitoring and evaluation platform...", "score": 0.95},
    {"content": "Opik provides trace-based monitoring...", "score": 0.88}
]
retrieval_span.update(
    output={"documents": retrieved_docs},
    metadata={"doc_count": len(retrieved_docs)}
)

# 创建 LLM 调用 span
generated_response = "Opik is an AI monitoring and evaluation platform that provides trace-based monitoring, experiment management, and evaluation tools for AI applications."
llm_span = trace.span(
    name="llm-generation",
    type="llm",
    input={"prompt": "Summarize: Opik is an AI monitoring...", "model": "gpt-3.5-turbo"},
    metadata={"temperature": 0.7}
)

# 模拟 LLM 生成并更新 span
llm_span.update(
    output={"response": generated_response},
    usage={"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200},
    total_cost=0.0004
)

# 更新 trace 输出
trace.update(
    output={"response": generated_response},
    metadata={"processing_time": 0.85}
)


# 创建一个线程 ID 来关联多个 traces
thread_id = "user-query-processing"

# 第一个 trace
trace = client.trace(
    name="user-query-1",
    input={"query": "What is Opik?"},
    thread_id=thread_id
)
trace.update(output={"response": "Opik is an AI monitoring platform."})

# 第二个 trace，同一线程
trace = client.trace(
    name="user-query-2",
    input={"query": "How does Opik work?"},
    thread_id=thread_id
)
# 处理逻辑...
trace.update(output={"response": "Opik uses traces and spans to monitor AI applications."})

# 第三个 trace，同一线程
trace = client.trace(
    name="user-query-3",
    input={"query": "What features does Opik have?"},
    thread_id=thread_id
)
# 处理逻辑...
trace.update(output={"response": "Opik has tracing, experiments, evaluation, and metrics."})


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

# 创建实验
experiment = client.create_experiment(
    name="llm-qa-performance",
    dataset_name="qa-eval-dataset"
)

# 运行实验 - 为每个数据集项创建 trace
for item in items:
    input_query = item["input"]
    expected_output = item["expected_output"]

    trace = client.trace(
            name="experiment-run",
            input={"query": input_query, "expected": expected_output},
            metadata={"dataset_item_id": item["id"]}
    )
    # 模拟 LLM 响应
    simulated_response = f"{expected_output} (simulated response)"

    # 创建评估 span
    eval_span = trace.span(
            name="qa-evaluation",
            type="evaluation",
            input={
                "query": input_query,
                "expected": expected_output,
                "actual": simulated_response
            })
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


if __name__ == "__main__":
    print("Opik client initialized")
    print(f"Project: advisor, Host: http://localhost:5173/api")