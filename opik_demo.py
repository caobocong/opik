import opik
from opik import Opik, track
from opik.evaluation.metrics import BaseMetric

# 配置 Opik（本地模式）
opik.configure(use_local=True)

# 创建 Opik 客户端，设置 project name
client = Opik(project_name="opik-demo-project")

print("=== Opik SDK Python 功能演示 ===\n")

# 1. 追踪功能演示
print("1. 追踪功能演示")
print("-" * 50)

# 装饰器追踪示例
@track
def my_llm_function(user_question: str) -> str:
    """使用装饰器追踪 LLM 函数"""
    response = f"Echoing: {user_question}"
    return response

# 上下文管理器追踪示例
def context_manager_demo():
    """使用上下文管理器追踪代码块"""
    with opik.start_as_current_trace("my-trace"):
        with opik.start_as_current_span("my-span"):
            result = "Context manager trace result"
            return result

# 客户端手动创建 trace 示例
def client_trace_demo():
    """使用客户端手动创建 trace"""
    trace = client.trace(
        name="client-manual-trace",
        input={"question": "How to use Opik client?"},
        output={"answer": "Use client.trace() method"}
    )
    return trace

# 运行追踪演示
trace_result1 = my_llm_function("Hello, Opik!")
print(f"   装饰器追踪结果: {trace_result1}")

trace_result2 = context_manager_demo()
print(f"   上下文管理器追踪结果: {trace_result2}")

# 运行客户端手动trace演示
client_trace = client_trace_demo()
print(f"   客户端手动trace结果: {client_trace}")

print()

# 2. 评估功能演示
print("2. 评估功能演示")
print("-" * 50)

# 创建演示数据集
def create_demo_dataset():
    """创建演示评估数据集"""
    dataset = [
        {
            "input": "What is 2 + 2?",
            "reference": "4"
        },
        {
            "input": "What is the capital of France?",
            "reference": "Paris"
        }
    ]
    return dataset

# 简单LLM任务函数
def simple_llm_task(input_text: str) -> str:
    """简单的LLM任务模拟"""
    if "2 + 2" in input_text:
        return "4"
    elif "capital of France" in input_text:
        return "Paris"
    else:
        return f"Answering: {input_text}"

# 执行评估
dataset = create_demo_dataset()

# 使用内置评估指标或简单评估
try:
    result = opik.evaluate(
        dataset=dataset,
        task=simple_llm_task,
        experiment_name="demo-experiment"
    )
    print(f"   评估结果: {result}")
except Exception as e:
    print(f"   评估功能可能需要更复杂的配置: {e}")

print()

print("=" * 50)
print("演示完成！")

# 关闭客户端连接
client.end()
