import opik
from opik import Opik, track
from opik.evaluation.metrics import BaseMetric

# 先配置本地模式
opik.configure(use_local=True)

# 然后创建 Opik 客户端，设置 project name
client = Opik(project_name="opik-demo-project")

print("=== Opik SDK Python 简化功能演示 ===\n")

# 1. 追踪功能演示
print("1. 追踪功能演示")
print("-" * 50)

# 装饰器追踪示例
@track
def my_llm_function(user_question: str) -> str:
    """使用装饰器追踪 LLM 函数"""
    response = f"Echoing: {user_question}"
    return response

# 客户端手动创建 trace 示例
def client_trace_demo():
    """使用客户端手动创建 trace"""
    # 手动创建 trace
    trace = client.trace(
        name="client-manual-trace",
        input={"question": "How to use Opik client?"},
        output={"answer": "Use client.trace() method"}
    )
    return trace

# 上下文管理器追踪示例（使用 opik 模块）
def context_manager_demo():
    """使用上下文管理器追踪代码块"""
    import opik
    with opik.start_as_current_trace("my-trace"):
        with opik.start_as_current_span("my-span"):
            result = "Context manager trace result"
            return result

# 运行追踪演示
trace_result1 = my_llm_function("Hello, Opik!")
print(f"   装饰器追踪结果: {trace_result1}")

trace_result2 = context_manager_demo()
print(f"   上下文管理器追踪结果: {trace_result2}")

# 运行客户端手动trace演示
client_trace = client_trace_demo()
print(f"   客户端手动trace结果: {client_trace}")

print()

# 2. 基本评估功能演示
print("2. 基本评估功能演示")
print("-" * 50)

# 创建简单数据集
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

# 简单的LLM模拟函数
def simple_llm(input_text: str) -> str:
    if "2 + 2" in input_text:
        return "4"
    elif "capital of France" in input_text:
        return "Paris"
    else:
        return f"Answer: {input_text}"

print("   数据集: {dataset}")
print("   模拟LLM响应:")
for item in dataset:
    response = simple_llm(item["input"])
    print(f"   - 输入: {item['input']}")
    print(f"     参考: {item['reference']}")
    print(f"     实际: {response}")
    print(f"     匹配: {response == item['reference']}")

print()
print("=" * 50)
print("演示完成！")

# 关闭客户端连接
client.end()
