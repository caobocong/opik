# Opik SDK Python 全面功能演示

## 1. 简介

Opik SDK 是一个用于集成 Python 应用程序与 Opik 平台的工具，提供了全面的 LLM 系统开发、评估和监控功能。

## 2. 安装和配置

### 2.1 安装

```bash
pip install opik
```

### 2.2 配置

**命令行配置**：
```bash
opik configure
```

**编程方式配置**：
```python
import opik

# 配置 Comet.com 云端
opik.configure(
    api_key="YOUR_API_KEY",
    workspace="YOUR_WORKSPACE",

)

# 或配置自托管 Opik 实例
opik.configure(use_local=True)
```

## 3. 核心功能演示

### 3.1 追踪功能

追踪功能用于记录和可视化 LLM 调用链和业务逻辑。

```python
import opik

# 配置 Opik（本地模式）
opik.configure(use_local=True)

# 1. 装饰器追踪示例
@opik.track
def my_llm_function(user_question: str) -> str:
    """使用装饰器追踪 LLM 函数"""
    response = f"Echoing: {user_question}"
    
    # 添加元数据
    opik.set_tags(["example", "decorator-usage"])
    opik.log_metadata({"question_length": len(user_question)})
    
    return response

# 2. 上下文管理器追踪示例
def context_manager_demo():
    """使用上下文管理器追踪代码块"""
    with opik.start_as_current_trace("my-trace"):
        with opik.start_as_current_span("my-span"):
            result = "Context manager trace result"
            opik.log_metadata({"result_length": len(result)})
            return result

# 3. 动态追踪控制示例
def dynamic_tracing_demo():
    """演示动态启用/禁用追踪"""
    # 禁用追踪
    opik.set_tracing_active(False)
    print(f"追踪状态: {opik.is_tracing_active()}")  # 输出: False
    
    # 重新启用
    opik.set_tracing_active(True)
    print(f"追踪状态: {opik.is_tracing_active()}")  # 输出: True

# 运行追踪演示
if __name__ == "__main__":
    print("=== 追踪功能演示 ===")
    
    # 测试装饰器追踪
    result1 = my_llm_function("Hello, Opik!")
    print(f"装饰器追踪结果: {result1}")
    
    # 测试上下文管理器追踪
    result2 = context_manager_demo()
    print(f"上下文管理器追踪结果: {result2}")
    
    # 测试动态追踪控制
    dynamic_tracing_demo()
```

### 3.2 评估功能

评估功能用于对 LLM 模型、提示词和实验进行多维度评估。

```python
import opik
from opik.evaluation.metrics import BaseMetric

# 配置 Opik
opik.configure(use_local=True)

# 1. 自定义评估指标示例
class CustomSimilarityMetric(BaseMetric):
    """自定义相似度评估指标"""
    
    def score(self, input: str, output: str, reference: str) -> float:
        """计算输出与参考之间的相似度分数"""
        # 简单的字符匹配相似度
        match_count = sum(1 for o, r in zip(output, reference) if o == r)
        max_length = max(len(output), len(reference))
        return match_count / max_length if max_length > 0 else 0.0

# 2. 评估数据集示例
def create_demo_dataset():
    """创建演示评估数据集"""
    dataset = [
        {
            "input": "What is 2 + 2?",
            "reference": "4",
            "metadata": {"difficulty": "easy"}
        },
        {
            "input": "What is the capital of France?",
            "reference": "Paris",
            "metadata": {"difficulty": "medium"}
        },
        {
            "input": "Explain quantum computing in simple terms.",
            "reference": "Quantum computing uses quantum bits or qubits to perform calculations. Unlike classical bits that can be either 0 or 1, qubits can be in multiple states at once, allowing quantum computers to solve certain problems much faster than classical computers.",
            "metadata": {"difficulty": "hard"}
        }
    ]
    return dataset

# 3. 简单LLM任务函数
def simple_llm_task(input_text: str) -> str:
    """简单的LLM任务模拟"""
    if "2 + 2" in input_text:
        return "4"
    elif "capital of France" in input_text:
        return "Paris"
    elif "quantum computing" in input_text:
        return "Quantum computing uses quantum bits to perform calculations faster than classical computers."
    else:
        return f"Answering: {input_text}"

# 4. 评估演示函数
def evaluation_demo():
    """演示评估功能"""
    print("=== 评估功能演示 ===")
    
    # 创建数据集
    dataset = create_demo_dataset()
    
    # 初始化自定义评估指标
    custom_metric = CustomSimilarityMetric()
    
    # 执行评估
    result = opik.evaluate(
        dataset=dataset,
        task=simple_llm_task,
        scoring_metrics=[custom_metric],
        experiment_name="demo-experiment",
        task_threads=2  # 使用2个线程并行执行
    )
    
    print(f"评估结果: {result}")
    print(f"平均分数: {result.avg_scores}")
    print(f"每个样本的分数: {result.scores}")

# 运行评估演示
if __name__ == "__main__":
    evaluation_demo()
```

### 3.3 实验管理

实验管理功能用于创建、运行和比较不同LLM实验。

```python
import opik

# 配置 Opik
opik.configure(use_local=True)

# 1. 实验创建和运行示例
def experiment_management_demo():
    """演示实验管理功能"""
    print("=== 实验管理演示 ===")
    
    # 创建 Opik 客户端
    client = opik.Opik()
    
    # 创建演示数据集
    dataset = [
        {
            "input": "What is 5 * 5?",
            "reference": "25",
            "metadata": {"type": "math"}
        },
        {
            "input": "Who wrote Hamlet?",
            "reference": "William Shakespeare",
            "metadata": {"type": "literature"}
        }
    ]
    
    # 定义两个不同的LLM任务函数
    def task_v1(input_text: str) -> str:
        """版本1的任务函数"""
        if "5 * 5" in input_text:
            return "25"
        elif "Hamlet" in input_text:
            return "Shakespeare"
        return f"V1: {input_text}"
    
    def task_v2(input_text: str) -> str:
        """版本2的任务函数"""
        if "5 * 5" in input_text:
            return "25"
        elif "Hamlet" in input_text:
            return "William Shakespeare"
        return f"V2: {input_text}"
    
    # 定义评估指标
    class ExactMatchMetric(opik.evaluation.metrics.BaseMetric):
        def score(self, input: str, output: str, reference: str) -> float:
            return 1.0 if output == reference else 0.0
    
    exact_match_metric = ExactMatchMetric()
    
    # 运行第一个实验
    print("运行实验 V1...")
    result1 = opik.evaluate(
        dataset=dataset,
        task=task_v1,
        scoring_metrics=[exact_match_metric],
        experiment_name="experiment-v1"
    )
    
    print(f"实验V1结果: 平均分数 = {result1.avg_scores}")
    
    # 运行第二个实验
    print("运行实验 V2...")
    result2 = opik.evaluate(
        dataset=dataset,
        task=task_v2,
        scoring_metrics=[exact_match_metric],
        experiment_name="experiment-v2"
    )
    
    print(f"实验V2结果: 平均分数 = {result2.avg_scores}")
    
    # 比较两个实验
    print("比较两个实验:")
    if result2.avg_scores[exact_match_metric.name] > result1.avg_scores[exact_match_metric.name]:
        print("实验V2表现优于实验V1")
    else:
        print("实验V1表现优于或等于实验V2")

# 运行实验管理演示
if __name__ == "__main__":
    experiment_management_demo()
```

### 3.4 模拟功能

模拟功能用于模拟用户与LLM系统的交互。

```python
import opik

# 配置 Opik
opik.configure(use_local=True)

# 1. 模拟功能示例
def simulation_demo():
    """演示模拟功能"""
    print("=== 模拟功能演示 ===")
    
    # 定义模拟用户
    simulated_user = opik.SimulatedUser(
        persona="customer",
        # 这里可以定义更复杂的用户行为规则
    )
    
    # 定义LLM任务函数
    def llm_support_agent(input_text: str) -> str:
        """简单的客户支持代理"""
        if "help" in input_text.lower():
            return "How can I assist you today?"
        elif "problem" in input_text.lower():
            return "I'm sorry to hear that. Can you please describe the issue in more detail?"
        elif "thank" in input_text.lower():
            return "You're welcome! Is there anything else I can help you with?"
        else:
            return f"I understand you're saying: {input_text}"
    
    # 运行模拟
    print("运行用户交互模拟...")
    result = opik.run_simulation(
        simulated_user=simulated_user,
        task=llm_support_agent,
        max_interactions=3
    )
    
    print(f"模拟结果: {result}")
    print(f"交互次数: {len(result.interactions)}")
    
    # 打印每次交互
    for i, interaction in enumerate(result.interactions):
        print(f"\n交互 {i+1}:")
        print(f"  用户: {interaction.user_input}")
        print(f"  代理: {interaction.agent_output}")

# 运行模拟演示
if __name__ == "__main__":
    simulation_demo()
```

## 4. 高级功能

### 4.1 本地记录追踪数据

```python
import opik

# 配置 Opik
opik.configure(use_local=True)

@opik.track
def my_function():
    return "Hello, World!"

# 本地记录追踪数据
with opik.record_traces_locally():
    my_function()
```

### 4.2 多模态支持

```python
import opik

# 配置 Opik
opik.configure(use_local=True)

# 多模态内容处理示例
def multimodal_demo():
    """演示多模态支持"""
    print("=== 多模态支持演示 ===")
    
    # 示例：处理图像内容（概念演示）
    def multimodal_llm_task(input_data):
        """处理多模态输入的LLM任务"""
        if isinstance(input_data, dict) and "image" in input_data:
            return f"Processing image: {input_data['image']}"
        elif isinstance(input_data, str):
            return f"Processing text: {input_data}"
        else:
            return f"Processing unknown type: {type(input_data).__name__}"
    
    # 创建多模态数据集
    multimodal_dataset = [
        {
            "input": {"image": "cat.jpg", "text": "Describe this image."},
            "reference": "This is a photo of a cat.",
            "metadata": {"type": "image-text"}
        },
        {
            "input": "What is the capital of Germany?",
            "reference": "Berlin",
            "metadata": {"type": "text-only"}
        }
    ]
    
    # 简单的评估指标
    class SimpleMetric(opik.evaluation.metrics.BaseMetric):
        def score(self, input: any, output: str, reference: str) -> float:
            return 1.0 if reference in output else 0.0
    
    # 执行多模态评估
    result = opik.evaluate(
        dataset=multimodal_dataset,
        task=multimodal_llm_task,
        scoring_metrics=[SimpleMetric()],
        experiment_name="multimodal-experiment"
    )
    
    print(f"多模态评估结果: {result.avg_scores}")

if __name__ == "__main__":
    multimodal_demo()
```

### 4.3 与LangChain集成

```python
import opik
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate

# 配置 Opik
opik.configure(use_local=True)

# LangChain 集成示例
def langchain_integration_demo():
    """演示与LangChain的集成"""
    print("=== LangChain 集成演示 ===")
    
    # 创建 LangChain 提示模板
    prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
    
    # 创建 LangChain LLM
    llm = OpenAI(temperature=0.7)
    
    # 定义使用 LangChain 的任务函数
    @opik.track
    def langchain_task(topic: str) -> str:
        """使用LangChain的任务函数"""
        chain = prompt | llm
        result = chain.invoke({"topic": topic})
        return result.content
    
    # 运行任务
    result = langchain_task("computers")
    print(f"LangChain 结果: {result}")

if __name__ == "__main__":
    langchain_integration_demo()
```

## 5. 最佳实践

1. **合理使用追踪**：只追踪关键的LLM调用和业务逻辑，避免过度追踪
2. **使用适当的评估指标**：根据任务类型选择合适的评估指标
3. **创建高质量数据集**：确保评估数据集具有代表性和多样性
4. **利用并行评估**：对于大型数据集，使用多线程加速评估
5. **定期监控**：定期检查LLM系统的性能和成本指标
6. **使用动态追踪控制**：在高流量场景下，考虑使用采样策略或动态启用/禁用追踪

## 6. 完整演示脚本

```python
import opik
from opik.evaluation.metrics import BaseMetric

# 配置 Opik（本地模式）
opik.configure(use_local=True)

# 1. 追踪功能演示
@opik.track
def llm_function(user_question: str) -> str:
    """LLM 函数演示"""
    response = f"Response to: {user_question}"
    
    # 添加元数据
    opik.set_tags(["demo", "complete"])
    opik.log_metadata({"question_length": len(user_question)})
    
    return response

# 2. 自定义评估指标
class SimpleAccuracyMetric(BaseMetric):
    """简单的准确率评估指标"""
    
    def score(self, input: str, output: str, reference: str) -> float:
        """计算准确率"""
        return 1.0 if reference.lower() in output.lower() else 0.0

# 3. 演示所有功能
def complete_demo():
    """完整演示所有功能"""
    print("=== Opik SDK 完整功能演示 ===")
    
    # 追踪功能演示
    print("\n1. 追踪功能演示:")
    trace_result = llm_function("What is Opik SDK?")
    print(f"   结果: {trace_result}")
    
    # 评估功能演示
    print("\n2. 评估功能演示:")
    
    # 创建演示数据集
    dataset = [
        {
            "input": "What is 2 + 2?",
            "reference": "4",
            "metadata": {"category": "math"}
        },
        {
            "input": "What is the capital of France?",
            "reference": "Paris",
            "metadata": {"category": "geography"}
        },
        {
            "input": "Who invented the light bulb?",
            "reference": "Thomas Edison",
            "metadata": {"category": "science"}
        }
    ]
    
    # 简单的LLM任务
    def simple_llm_task(input_text: str) -> str:
        """简单的LLM任务模拟"""
        if "2 + 2" in input_text:
            return "The answer is 4."
        elif "capital of France" in input_text:
            return "Paris is the capital of France."
        elif "light bulb" in input_text:
            return "Thomas Edison invented the light bulb."
        else:
            return f"I don't know the answer to: {input_text}"
    
    # 执行评估
    accuracy_metric = SimpleAccuracyMetric()
    evaluation_result = opik.evaluate(
        dataset=dataset,
        task=simple_llm_task,
        scoring_metrics=[accuracy_metric],
        experiment_name="complete-demo-experiment",
        task_threads=2
    )
    
    print(f"   评估结果: {evaluation_result.avg_scores}")
    print(f"   每个样本分数: {evaluation_result.scores}")
    
    # 实验管理演示
    print("\n3. 实验管理演示:")
    
    # 运行第二个版本的实验
    def improved_llm_task(input_text: str) -> str:
        """改进版的LLM任务"""
        if "2 + 2" in input_text:
            return "4"
        elif "capital of France" in input_text:
            return "Paris"
        elif "light bulb" in input_text:
            return "Thomas Edison"
        else:
            return f"Unknown: {input_text}"
    
    improved_result = opik.evaluate(
        dataset=dataset,
        task=improved_llm_task,
        scoring_metrics=[accuracy_metric],
        experiment_name="improved-demo-experiment",
        task_threads=2
    )
    
    print(f"   改进版实验结果: {improved_result.avg_scores}")
    
    # 比较两个实验
    if improved_result.avg_scores[accuracy_metric.name] > evaluation_result.avg_scores[accuracy_metric.name]:
        print("   改进版实验表现更好!")
    else:
        print("   原始实验表现更好!")
    
    print("\n=== 演示完成 ===")

# 运行完整演示
if __name__ == "__main__":
    complete_demo()
```

## 7. 运行演示

要运行上述演示，您可以将代码保存到 Python 文件中，然后执行：

```bash
python opik_demo.py
```

## 8. 集成支持

Opik SDK 提供了与多种 LLM 框架和工具的集成：

- **LiteLLM**：支持通过环境变量自动配置
- **LangChain**：提供专用集成
- **SageMaker**：提供认证支持
- **pytest**：提供 `@opik.llm_unit` 装饰器用于 LLM 单元测试

## 9. 总结

Opik SDK 为 LLM 系统开发提供了全面的工具链，从开发阶段的追踪和调试，到评估阶段的性能测试，再到生产阶段的监控和优化，都提供了强大的支持。通过合理使用 Opik SDK，可以构建更可靠、更高效、更经济的 LLM 系统。

要了解更多信息，请访问 [Opik 官方文档](https://docs.opik.com/)。