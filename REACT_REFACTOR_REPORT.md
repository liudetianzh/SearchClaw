# ReAct 核心抽离与仓库修改报告

## 1. 修改背景

原先 `src/core/loop.py` 同时承担了两类职责：一方面驱动搜索代理的完整业务流程，另一方面实现 ReAct（Reason / Act / Observe）机制本身。这样会让主循环包含大量模型事件解析、助手消息组装、工具校验、并发执行和异常重试代码，阅读和复用成本较高。

本次修改按照“把核心的 react 抽出来”的目标，将通用 ReAct 能力提取到 `src/core/react.py`，让 `loop.py` 回到业务编排层。

## 2. 文件变更

### 新增 `src/core/react.py`

新增一个最小、独立的 ReAct 实现，主要包含：

- `ReActTurn`：记录一次 Reason 阶段的结果，包括模型产生的工具调用和是否发生 LLM 错误。
- `ReActContext`：封装工具执行所需的运行时上下文，例如会话 ID、缓存目录、限流器和允许访问的本地目录。
- `ReActEngine`：提供 `reason()` 和 `act()` 两个核心阶段，并负责相关辅助逻辑。

### 修改 `src/core/loop.py`

- 导入并初始化 `ReActEngine`。
- 用 `react.reason(...)` 替换原本直接调用 LLM、累积流事件和组装 assistant message 的大段代码。
- 用 `react.act(...)` 替换原有 `_execute_tools()` 与 `_execute_single_tool()`。
- 删除已迁移的工具执行函数及 `_extract_research_query()`。
- 保留搜索代理特有的策略：最大轮次、搜索/抓取次数限制、上下文压缩、停止钩子、研究计划提醒、交互式 `ask_user`、引用收集、事件发送和最终答案生成。

## 3. `react.py` 代码解读

### 3.1 `ReActTurn`

```python
@dataclass
class ReActTurn:
    tool_calls: list[dict] = field(default_factory=list)
    had_error: bool = False
```

它是一次模型响应的轻量结果对象。`reason()` 在收到 `TOOL_USE` 事件时追加工具调用；如果收到错误事件或流式调用抛出异常，则设置 `had_error=True`。主循环据此跳过停止钩子并结束当前会话，避免“API 出错后又被质量钩子推动重试”的循环。

### 3.2 `ReActContext`

这是不可变的工具运行参数容器。`cache_dir` 用于保存超长工具结果，`rate_limiter` 用于复用限流策略，`allowed_roots` 将 CLI 用户显式授权的本地目录传给文件类工具。把这些参数集中起来，可以避免 `ReActEngine` 依赖完整的 `QueryParams`，从而保持模块边界清晰。

### 3.3 `ReActEngine.reason()`：Reason 阶段

该方法执行一次 LLM 流式调用：

1. 将 `LoopState.messages` 转成 API 消息格式。
2. 传入系统提示词、工具 schema、最大 token 数和 session ID。
3. 原样 `yield` 每个 `StreamEvent`，因此上层 UI 仍能实时显示文本、思考过程和工具调用。
4. 同时在内部收集文本片段、reasoning 片段、结构化 reasoning blocks 和工具调用。
5. 调用 `_record_assistant_message()` 将本轮 assistant 内容写回状态。

错误处理分为两类：LLM 返回 `ERROR` 事件时记录并停止收集；底层流抛异常时生成统一的 `ERROR` 事件。两种情况都会设置 `ReActTurn.had_error`，且不会把不完整的 assistant 响应写入历史。

### 3.4 `_record_assistant_message()`：保持对话可回放

当模型产生工具调用时，方法按以下顺序构造 assistant 消息：

1. 结构化或纯文本 reasoning block；
2. 可见文本（如果有）；
3. 一个或多个 `tool_use` block。

其中 Anthropic thinking 模式的 `signature` 会被保留，DeepSeek 等模型的 reasoning 文本也会保留，保证下一轮 API 请求能够继续使用思考链。若本轮只有最终文本，则只保存文本，不把已经结束的 reasoning 再带入后续历史。

### 3.5 `ReActEngine.act()`：Act 阶段

`act()` 先创建 `ToolUseContext`，再根据注册表提供的并发安全名单拆分工具调用：

- 标记为并发安全的工具使用 `asyncio.gather()` 并行执行；
- 其他工具按原顺序串行执行；
- 通过预分配结果列表和原始索引，保证返回结果与模型工具调用顺序一致。

这里仅处理“如何执行动作”，不处理“是否允许动作”。搜索次数和抓取次数上限仍由 `loop.py` 过滤；被过滤的调用会在那里生成“达到上限”的伪工具结果，以维持对话结构。

### 3.6 `_execute_one()`：单个工具的可靠执行

单个调用依次经过：

1. 按名称从 `ToolRegistry` 查找工具；找不到时返回可读的错误及可用工具列表。
2. 调用 `validate_input()` 校验参数；失败时把校验信息和实际 JSON 参数反馈给模型。
3. 调用工具的异步 `call()` 方法。
4. 遇到包含 `429` 的异常时按 `1s、2s、4s` 退避，最多重试 3 次。
5. 其他异常或重试耗尽时记录日志，并返回服务暂不可用的 `ToolResult`，避免单个工具故障击穿整个代理循环。

### 3.7 `_research_query()`

该辅助方法从消息历史中提取第一条未标记的真实用户消息，作为 `research_query` 放入工具上下文，供需要原始研究问题的工具使用；系统注入的 `plan_nudge` 等内部消息会被跳过。

## 4. 修改后的整体调用流程

```text
query_loop
  ├─ 业务 guard / context compact / tool limit
  ├─ ReActEngine.reason
  │    ├─ LLM stream
  │    ├─ 转发 StreamEvent
  │    └─ 回写 assistant message
  ├─ 无 tool_calls？执行 stop hooks 或结束
  ├─ loop.py 按业务限额过滤 tool calls
  ├─ ReActEngine.act
  │    ├─ 参数校验
  │    ├─ 并发或串行执行
  │    └─ 429 重试与错误封装
  ├─ loop.py 注入 tool messages、引用和 UI 事件
  └─ 继续下一轮或生成 final answer
```

这种分层使 ReAct 核心只依赖状态、LLM 客户端和工具注册表，而 SearchClaw 的研究流程仍由 `loop.py` 统一控制。

## 5. 重构收益与注意事项

### 收益

- **职责更清晰**：模型交互和工具执行与搜索产品策略分离。
- **代码更易读**：`loop.py` 删除约 260 行底层执行细节，主流程更接近业务叙述。
- **更易复用和测试**：`ReActEngine` 可以在不启动完整 Web/CLI harness 的情况下单独测试。
- **行为保持一致**：流式事件、thinking block round-trip、工具并发、参数校验和 429 重试均被迁移保留。

### 边界

- `ReActEngine` 不负责搜索/抓取配额、停止钩子、上下文压缩、研究计划或最终答案策略。
- `act()` 返回结果的顺序与输入调用顺序一致，但结果如何发送给用户、写入会话历史和累计引用，仍由 `loop.py` 完成。
- LLM 错误不会写入不完整 assistant 消息；上层循环会跳过 stop hooks 并结束，避免错误重试死循环。

## 6. 验证

仓库已有针对抽离模块的测试入口 `tests/test_react.py`，覆盖 `reason()` 的事件处理、assistant 历史回写以及 `act()` 的工具执行路径。后续运行项目测试时可重点确认：并发工具顺序、429 重试、Anthropic reasoning signature 回传，以及 `query_loop` 的限额和交互工具流程。

