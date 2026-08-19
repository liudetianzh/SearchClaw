# ReAct 框架学习笔记

## 1. 目标与边界

本项目将 ReAct 拆成两层：`src/core/react.py` 实现通用的
**Reason - Act** 机制；`src/core/loop.py` 负责把工具结果写回对话历史，
形成 **Observation**，并处理搜索配额、停止钩子、上下文压缩和 UI 事件。

因此，ReAct 不是单个函数，而是下列闭环：模型决定动作，框架执行动作，
执行结果作为下一轮模型输入。

```text
用户问题 -> Reason -> tool_use -> Act -> ToolResult -> Observation -> Reason -> ... -> 最终答案
```

## 2. 一轮调用链

```text
query_loop (loop.py)
  1. 创建 LoopState，并加入 user message
  2. ReActEngine.reason(...)
       LLMClient.stream(...) 生成 TEXT / TOOL_USE / ERROR 事件
       收集 tool_calls，并把 assistant tool_use 写入 LoopState
  3. loop.py 根据配额过滤调用
  4. ReActEngine.act(...)
       校验输入，按工具并发安全性执行 Tool.call()
       返回 ToolResult
  5. loop.py 为每个 ToolResult 写入 role="tool" 的 Message
       这条 Message 就是 Observation；下一次 reason 会将它传回模型
  6. 无工具调用时，模型文本成为最终答案（或先经停止钩子校验）
```

## 3. 核心模块与关键实现原因

| 模块 | 职责 | 关键原因 |
| --- | --- | --- |
| `ReActTurn` | 保存一轮模型请求产生的 `tool_calls` 和错误状态 | 将流式 LLM 事件转换为主循环可判断的结果。 |
| `ReActContext` | 传入 session、缓存目录、限流器和可访问目录 | 不让 ReActEngine 依赖完整 Web/CLI 参数对象。 |
| `ReActEngine.reason()` | 调用 LLM、转发流事件、收集文本/推理/工具调用并回写 assistant 消息 | assistant 的 `tool_use_id` 必须进入历史，下一轮的 tool result 才能和调用匹配。 |
| `_record_assistant_message()` | 将文本、推理块和工具调用按 API 所需结构组装 | 保留 reasoning signature，兼容带思维链状态的模型。 |
| `ReActEngine.act()` | 创建 `ToolUseContext`，并发或串行执行工具，保持返回顺序 | 搜索可并发，但有副作用或顺序要求的工具不能随意并发。 |
| `_execute_one()` | 查找工具、校验参数、处理 429 重试、封装普通错误 | 让错误以 ToolResult 反馈给模型，而不是使整个会话崩溃。 |
| `query_loop()` | 配额控制与 Observation 回写 | 配额是产品策略，不属于通用 ReAct；Observation 必须由此处添加为 `role="tool"`。 |

## 4. 核心伪代码

```python
state.messages.append(user_question)

while state.turn_count < max_turns:
    turn = ReActTurn()

    # Reason: 模型从历史中决定回答还是调用工具
    for event in llm.stream(messages=state.messages, tools=registry.schemas):
        emit_to_ui(event)
        if event.type == TOOL_USE:
            turn.tool_calls.append(event.data)
    state.messages.append(assistant_message_with_tool_calls_or_text)

    if turn.had_error:
        break
    if not turn.tool_calls:
        if quality_gate_accepts(state):
            break
        state.messages.append(quality_feedback)
        continue

    # Act: 先执行可用且未超过配额的动作
    calls = filter_by_limits(turn.tool_calls)
    results = await execute_tools(calls)

    # Observation: 工具输出成为下一轮 Reason 的证据
    for call, result in zip(calls, results):
        state.messages.append(Message(
            role="tool",
            content=result.data,
            metadata={"tool_call_id": call["tool_use_id"], "tool_name": call["tool_name"]},
        ))
```

## 5. 搜索与读取工具的输入输出

### 旧的外部工具

- `search_web(query, num_results)`：通过 Serper/浏览器/搜索页面获取标题、URL、摘要。
- `fetch_url(url)`：优先经 Jina Reader，失败时直接抓取网页并转为 Markdown。

这两个工具依赖外网服务，且语料会随互联网变化，不适合低成本、可复现的离线实验。

### 本地语料工具

启用 `tools.local_retrieval.enabled: true` 后，注册表只保留：

- `search_local(query, top_k)`：`POST /search`，默认请求体为
  `{"query": "...", "top_k": 5}`；结果必须是列表，或放在
  `results`、`data`、`documents`、`hits` 中的列表。每条结果至少应有稳定
  `id`/`doc_id`，并建议提供 `title` 和短 `content`/`text`。
- `read_local_document(document_id)`：`POST /document`，默认请求体为
  `{"document_id": "..."}`；返回一篇带 `content`/`text` 的文档。

实现兼容 `id`、`doc_id`、`document_id` 等标识字段，和 `content`、`text`、
`contents`、`body`、`snippet`、`passage` 等文本字段。工具会将结果转换为
`ToolResult(data=...)`，因此与 ReAct 的统一工具接口兼容。

**正文策略：** 搜索只返回摘要和 `document_id`；正文由单独的
`read_local_document` 提供。这样更接近“搜索 - 读页”工作流，也避免首次检索
把大量文本注入上下文。若检索器只有单一搜索端点，可让它在结果中返回完整
`content`；该内容会作为摘要被 ReAct 消费，但仍建议提供读取端点以支持长文。

## 6. 已验证闭环

`tests/test_local_react_loop.py` 使用 `httpx.MockTransport` 模拟一个本地
`/search` 服务。假模型第一轮调用 `search_local`，第二轮显式断言收到
`role="tool"` 的 Observation（包含“Paris is the capital of France.”），随后
输出答案。该测试证明：本地检索响应能被工具消费，并且标准 ReAct 可以完成
一轮“搜索 - Observation - 回答”。

真实 FlashRAG 或 Wikipedia 服务接入时，只需在 `config/settings.yaml` 填写其
地址、路径和字段名；应额外用一条实际请求确认服务契约。
