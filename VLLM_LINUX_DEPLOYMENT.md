# 在 Linux 部署 Qwen 7B + vLLM

本目录中的脚本用于将 SearchClaw 接到本地 Qwen 模型。它们必须在
**Linux + NVIDIA GPU + 已安装 CUDA 驱动**的机器上运行；当前 Apple Silicon
Mac 不能运行 vLLM。

## 1. 搬运到 Linux

将整个项目复制到 Linux，或至少复制 `scripts/start_vllm_qwen.sh`、
`scripts/check_vllm_endpoint.sh` 与本说明。不要复制 macOS 创建的 Python
虚拟环境 `.venv`；Linux 上重新创建环境。

```bash
git clone <your-repository-url> SearchClaw
cd SearchClaw
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
pip install vllm
chmod +x scripts/start_vllm_qwen.sh scripts/check_vllm_endpoint.sh
```

先确认 CUDA 正常：

```bash
nvidia-smi
```

## 2. 启动 Qwen 7B

默认模型为 `Qwen/Qwen2.5-7B-Instruct`。首次执行时，vLLM/Hugging Face 会把
权重下载到 Linux 的 Hugging Face 缓存；需要联网并预留约 15 GB 磁盘空间。

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/start_vllm_qwen.sh
```

默认监听 `127.0.0.1:8001`。如需局域网访问，显式绑定私网地址或 `0.0.0.0`，
并通过防火墙限制访问：

```bash
CUDA_VISIBLE_DEVICES=0 HOST=0.0.0.0 PORT=8001 ./scripts/start_vllm_qwen.sh
```

若模型已在服务器本地，可将 `MODEL` 改为绝对路径，避免重新下载：

```bash
CUDA_VISIBLE_DEVICES=0 MODEL=/data/models/Qwen2.5-7B-Instruct \
  ./scripts/start_vllm_qwen.sh
```

`Qwen2.5-7B-Instruct` 在 40 GB 显存的单张 A40 上可运行。`MAX_MODEL_LEN=8192`
为起始设置；显存紧张时调低它，例如 `MAX_MODEL_LEN=4096`。

## 3. 验证服务

另开一个终端执行：

```bash
./scripts/check_vllm_endpoint.sh
```

它依次请求 `/v1/models` 和 `/v1/chat/completions`。两者返回 JSON 且第二个
响应含 `vLLM is ready`，才说明模型服务可用。

## 4. 接入 SearchClaw

在 `config/settings.yaml` 的 `llm` 段中设置：

```yaml
llm:
  default_model: "openai/Qwen2.5-7B-Instruct"
  side_query_model: "openai/Qwen2.5-7B-Instruct"
  fallback_model: "openai/Qwen2.5-7B-Instruct"
  base_url: "http://127.0.0.1:8001/v1"
  side_query_base_url: "http://127.0.0.1:8001/v1"
  max_tokens: 2048
```

如果 SearchClaw 和 vLLM 不在同一台机器，将 `127.0.0.1` 替换为 vLLM 服务器
可达的私网 IP，并确保端口仅向必要的机器开放。

## 5. 部署原理

`vllm serve` 读取 Hugging Face 模型权重，将其加载到 CUDA GPU，并启动一个
OpenAI Chat Completions 兼容 HTTP 服务。`--served-model-name` 指定 API 请求中
的 `model` 字段；SearchClaw 的 LiteLLM 客户端以 `openai/` 前缀和 `base_url`
将对话与工具 schema 发送给该服务。vLLM 返回工具调用或文本，SearchClaw 的
ReAct 循环再执行本地检索工具并将 Observation 传回模型。
