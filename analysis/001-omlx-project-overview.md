# 001 · oMLX 项目全维度深度分析

> **文档编号**：`analysis/001-omlx-project-overview.md`
> **项目**：oMLX —— "LLM inference, optimized for your Mac"
> **定位**：面向 Apple Silicon (M1/M2/M3/M4) 的生产级本地 LLM 推理服务器
> **技术栈**：Python 3.11+ · MLX · mlx-lm · mlx-vlm · FastAPI · Uvicorn · Swift/SwiftUI
> **License**：Apache 2.0
> **分析维度**：需求 (Requirements) · 功能 (Features) · 架构 (Architecture) · 业务 (Business) · 原理 (Principles)

> 📁 本文档属于 [analysis/](../analysis/) 目录。编号管理规范请参见 [analysis/README.md](./README.md)。

---

## 目录

- [一、需求分析](#一需求分析-requirements)
- [二、功能分析](#二功能分析-features)
- [三、架构分析](#三架构分析-architecture)
- [四、业务分析](#四业务分析-business)
- [五、原理分析](#五原理分析-principles--底层实现)
- [六、关键文件清单](#六关键文件清单)
- [七、总结](#七总结)

---

## 一、需求分析 (Requirements)

### 1.1 核心用户痛点

从 README 的作者自述可以提取核心动机：

> *"Every LLM server I tried made me choose between convenience and control. I wanted to pin everyday models in memory, auto-swap heavier ones on demand, set context limits - and manage it all from a menu bar."*

> *"oMLX persists KV cache across a hot in-memory tier and cold SSD tier - even when context changes mid-conversation, all past context stays cached and reusable across requests, making local LLMs practical for real coding work with tools like Claude Code."*

### 1.2 显式需求

| 类别 | 需求 |
|------|------|
| **平台** | 仅支持 macOS 15.0+ (Sequoia) 和 Apple Silicon (M1-M4)，利用 MLX 框架的 Metal GPU 加速 |
| **多模型** | LLM、VLM、OCR、Embedding、Reranker 在同一进程内并行服务 |
| **大上下文** | 解决本地 LLM 在长上下文（>128K token）下的实用性，对标 Claude Code 等编码 agent |
| **多模型同驻** | 通过 LRU 淘汰、Pin、TTL 等策略在统一内存中协调多个模型 |
| **API 兼容** | OpenAI Chat/Completion + Anthropic Messages + Embedding + Rerank + Responses API |
| **KV 缓存** | 热冷分层（前缀共享 + Copy-on-Write），重启后 KV 仍可用 |
| **持续批处理** | 通过 mlx-lm BatchGenerator 实现 vLLM 式连续批处理 |
| **本地优先** | Admin UI 完全离线（CDN 依赖全部 vendored），不需要外网 |
| **可观测性** | Web admin 面板（含中文/英文/韩文/日文/法文/俄文/西班牙文/葡语）+ macOS 菜单栏 |
| **Claude Code 集成** | 一键配置 OpenClaw、OpenCode、Codex、Hermes Agent、Copilot、Pi |
| **自动更新** | macOS App 通过 Sparkle 框架自动更新 |

### 1.3 隐式需求（从代码细节推断）

| 隐式需求 | 代码证据 |
|---------|----------|
| **避免 OOM** | `ProcessMemoryEnforcer` + 三档 memory_guard_tier (safe/balanced/aggressive/custom) |
| **防止 MLX Metal 命令缓冲竞争** | issue #85 提及 → 全局单线程 `_global_mlx_executor` |
| **防止 trust_remote_code 远程代码执行** | issue #926 → 默认关闭 `trust_remote_code` |
| **解决 WhisperProcessor "Processor not found"** | pmarreck/omlx#1 → 强制 `mistral-common>=1.10` |
| **Anthropic SDK 兼容性** | 同时支持 `Authorization: Bearer` 和 `x-api-key` header |
| **多模型互不阻塞** | 每 EngineCore 独立线程 + mx.Stream（issue #1248） |
| **Worker 线程崩溃防护** | `_immortal_mlx_executors` 永不关闭，避免 @mx.compile 在线程局部析构崩溃 |
| **菜单栏 App 与 CLI 双向通信** | Unix Domain Socket (`AppControlServer`) |

### 1.4 部署形态需求

| 形态 | 说明 |
|------|------|
| **macOS App** | `.dmg` 拖到 Applications 即可，含 in-app 自动更新 |
| **Homebrew** | `brew install omlx`，可作为 `brew services` 后台服务运行 |
| **从源码** | `pip install -e .` 或 `pip install -e ".[mcp]"` |
| **venvstacks 分层打包** | macOS App 内嵌 Python 解释器，零外部依赖 |

---

## 二、功能分析 (Features)

### 2.1 功能矩阵

| 功能层 | 子功能 | 入口文件 |
|--------|--------|----------|
| **HTTP API** | OpenAI `/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`、`/v1/rerank`、`/v1/models`；Anthropic `/v1/messages`、`/v1/messages/count_tokens`；Responses `/v1/responses`、`/v1/responses/{id}` | `omlx/server.py` |
| **Admin UI** | 实时监控、模型管理、Chat、Benchmark、Per-model settings、Downloader、Integrations 一键配置、i18n 8 种语言 | `omlx/admin/routes.py` + `dashboard.js` (5217 行 Alpine.js) |
| **macOS App** | SwiftUI 菜单栏 + Welcome + Sparkle 自动更新 + App Control Socket (与 CLI 双向通信) | `apps/omlx-mac/Sources/` |
| **CLI** | `omlx serve` / `start` / `stop` / `restart` / `diagnose` / `launch codex` | `omlx/cli.py` |
| **多模型引擎** | LLM (BatchedEngine) / VLM (VLMBatchedEngine) / Embedding / Reranker / STT / TTS / STS / DFlash | `omlx/engine/*.py` |
| **KV 缓存** | 热层 (RAM) + 冷层 (SSD safetensors) + 前缀共享 + CoW + Block 256 token | `omlx/cache/paged_cache.py`, `paged_ssd_cache.py` |
| **调度** | FCFS、连续批处理、Chunked prefill、Decode burst、preflight eviction | `omlx/scheduler.py` (10114 行) |
| **工具调用** | 7+ 种格式：JSON `<tool_call>` / Qwen3.5 XML / Gemma / GLM XML / MiniMax namespaced / Mistral bracket / Kimi K2 / Longcat | `omlx/api/tool_calling.py` |
| **结构化输出** | JSON Schema 校验、Grammar 编译 (xgrammar) | `omlx/api/grammar.py` |
| **MCP** | Model Context Protocol 集成，支持 stdio/SSE 传输 | `omlx/mcp/` + `mcp_routes.py` |
| **oQ 量化** | 通用混合精度量化（按敏感度分级 1-5 档），支持 VLM 视觉塔保护 | `omlx/oq.py` |
| **TurboQuant KV** | 量化 KV cache 进一步降内存 | `omlx/turboquant_kv.py` |
| **DFlash** | Block diffusion 推测解码（Apple Silicon） | `omlx/engine/dflash.py` |
| **HF/ModelScope 下载** | 浏览器内直接搜/下 MLX 模型 | `admin/hf_downloader.py`, `ms_downloader.py` |
| **基准测试** | 一键测 PP/TG tokens/s，支持部分前缀缓存命中场景 | `admin/benchmark.py` |
| **精度基准** | oQ 量化前后精度对比 | `admin/accuracy_benchmark.py` |

### 2.2 功能开关 (Feature Flags)

```python
@dataclass
class EngineConfig:
    decode_burst_max_steps: int = 64            # OMLX_DECODE_BURST_MAX_STEPS
    decode_burst_budget_single_s: float = 0.1   # 单请求时激进 burst
    decode_burst_budget_s: float = 0.03          # 并发时紧凑 burst
```

```python
@dataclass
class ProcessMemoryEnforcer:
    memory_guard_tier: str = "balanced"          # safe / balanced / aggressive / custom
    soft_threshold: float = 0.90                  # 软阈值：LRU 驱逐 + 暂停 admission
    hard_threshold: float = 0.95                  # 硬阈值：abort in-flight + loading
    prefill_safe_zone_ratio: float = 0.89         # prefill chunk 自适应 shrink
    prefill_min_chunk_tokens: int = 256
```

### 2.3 Per-Model Settings

- **Sampling 参数**：temperature、top_p、top_k、max_tokens、repetition_penalty、seed
- **Chat template kwargs**：`enable_thinking`、`preserve_thinking` 等 Jinja 变量
- **TTL**：每个模型独立的空闲超时
- **Model alias**：自定义 API 可见名
- **Model type override**：手动覆盖 LLM/VLM
- **Profiles**：保存命名设置包，可作为独立模型暴露（`qwen3-8b:thinking`）

---

## 三、架构分析 (Architecture)

### 3.1 总体分层架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    macOS App (SwiftUI)                          │
│  MenubarController / ServerProcess / AppControlServer          │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Unix Socket (AppControl)
┌──────────────────────▼──────────────────────────────────────────┐
│              CLI (omlx start/stop/restart)                      │
│  cli.py / lifecycle_command                                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │ spawn: python -m omlx.cli serve
┌──────────────────────▼──────────────────────────────────────────┐
│                  Uvicorn ASGI / FastAPI                         │
│                       server.py                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Middleware: API Key 验证 / 限流 / Request 验证         │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐     │
│  │ /v1/chat/...   │ │ /v1/messages   │ │ /v1/embeddings │     │
│  │ /v1/responses  │ │ /v1/rerank     │ │ /admin/...     │     │
│  └────────────────┘ └────────────────┘ └────────────────┘     │
└──────────┬───────────────────────────────────────────────────────┘
           │ lease acquire / release
┌──────────▼───────────────────────────────────────────────────────┐
│                   EnginePool (multi-model)                       │
│   - LRU eviction + Pin + TTL + pre-load memory check             │
│   - AsyncEngineCore per model                                    │
│   - lease 机制 (in_use count) 防边用边驱逐                        │
└──────────┬───────────────────────────────────────────────────────┘
           │ delegate
┌──────────▼───────────────────────────────────────────────────────┐
│             AsyncEngineCore / EngineCore (per model)            │
│   - Per-engine MLX executor + mx.Stream                          │
│   - decode_burst 优化 (避免 GIL ping-pong)                        │
│   - Wake event 唤醒 idle loop                                     │
└──────────┬───────────────────────────────────────────────────────┘
           │ step()
┌──────────▼───────────────────────────────────────────────────────┐
│                       Scheduler                                  │
│   - Waiting/Running set, FCFS                                    │
│   - mlx-lm BatchGenerator (prefill + decode 合并)                │
│   - Preflight eviction (prefill 前触发 LRU 卸载)                  │
│   - Continuous batching                                          │
└──────┬──────────────────────────────────────────────────┬────────┘
       │                                                  │
┌──────▼────────────────────┐                ┌──────────▼────────┐
│  Cache Stack             │                │  Memory Enforcer  │
│  ┌──────────────────────┐│                │ ProcessMemory-    │
│  │ PagedCacheManager    ││                │ Enforcer (后台poll)│
│  │ (block 256 token)    ││                │  - 3档 tier       │
│  │ - LRU FreeBlockQueue ││                │  - soft/hard 阈值 │
│  │ - CoW + prefix share ││                │  - TTL eviction   │
│  └──────────┬───────────┘│                │  - prefill abort  │
│             │             │                └───────────────────┘
│  ┌──────────▼───────────┐│
│  │ HotCache (RAM)       ││  ← write-back, SharedHotCacheBudget
│  │ - safetensors 内存   ││
│  └──────────┬───────────┘│
│             │ demote      │
│  ┌──────────▼───────────┐│
│  │ PagedSSDCache        ││  ← 异步写入队列, block-level safetensors
│  │ (冷层)              ││  ← hash-based subdir 结构
│  └──────────────────────┘│
└──────────────────────────┘
```

### 3.2 关键架构决策

#### (a) MLX Stream 隔离

```python
# 每个 EngineCore 独立线程 + mx.Stream
self._mlx_stream = mx.new_thread_local_stream(mx.default_device())
self._mlx_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix=f"mlx-engine-{engine_id[:8]}")
```

**原因**：MLX Metal command buffer 在多线程下会段错误（issue #85）。但单线程吞吐受限，所以不同模型各自独占线程 → 多模型可并行。

#### (b) Decode Burst 优化

```python
def _step_burst(self) -> list:
    """在一次 executor hand-off 内连续 step 多次"""
    outputs = [self.scheduler.step()]
    # ...持续 burst 直到无工作/prefill eviction/预算耗尽
```

**原因**：每个 decode token 反复 asyncio ↔ executor 切换会产生 ~1ms GIL 竞争。burst 让 MLX 线程连续持有 GIL，实测从 74 → 80 tok/s。

**自适应预算**：
- 单请求时：`decode_burst_budget_single_s = 0.1s`（激进）
- 并发时：`decode_burst_budget_s = 0.03s`（紧凑）
- `max_steps = 64` 作为 host-side 列表长度的硬上限

#### (c) Lease 机制防驱逐中请求

```python
@dataclass
class EngineEntry:
    in_use: int = 0  # in-flight acquire/use lease count; never evict while > 0
    abort_requested: bool = False  # Set under hard pressure for leased requests
    pending_unload_reason: str | None = None  # Unload as soon as leases/activity drain
```

→ 即使处于 hard pressure，也必须等活跃请求完成才能卸载。

#### (d) Cache Block 链式哈希

```python
def compute_block_hash(parent_hash, token_ids, extra_keys, model_name):
    hasher = hashlib.sha256()
    hasher.update(model_name.encode())  # 模型隔离
    hasher.update(parent_hash or b"omlx-root")  # 链式
    hasher.update(bytes(str(tuple(token_ids)), "utf-8"))
    hasher.update(bytes(str(extra_keys), "utf-8"))
    return BlockHash(hasher.digest())
```

→ SHA-256 链式哈希，让相同前缀的请求能立即复用 KV。

#### (e) 双层 Block Queue (O(1) LRU)

```python
class FreeKVCacheBlockQueue:
    fake_head ──► [B3] ⇄ [B1] ⇄ [B5] ⇄ [B2] ⇄ [B7] ⇄ [B4] ⇄ fake_tail
                  LRU                                  MRU
```

- `popleft()`: 取 B3 用于分配（evict 候选）
- `remove(B5)`: 前缀命中时从中间取出（避免淘汰热块）
- `append(block)`: 释放时放回 MRU 端
- 全部 O(1)，借鉴 vLLM v1 的 `vllm/v1/core/block_pool.py`

#### (f) Module-Level Patch 而非 Fork

DeepSeek V4 / MiniMax M3 / Step 3.7 等新模型通过 `sys.modules` 注入而非修改 `mlx-lm` 源码：

```
omlx/patches/
├── deepseek_v4/        # 1:1 复制 PR 1192 源码
├── glm_moe_dsa/        # GLM MoE + DSA 注意力
├── step3p7/            # Step 3.7 模型
├── minimax_m3_sparse_attention.py
├── llama4_attention.py
├── qwen3_6_nested_visual.py
└── ...
```

**避免对 pinned commit 的分叉**。完整说明在 `omlx/patches/deepseek_v4/README.md`。

#### (g) Engine 抽象层次

```python
class BaseEngine(ABC):                # 抽象基类
    async def start(): ...
    async def stop(): ...
    async def generate(): ...
    async def stream_generate(): ...
    async def chat(): ...
    async def stream_chat(): ...

class BatchedEngine(BaseEngine):      # 通用 LLM
class VLMBatchedEngine(BaseEngine):   # VLM + Boundary Snapshot
class EmbeddingEngine(BaseNonStreamingEngine):
class RerankerEngine(BaseNonStreamingEngine):
class DFlashEngine(BaseEngine):       # Block Diffusion 推测解码
class STTEngine / TTSEngine / STSEngine:  # 语音
```

### 3.3 数据流 (Chat Completion 请求全链路)

```
HTTP POST /v1/chat/completions
  │
  ├─ verify_api_key (Bearer / x-api-key)
  ├─ _preprocess_markitdown_files_for_llm  (附件→文本)
  ├─ EnginePool.get_engine_for_model()       → LRU 淘汰 / 加载 / lease 申请
  │     └─ ProcessMemoryEnforcer.preflight()  → 检查 prefill 内存预算
  ├─ get_model_settings_for_request()         → Per-model settings (alias/profile)
  ├─ ChatCompletionRequest → InternalRequest (adapter)
  ├─ BatchedEngine.stream_chat()
  │     └─ AsyncEngineCore.generate()
  │           └─ Scheduler.add_request()      → waiting queue
  │           └─ Engine loop → scheduler.step()
  │                 ├─ Preflight eviction callback (if needed)
  │                 ├─ BatchGenerator prefill (chunked if needed)
  │                 ├─ BatchGenerator decode
  │                 ├─ Cache: store hot tier (RAM)
  │                 ├─ Cache: async demote to SSD tier
  │                 └─ Tool call parser (7 种格式之一)
  └─ SSE streaming via uvicorn
       └─ _LLMEngineLease.release() (finally)
```

### 3.4 模块依赖关系

```
server.py (FastAPI)
  ├── engine_pool.py
  │     ├── engine/batched.py    (BatchedEngine)
  │     ├── engine/vlm.py        (VLMBatchedEngine)
  │     ├── engine/embedding.py
  │     ├── engine/reranker.py
  │     └── engine_core.py       (EngineCore / AsyncEngineCore)
  │           └── scheduler.py   (Scheduler / BatchGenerator)
  │                 ├── cache/paged_cache.py    (PagedCacheManager)
  │                 ├── cache/paged_ssd_cache.py (PagedSSDCacheManager)
  │                 ├── cache/prefix_cache.py    (BlockAwarePrefixCache)
  │                 └── cache/hybrid_cache.py    (LayerCacheConfig)
  ├── process_memory_enforcer.py  (后台轮询任务)
  ├── memory_monitor.py
  ├── api/
  │     ├── openai_models.py
  │     ├── anthropic_models.py
  │     ├── anthropic_utils.py
  │     ├── tool_calling.py      (7 种 parser)
  │     ├── grammar.py
  │     ├── mcp_routes.py
  │     └── adapters/             (OpenAI / Anthropic adapter)
  ├── admin/
  │     ├── routes.py            (HTML + JSON API)
  │     ├── hf_downloader.py
  │     ├── benchmark.py
  │     └── static/js/dashboard.js  (5217 行 Alpine.js)
  ├── oq.py / oq_manager.py       (量化)
  ├── turboquant_kv.py
  ├── patches/                    (新模型 monkey-patch)
  └── cli.py                      (CLI 入口)
```

---

## 四、业务分析 (Business)

### 4.1 目标市场

| 客群 | 价值主张 |
|------|----------|
| **独立开发者 / Researcher** | 本地运行 SOTA LLM，无需 API key、无数据外传 |
| **Claude Code / Codex / OpenCode / Hermes Agent 用户** | 通过 "context scaling" 让小模型也能驱动长上下文，自动 compact 触发时机正确 |
| **macOS 工作流重度用户** | 菜单栏 + Sparkle 自动更新 + 离线 admin UI |
| **多模型爱好者** | 同时在内存中持有小模型（钉住）+ 大模型按需 swap |
| **隐私敏感企业 / 个人** | 完全本地部署，敏感数据不出本机 |

### 4.2 商业模式

- **Apache 2.0 开源**（无盈利压力，纯开源驱动）
- **个人副业激励**：README 顶部有 Buy Me A Coffee
- **个人品牌驱动**：作者邮箱 `junkim.dot@gmail.com`，官网 `omlx.ai`
- **无云服务依赖**：用户自托管，作者不赚托管费

### 4.3 竞争格局

| 竞品 | 差异化 |
|------|--------|
| **Ollama** | UI 友好但不支持连续批处理；KV 缓存策略较弱；不支持 Anthropic API |
| **LM Studio** | 图形化强，但不开源核心引擎 |
| **llama.cpp / llama-server** | 跨平台但 Apple Silicon 性能不如 MLX；无 tiered KV cache |
| **vLLM (CUDA)** | 高性能但不支持 Apple Silicon |
| **vllm-mlx** | oMLX 直接 fork 自 vllm-mlx v0.1.0，加入多模型/tiered cache/VLM/admin/菜单栏 |
| **MLX-LM 直接调用** | 无服务端能力 |

### 4.4 业务创新点

1. **Tiered KV Cache 跨重启** —— 解决 "context changes mid-conversation" 的关键创新，是竞品都没做的（`BoundarySnapshotStore`、`PagedSSDCacheManager`）。
2. **Claude Code 优化** —— `context_scaling` 让 8B 模型在 Claude Code 中能跑长上下文（prompt token 计数虚拟放大 → auto-compact 触发时机正确）。
3. **菜单栏一键启停 + CLI 双向控制** —— Unix Domain Socket (`AppControlServer.swift`) 实现 App ↔ CLI 通信。
4. **Per-model Profiles + Alias** —— `qwen3-8b:thinking` 在不加载第二个模型的前提下提供不同 chat_template_kwargs 配置。
5. **oQ 量化引擎** —— 用户可在 admin UI 中可视化地对已加载模型做混合精度二次量化。
6. **DFlash 推测解码** —— 集成 bstnxbt/dflash-mlx，提供 block diffusion 加速。

### 4.5 产品成熟度信号

- **CI/CD**：`.github/workflows/` 多平台测试
- **测试覆盖**：`tests/` 目录包含单元测试 + 慢速测试（`-m "not slow"`）
- **多语言 README**：英/中/韩/日/法
- **详细文档**：`docs/CONTRIBUTING.md`、`docs/oQ_Quantization.md`
- **Homebrew Formula**：可一键安装升级
- **详细补丁同步指南**：`patches/deepseek_v4/README.md` 说明如何跟随上游 PR 更新

---

## 五、原理分析 (Principles / 底层实现)

### 5.1 Paged KV Cache 实现原理

**借鉴 vLLM v1 block pool 设计**：

```
GPU Memory                    SSD (safetensors)
┌───────────────────┐         ┌───────────────────┐
│ CacheBlock        │  demote │ models/<m>/<hash>/│
│  - block_id       │ ──────► │  block_<hash>.safetensors│
│  - ref_count      │ ◄────── │                   │
│  - block_hash     │  restore│                   │
│  - token_count    │         │                   │
└───────────────────┘         └───────────────────┘
       ▲
       │ Hash chain:
       │ parent_hash → token_ids → SHA-256
       ▼
┌───────────────────┐
│ BlockHashToBlockMap │  全局 O(1) 哈希→block 映射
└───────────────────┘
```

**FreeKVCacheBlockQueue**（双链表 LRU）：

```python
fake_head ──► [B3] ⇄ [B1] ⇄ [B5] ⇄ [B2] ⇄ [B7] ⇄ [B4] ⇄ fake_tail
              LRU                                  MRU
```

- `popleft()`: 取 B3 用于分配（evict 候选）
- `remove(B5)`: 前缀命中时从中间取出（避免淘汰热块）
- `append(block)`: 释放时放回 MRU 端

### 5.2 连续批处理 (Continuous Batching) 原理

**核心问题**：传统 static batching 必须等所有请求生成完毕才能前进 → 长请求拖死短请求。

**oMLX 的方案**：

```python
# mlx-lm BatchGenerator 内部状态
GenerationBatch / PromptProcessingBatch
SequenceStateMachine (每个序列: prefill → decode → finished)

# Scheduler.step() 每轮：
1. waiting → running (按 FCFS)
2. prefill_or_raise(num_prompt_tokens)  ← 内存预算检查
3. BatchGenerator.step():
   - 所有 running 序列并行 decode 一个 token
   - 新 prefill 序列加入 batch
4. finished 序列输出 → SSE 发送 → 移除
5. store_cache: hot tier 写回 RAM，demote 到 SSD
```

**Chunked Prefill**（`prefill_step_size=2048`）：
- 大 prompt 切分，避免一个 prefill 占满 GPU → 阻塞 decode
- `chunked_prefill=True` 时每个 step 处理一个 chunk，与 decode 交错

### 5.3 Memory Enforcer 原理

```
每 1s 轮询:
┌────────────────────────────────────────────────────┐
│ _current_usage_bytes                                │
│   = RSS + active MLX + hot_cache + paged_ssd_meta │
│                                                    │
│ _get_dynamic_ceiling                                │
│   = system_available - static_reserve(tier)        │
│   tier: safe (20%) | balanced (50%) | aggressive   │
│                                                    │
│ pressure_level = ok | soft | hard                  │
│   soft ≥ 90%: LRU 驱逐非 pinned；暂停 admission    │
│   hard ≥ 95%: 还要 abort in-flight + abort loading │
└────────────────────────────────────────────────────┘
```

**三层保护**：

1. **Ceiling**（硬上限）：max(动态上限, 静态上限, Metal cap) 取最小
2. **Soft threshold (90%)**：温和动作，不影响在途请求
3. **Hard threshold (95%)**：极端动作，可能 abort prefill

**Prefill safe zone**：

```python
prefill_safe_zone_ratio: float = 0.89  # 低于此 ratio prefill 跑全 chunk
prefill_min_chunk_tokens: int = 256     # 自适应 shrink 下限
```

→ 临近 hard cap 时 prefill chunk 自动缩小，避免一次性占光内存。

### 5.4 MLX/Metal 底层适配

**Thread-Local Stream**（Metal command buffer 隔离）：

```python
def _init_mlx_thread():
    stream = mx.new_thread_local_stream(mx.default_device())
    gen_mod.generation_stream = stream  # 替换 mlx_lm.generate 全局变量
    sched_mod.generation_stream = stream
```

→ 避免 mlx-lm 在主线程创建的 stream 被其他线程 `.item()` 调用时崩溃。

**Compile Cache Clear**（避免 @mx.compile 析构崩溃）：

```python
# ThreadPoolExecutor 关闭时 @mx.compile 的 ~CompilerCache 在 worker 线程释放会崩
# 所以保留 executor 引用，进程生命周期内不复用
_immortal_mlx_executors: list = []  # 永不关闭
```

**Apple Silicon 统一内存特性**：

- `get_max_working_set_bytes()` 通过 `mx.device_info()` 获取 Metal 实际可用 RAM
- `mx.set_wired_limit(bytes)` 调整 iogpu.wired_limit（kernel 级 GPU 内存上限）
- 不需要像 CUDA 那样显式管理 host↔device 传输

### 5.5 Block SSD 序列化原理

```python
# 保存到 SSD
def save_block(block_hash, kv_tensors):
    # mlx 格式：按层保存 keys 和 values
    # 文件结构: <cache_dir>/<model_hash>/<block_hash>/block_<N>.safetensors
    # 元数据: metadata.json (block_size, num_layers, dtype)
    safetensors.save_file(tensors, path, metadata={"format": "mlx-paged"})

# 加载
def load_block(block_hash):
    tensors = safetensors.load_file(path)
    # 反量化 (如果是 TurboQuant 格式则重建 codec)
    if is_turboquant:
        _rebuild_codecs(tq_cache, key_state, value_state)  # 由 (head_dim, bits, seed) 确定性生成
```

**TurboQuant Codec 确定性重建**：因为 codec 仅依赖 (head_dim, bits, seed)，所以可以从 SSD 元数据完全恢复，无需额外存储 codec state，节省 SSD 空间。

**异步写入队列**：

```python
_PENDING_WRITES_TARGET_RAM_FRACTION = 0.10   # 目标：占 host RAM 10%
_PENDING_WRITES_HARD_RAM_FRACTION = 0.30     # 硬上限
_PENDING_WRITES_SOFT_FLOOR = 32              # 软下限（避免零写入导致串行化）
_PENDING_WRITE_PUT_TIMEOUT_SECONDS = 1.0     # 阻塞超时
```

→ 后台 writer thread 持续消费 `pending_writes` 队列，自适应 batch 大小。

### 5.6 VLM 边界快照 (Boundary Snapshot) 原理

**问题**：VLM 的图像 token 嵌入在 prefill 时一次性注入，prefix 缓存的边界不连续。

**解决**：`BoundarySnapshotStore` 在 VLM 的"图像结束位置"保存独立 KV snapshot，下次同图请求可直接复用：

```python
class _BoundarySnapshotProvider:
    """Dict-like loader for extracted boundary snapshots"""
    def __getitem__(self, tc: int) -> Any:
        # 先查 in-memory 已提取快照
        snap = self._in_memory.get(tc)
        if snap is not None:
            return snap
        # 回退到 SSD 按需加载
        if self._store is not None:
            return self._store.load(self._request_id, tc)
```

→ 这让 "chat with same images across requests" 也享受 prefix cache。

### 5.7 Tool Calling 解析原理

每个模型族有自己的 tool call 标记，oMLX 实现 7+ 种 parser：

| 格式 | 触发示例 | 解析器 |
|------|----------|--------|
| JSON `<tool_call>` | Llama / Qwen / DeepSeek | `_parse_xml_tool_calls` |
| Qwen3.5 XML `<function=...>` | Qwen3.5 | `_parse_xml_tool_calls` |
| Gemma `<start_function_call>` | Gemma | `_parse_gemma4_tool_call_fallback` + 自定义 `_squote_close_positions` 状态机 |
| GLM `<arg_key>/<arg_value>` | GLM-4.7 / 5 | 自定义 XML 解析 |
| MiniMax namespaced | MiniMax | `_parse_namespaced_tool_calls` |
| Mistral `[TOOL_CALLS]` | Mistral | `_parse_bracket_tool_calls` |
| Kimi K2 `<\|tool_calls_section_begin\|>` | Kimi | 自定义 token-aware parser |
| Longcat `<longcat_tool_call>` | Longcat | 自定义 parser |

**Gemma 4 鲁棒性回退**：

```python
def _gemma4_args_to_json_robust(args_str: str) -> dict:
    # iterative parsing 应对嵌套 JSON / 转义字符 / 引号未闭合
```

→ 通过 iterative parsing 处理异常情况，比 naive 正则更鲁棒。

### 5.8 oQ 量化原理

```
模型 weight (FP16)
  ↓ universal_quant_predicate
判断每个 tensor 是否需要量化
  ↓ 提取 sensitivity score
敏感度分级 tier 1-5 (5 = 最敏感，如 lm_head / vision_tower / 早期层)
  ↓ oq_level (1-5)
目标 BPW = 1.5-8 bit 范围
  ↓ QuantPlan
group_size=32/64/128, mode=affine/quantile/mxfp
  ↓ 输出
混合精度模型 (e.g., 4.5 BPW 整体)
```

**保护策略**：

```python
def _is_vision_tensor(name: str) -> bool:
    return "vision" in name.lower() or "image" in name.lower()
# vision tower 默认不量化
def _is_moe_router(path: str) -> bool:
    return ".gate." in path or ".router." in path
# MoE 路由层低 bits（高敏感）
```

### 5.9 macOS App 与 CLI 通信原理

```
┌──────────────────┐                        ┌──────────────────┐
│   macOS App      │                        │   omlx CLI       │
│ (SwiftUI/Swift)  │                        │   (Python)       │
└────────┬─────────┘                        └─────────┬────────┘
         │                                            │
         │  Unix Domain Socket                        │
         │  ~/.omlx/run/app-control.sock              │
         │                                            │
         │   {"cmd": "start", "port": 8000}           │
         ├───────────────────────────────────────────►│
         │                                            │  spawn:
         │                                            │  python -m omlx.cli serve
         │                                            │
         │   {"state": "running", "pid": 12345}       │
         │◄───────────────────────────────────────────┤
```

- App 通过 `AppControlServer.swift` 监听 socket
- CLI 通过 `cli.py::_send_app_control` 发送请求
- 状态变更通过 `NotificationCenter` 在 Swift 侧广播到 MenubarController

### 5.10 启动时的 venvstacks 分层打包

macOS App 内嵌 Python 解释器：

```bash
# 阶段 1: 构建 Python layer（10-20 分钟冷启动）
packaging/_export/  ← 缓存的 layer

# 阶段 2: xcodebuild + Python layer 嵌入 + ad-hoc 签名
apps/omlx-mac/Scripts/build.sh release

# 产出: apps/omlx-mac/build/Stage/oMLX.app
```

**venvstacks** 允许将 Python 解释器、依赖、omlx 包分层打包，用户双击 .app 即可运行，零安装。

### 5.11 Engine Lifecycle 状态机

```
ServerProcess.swift:
   stopped ─start()→ starting ─/health 200→ running ─/health fail×3→ unresponsive
                       │                       │ ↑                       │
                       │                       │ └─/health or status OK──┘
                       │                       │
                       │                       └─process exit → auto-restart
                       └─process exit during startup → auto-restart

   stop()  : * → stopping → SIGTERM → wait ≤10s → SIGKILL → stopped
   forceRestart() : * → SIGKILL → start()
   crashes : auto-restart with 5s/10s/20s backoff, max 3 attempts, counter
             resets after 60s of stable .running
```

### 5.12 SSE 流式输出与 keep-alive

```python
# 关键设计：避免 Claude Code 等长 prefill 时 SSE read timeout
# chunk 模式：发出符合协议的无操作事件，兼容严格客户端（OpenClaw / WorkBuddy）
# comment 模式：传统 SSE `: keep-alive` 注释
# off 模式：禁用
serve_parser.add_argument(
    "--sse-keepalive-mode",
    type=str,
    choices=["chunk", "comment", "off"],
    default="chunk",
)
```

### 5.13 模型自动发现与分类

```python
# omlx/model_discovery.py
class DiscoveredModel:
    model_id: str
    model_path: str
    model_type: ModelType       # "llm" / "vlm" / "embedding" / "reranker" / "audio_*"
    engine_type: EngineType
    estimated_size: int         # 预先从 safetensors 大小估算
    thinking_default: bool | None
    preserve_thinking_default: bool | None
    model_context_length: int | None
```

**启发式识别**：

```python
def _is_causal_lm_reranker(model_path):
    return "reranker" in model_path.name.lower()

def _is_causal_lm_embedding(model_path):
    return "embedding" in model_path.name.lower()
```

→ CausalLM 微调的 embedding/reranker 模型无法从 config.json 区分，靠目录名识别。

---

## 六、关键文件清单

| 文件 | 行数 | 作用 |
|------|------|------|
| `omlx/scheduler.py` | 10114 | 调度核心（最复杂） |
| `omlx/server.py` | 6569 | FastAPI 路由 + 中间件 |
| `omlx/oq.py` | ~168K | 量化引擎（含 calibration data） |
| `omlx/engine_pool.py` | 1717 | 多模型 LRU 管理 |
| `omlx/engine_core.py` | 1225 | 单模型引擎 + MLX executor |
| `omlx/cache/paged_cache.py` | ~1583 | Block-based KV 缓存 |
| `omlx/cache/paged_ssd_cache.py` | ~3439 | SSD 冷层 |
| `omlx/cache/prefix_cache.py` | ~2895 | 前缀共享 + CoW |
| `omlx/cache/hybrid_cache.py` | ~259 | 混合 cache 类型（KVCache + ArraysCache） |
| `omlx/process_memory_enforcer.py` | ~1126 | 内存守护 |
| `omlx/memory_monitor.py` | ~834 | 内存监控工具 |
| `omlx/api/tool_calling.py` | ~1450 | 7 种 tool call 格式 |
| `omlx/api/openai_models.py` | ~541 | OpenAI Pydantic schema |
| `omlx/api/anthropic_models.py` | ~300 | Anthropic Pydantic schema |
| `omlx/api/adapters/openai.py` | - | OpenAI adapter |
| `omlx/api/adapters/anthropic.py` | ~218 | Anthropic adapter |
| `omlx/engine/batched.py` | ~885 | BatchedEngine |
| `omlx/engine/vlm.py` | ~2583 | VLMBatchedEngine |
| `omlx/engine/embedding.py` | - | EmbeddingEngine |
| `omlx/engine/reranker.py` | - | RerankerEngine |
| `omlx/engine/dflash.py` | - | DFlashEngine |
| `omlx/cli.py` | 971 | CLI 入口 |
| `omlx/config.py` | 251 | 集中配置管理 |
| `omlx/turboquant_kv.py` | ~415 | TurboQuant KV 集成 |
| `omlx/model_discovery.py` | ~1221 | 模型发现与分类 |
| `omlx/admin/routes.py` | ~3000+ | Admin HTTP 路由 |
| `omlx/admin/static/js/dashboard.js` | 5217 | Admin Web UI (Alpine.js) |
| `apps/omlx-mac/Sources/Server/ServerProcess.swift` | 444 | 进程生命周期 |
| `apps/omlx-mac/Sources/Server/AppControlServer.swift` | - | CLI 通信 socket |
| `apps/omlx-mac/Sources/Menubar/MenubarController.swift` | - | 菜单栏 UI |
| `omlx/patches/deepseek_v4/` | - | DeepSeek V4 monkey-patch |
| `omlx/patches/glm_moe_dsa/` | - | GLM MoE + DSA |
| `omlx/patches/step3p7/` | - | Step 3.7 模型 |
| `tests/` | - | 单元测试 + 慢速测试 |

---

## 七、总结

oMLX 是一个**深度垂直整合的 Apple Silicon LLM 推理栈**，它的核心壁垒不在单一算法，而在三层整合：

### 1. MLX 框架深度适配

- Thread-local stream（避免 Metal 命令缓冲竞争）
- Compile cache 生命周期管理（避免 worker 线程崩溃）
- Metal wired limit 调整（kernel 级 GPU 内存上限）

### 2. vLLM 风格的系统软件

- PagedAttention（block-based KV cache）
- Continuous batching（FCFS + chunked prefill + decode burst）
- Prefix sharing + Copy-on-Write
- 全部跑在 Apple Silicon 统一内存上

### 3. 本地化的产品体验

- 菜单栏 + admin UI + 一键集成 Claude Code/Codex 等
- 完全离线 admin UI（CDN vendored）
- 8 种语言 i18n
- Sparkle 自动更新
- venvstacks 内嵌 Python，零安装

### 真正原创的工程创新

**Tiered KV Cache 跨重启**：让 100GB+ 的"逻辑上下文"能在 32GB 内存的 Mac 上跑起来，是 Claude Code 等编码 agent 能跑小模型的关键。这个能力来自：

- `PagedCacheManager` (block 256 token)
- `PagedSSDCacheManager` (异步 demote + 链式哈希)
- `BoundarySnapshotStore` (VLM 图像边界)
- `HotCache` (RAM write-back)
- `SharedHotCacheBudget` (进程级共享)

→ 这些组件协同工作，让"重启后 KV 仍可用"、"上下文跨会话复用"、"多用户共享前缀"都成为可能。

### 整体定位

oMLX 不是单一算法突破，而是 **Apple Silicon 上 vLLM 的完整产品化**：把 MLX 的 Metal 性能、vLLM 的批处理/缓存理论、本地用户体验三层堆叠，做出一个让本地 LLM 真正"实用"的桌面级服务。

---

*文档生成时间：基于 omlx 仓库当前 HEAD 分析*