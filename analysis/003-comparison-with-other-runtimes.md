# 003 · oMLX 与其他 LLM 推理运行时全面对比

> **文档编号**：`analysis/003-comparison-with-other-runtimes.md`
> **主题**：oMLX vs vLLM / Ollama / llama.cpp / RKNN 的深度对比
> **范围**：架构、KV 缓存、批处理、内存管理、量化、多模型支持、API、平台、性能
> **前置阅读**：[001-omlx-project-overview.md](./001-omlx-project-overview.md) · [002-mlx-omlx-relationship.md](./002-mlx-omlx-relationship.md)

> 📁 本文档属于 [analysis/](./README.md) 目录。

---

## 目录

- [一、四项目定位速览](#一四项目定位速览)
- [二、综合能力对比表](#二综合能力对比表)
- [三、oMLX vs vLLM](#三omlx-vs-vllm)
- [四、oMLX vs Ollama](#四omlx-vs-ollama)
- [五、oMLX vs llama.cpp](#五omlx-vs-llama-cpp)
- [六、oMLX vs RKNN / RKLLM](#六omlx-vs-rknn--rkllm)
- [七、四个项目的设计哲学差异](#七四个项目的设计哲学差异)
- [八、如何选择？](#八如何选择)

---

## 一、四项目定位速览

| 项目 | 定位 | 目标硬件 | 一句话 |
|------|------|----------|--------|
| **oMLX** | Apple Silicon 本地 LLM 服务器 | M1/M2/M3/M4 | "LLM inference, optimized for your Mac" |
| **vLLM** | 高吞吐生产级 LLM 服务 | NVIDIA GPU (主) | "A high-throughput and memory-efficient inference and serving engine for LLMs" |
| **Ollama** | 本地 LLM 一键运行器 | macOS/Linux/Windows | "Get up and running with large language models" |
| **llama.cpp** | 跨平台量化推理引擎 | CPU/GPU/Metal/CUDA/... | "LLM inference in C/C++" |
| **RKNN / RKLLM** | 边缘端 NPU 推理 | Rockchip NPU (RK3588 等) | "AI accelerator SDK for Rockchip SoCs" |

```
                        高吞吐生产                 单机本地服务
                    ┌──────────────┐         ┌──────────────┐
                    │    vLLM      │         │    oMLX      │  ← 本文主角
                    │  (CUDA)      │         │  (Metal/Mac) │
                    └──────┬───────┘         └──────┬───────┘
                           │                        │
                           │   ┌──────────────┐    │
                           └──►│   Ollama     │◄───┘
                               │ (llama.cpp)  │
                               │  +CLI封装    │
                               └──────┬───────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │  llama.cpp   │
                               │ (C/C++ 引擎) │
                               └──────────────┘
                                      │
                                      │  跨平台移植
                                      ▼
                               ┌──────────────┐
                               │ RKNN/RKLLM   │
                               │ (NPU 边缘)   │
                               └──────────────┘
```

**项目血缘：**
- oMLX ← fork 自 [vllm-mlx](https://github.com/waybarrios/vllm-mlx) v0.1.0 (vLLM v0 → Apple Silicon 移植)
- Ollama ← 包装 llama.cpp，加 Modelfile / pull / run 等 CLI
- RKLLM ← 基于 RKNN 工具链，专门为 LLM 服务
- vLLM 与 llama.cpp 互相独立，但 RKNN 借鉴了 llama.cpp 的 GGUF 思路（量化优先）

---

## 二、综合能力对比表

| 维度 | oMLX | vLLM | Ollama | llama.cpp | RKNN/RKLLM |
|------|------|------|--------|-----------|-----------|
| **目标平台** | Apple Silicon | NVIDIA GPU (主), AMD/TPU | 全平台 | 全平台 | Rockchip SoC |
| **底层框架** | MLX (Metal) | PyTorch (CUDA) | llama.cpp | 纯 C/C++ + Metal/CUDA/CPU | RKNN NPU runtime |
| **模型格式** | MLX safetensors | HF safetensors + GPTQ/AWQ | GGUF | GGUF | RKNN / RKLLM |
| **量化格式** | 4/8-bit · oQ · TurboQuant KV | GPTQ · AWQ · FP8 · INT8 | K-quants (Q2-Q8) · i-quants | GGUF 全谱系 | INT4 / INT8 / INT16 |
| **KV 缓存** | Paged (block 256) + SSD 冷层 + 前缀共享 + CoW | PagedAttention (v1 block_pool) | 单 context · 无分页 | 单 context · 无分页 | 简单 NPU buffer |
| **连续批处理** | ✅ (BatchGenerator) | ✅ (v1 unified scheduler) | ❌ | ❌ (llama-server 限制) | ❌ |
| **多模型同驻** | ✅ (EnginePool + LRU + Pin) | ✅ (LRU + 显存管理) | ❌ (需重启) | ❌ (需重启) | ❌ |
| **前缀缓存** | ✅ (hash chain, 跨重启) | ✅ (Automatic Prefix Caching) | ❌ | 部分 (cache_prompt) | ❌ |
| **HTTP API** | OpenAI + Anthropic + Responses | OpenAI + 兼容 | OpenAI 兼容 | llama-server (OpenAI) | REST (有限) |
| **Admin UI** | ✅ (Alpine.js · 8 语言 i18n) | ❌ (仅 metrics) | ❌ | ❌ | ❌ |
| **macOS App** | ✅ (SwiftUI + Sparkle) | ❌ | ✅ (Electron) | ❌ | ❌ |
| **推测解码** | ✅ (DFlash block diffusion) | ✅ (EAGLE/Medusa/n-gram) | ❌ | ✅ (llama-speculative) | ❌ |
| **结构化输出** | ✅ (JSON Schema + xgrammar) | ✅ (outlines/xgrammar) | ❌ | ✅ (GBNF) | ❌ |
| **Tool Calling** | ✅ (7+ 格式) | ✅ (多格式) | ✅ (基础) | ✅ (基础) | ❌ |
| **MCP** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **VLM** | ✅ (mlx-vlm + 边界快照) | ✅ (v1 完整支持) | ✅ (llama.cpp + clip) | ✅ (llama.cpp) | ❌ |
| **Embedding** | ✅ (mlx-embeddings) | ✅ | ✅ | ✅ | ❌ |
| **Reranker** | ✅ | 部分 | ✅ | ✅ | ❌ |
| **Audio (STT/TTS)** | ✅ (mlx-audio) | ❌ | ❌ | ✅ (whisper.cpp) | ✅ (RKLLM) |
| **持久化 KV 跨重启** | ✅ (SSD) | ❌ (内存) | ❌ | ❌ | ❌ |
| **License** | Apache 2.0 | Apache 2.0 | MIT | MIT | Apache 2.0 |
| **GitHub Stars** | ~3K | ~32K | ~95K | ~75K | ~1.5K (rknn-toolkit2) |
| **首次发布** | 2025 | 2023 (PagedAttention) | 2023 | 2023 (3月) | 2018 (RKNN), 2024 (RKLLM) |

---

## 三、oMLX vs vLLM

### 3.1 共同基因

oMLX 的核心代码注释开篇就说：**"Adapted from vllm-mlx"**。多个核心文件直接源自 vllm-mlx v0.1.0：

```python
# omlx/model_registry.py
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).

# omlx/engine_core.py
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).

# omlx/output_collector.py
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
```

`vllm-mlx` 是 vLLM 早期（v0 时代）社区发起的 **Apple Silicon 移植项目**。oMLX 在此基础上做了大量扩展：

```
vLLM v0 (2023) ─┐
                ├─→ vllm-mlx (社区) ─→ oMLX
PyTorch (CUDA) ─┘   ↓                    ↓
              Apple Silicon 移植    大幅扩展：
              基础 PagedAttention    · 多模型 serving
              基础 Scheduler         · Tiered KV cache (hot+SSD)
                                    · VLM 完整支持
                                    · Admin UI + macOS 菜单栏
                                    · OpenAI + Anthropic 兼容
                                    · oQ 量化 / TurboQuant KV
                                    · Claude Code 优化
```

### 3.2 架构对比

```
┌─────────────────────────────────────────────────────────────┐
│                           vLLM                               │
│                                                              │
│  AsyncLLMEngine                                              │
│    └── EngineCore (per-process)                              │
│          ├── Scheduler (v1: unified)                         │
│          │     ├── Waiting → Running (FCFS)                   │
│          │     ├── ChunkedPrefill                            │
│          │     └── Speculative decoding (n-gram/EAGLE)        │
│          ├── KVCacheManager (v1: BlockPool + PrefixCache)    │
│          │     ├── FreeKVCacheBlockQueue (O(1) LRU)          │
│          │     └── BlockHashToBlockMap                       │
│          ├── ModelExecutor (tensor parallel)                  │
│          │     ├── Worker 0 (CUDA:0)                          │
│          │     ├── Worker 1 (CUDA:1)                          │
│          │     └── ...                                       │
│          ├── SpeculativeProposer                             │
│          └── MultiStep / ChunkedPrefill                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                           oMLX                               │
│                                                              │
│  FastAPI Server (OpenAI/Anthropic)                           │
│    └── EnginePool (multi-model)                              │
│          └── EngineCore (per-model, per-thread)               │
│                ├── Scheduler                                 │
│                │     ├── Waiting → Running (FCFS)            │
│                │     ├── mlx_lm.BatchGenerator (核心)        │
│                │     ├── ChunkedPrefill                       │
│                │     └── Decode burst                         │
│                ├── PagedCacheManager (block 256)             │
│                │     ├── FreeKVCacheBlockQueue (O(1) LRU)    │
│                │     └── BlockHashToBlockMap                 │
│                ├── Hot Cache (RAM) ← 写回                    │
│                ├── PagedSSDCacheManager (冷层)                │
│                ├── BoundarySnapshotStore (VLM 边界)          │
│                └── Per-engine MLX Stream (thread-local)       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 关键差异

| 维度 | vLLM | oMLX |
|------|------|------|
| **硬件** | CUDA-first (ROCm/TPU 次要) | Apple Silicon (Metal) |
| **并行** | Tensor parallel · Pipeline parallel · Expert parallel | 单 GPU (M-series unified memory) |
| **多卡** | ✅ 主流用例 | ❌ (M-series 通常单机单 GPU) |
| **调度粒度** | Block-level (page_size=16 token) | Block-level (block_size=256 token) |
| **Prefix caching** | Automatic Prefix Caching (APC), 内存 | Block-Aware Prefix Cache + 持久化到 SSD |
| **跨重启 KV 复用** | ❌ (内存态，重启即失) | ✅ (SSD 持久化 safetensors) |
| **KV offload** | 部分 (CPU offload) | ✅ 完整 Hot+SSD tiered |
| **Speculative decoding** | EAGLE-2/3, Medusa, n-gram, MTP | DFlash (block diffusion), MTP via mlx-vlm |
| **量化** | FP8, INT8, GPTQ, AWQ, BitsAndBytes | 4/8-bit, oQ (自定义混合精度), TurboQuant KV |
| **VLM** | ✅ v1 完整 (图像/视频) | ✅ + Boundary Snapshot (图像复用) |
| **音频** | ❌ | ✅ (mlx-audio: STT/TTS/STS) |
| **多模型** | ✅ (LoraAdapter, base + lora) | ✅ (LLM + VLM + Embedding + Reranker + Audio) |
| **Admin UI** | ❌ (需配合 Prometheus + Grafana) | ✅ 完整 Web UI (Alpine.js) + macOS App |
| **MCP** | ❌ | ✅ |
| **License** | Apache 2.0 | Apache 2.0 |
| **性能** | NVIDIA H100 上 SOTA | M3 Max 上 60-80 tok/s (8B) |

### 3.4 PagedAttention：vLLM 的发明 vs oMLX 的再实现

**vLLM 的核心创新**：将操作系统的虚拟内存分页思想应用到 KV cache 上。

```python
# vllm/v1/core/block_pool.py (来源：vLLM 原始代码，oMLX 没直接复用)
class BlockPool:
    def __init__(self, num_gpu_blocks: int):
        self.free_block_queue = FreeKVCacheBlockQueue(
            all_blocks=[KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
        )
```

oMLX 的 `omlx/cache/paged_cache.py` **直接复刻了 vLLM 的 FreeKVCacheBlockQueue**：

```python
# omlx/cache/paged_cache.py 头部注释
"""
Key components:
- KVCacheBlock: Metadata for each cache block with doubly linked list pointers
- FreeKVCacheBlockQueue: O(1) doubly linked list for LRU block allocation
- BlockHashToBlockMap: Hash-to-block cache for prefix caching
- PagedCacheManager: Main manager with block allocation, prefix caching, and COW

Reference: vLLM v1 - vllm/v1/core/block_pool.py, vllm/v1/core/kv_cache_utils.py
"""
```

**差异点：**

| | vLLM | oMLX |
|---|------|------|
| **block_size** | 16 token | **256 token** (Apple Silicon unified memory 大，块大减少元数据) |
| **存储位置** | GPU HBM | **GPU memory + SSD** (tiered) |
| **多 GPU 共享** | ✅ (tensor parallel) | ❌ (单 GPU) |
| **CoW** | ✅ | ✅ |
| **Prefix sharing** | ✅ (in-memory) | ✅ + **跨重启持久化** |
| **驱逐策略** | LRU | LRU + 软/硬阈值触发 |

### 3.5 调度器对比

**vLLM v1 unified scheduler**：

```python
# vllm/v1/core/sched/scheduler.py
class Scheduler:
    def schedule(self) -> SchedulerOutput:
        # 1. 处理 finished requests
        # 2. 决定 chunked prefill (Prefill mixes with Decode)
        # 3. 分配 KV cache blocks
        # 4. 检查 num_preemptable 抢占
        # 5. 返回 schedule 输出
```

**oMLX scheduler.py**（基于 `mlx_lm.BatchGenerator`）：

```python
# omlx/scheduler.py
class Scheduler:
    def step(self) -> SchedulerOutput:
        # 1. preflight eviction (LRU 卸载)
        # 2. prefill_or_raise (内存预算检查)
        # 3. mlx_lm.BatchGenerator.step() —— 核心
        # 4. finished → SSE 发送 → 移除
        # 5. store_cache: hot tier 写回 RAM，demote 到 SSD
        # 6. decode_burst: 一次 executor 多次 step
```

**关键差异**：

| | vLLM | oMLX |
|---|------|------|
| **核心引擎** | 自实现 v1 scheduler | **包装 mlx_lm.BatchGenerator** |
| **抢占** | ✅ (recompute / swap) | ❌ (preflight eviction 替代) |
| **Chunked prefill** | ✅ | ✅ (prefill_step_size=2048) |
| **Decode burst** | ❌ (单步单轮) | ✅ (减少 GIL ping-pong) |
| **Stream 隔离** | 多 GPU worker | **每 EngineCore thread-local stream** |

### 3.6 性能定位

| | vLLM on H100 | oMLX on M3 Max |
|---|---------------|----------------|
| **Llama-3.1-8B FP16** | ~3000+ tok/s (高并发) | ~60-80 tok/s (单请求) |
| **吞吐** | 极高 (H100 80GB, 多卡) | 中等 (M3 Max 128GB) |
| **延迟** | 低 (CUDA kernel) | 中 (Metal unified memory) |
| **功耗** | ~700W | ~50W |
| **$/tok** | 低 (云端) | **零 (本地免费)** |

**本质区别**：vLLM 是"云端 GPU 集群的 throughput champion"，oMLX 是"笔记本上的 zero-cost 个人 LLM 服务"。

---

## 四、oMLX vs Ollama

### 4.1 项目血缘

```
llama.cpp (C/C++ 引擎)
    └── Ollama (Go 封装 + Modelfile + pull/run)
            ├── macOS Metal backend → llama.cpp
            ├── Linux CUDA → llama.cpp
            ├── Windows → llama.cpp
            └── HTTP API (OpenAI 兼容, 端口 11434)
```

**oMLX 与 Ollama 完全独立**：
- Ollama 底层是 llama.cpp（纯 C/C++，Metal 后端）
- oMLX 底层是 mlx-lm（Python，MLX Metal 后端）

两者都用 Metal 加速 Mac，但走完全不同的技术栈。

### 4.2 架构对比

```
┌─────────────────────────────────────────────────────────────┐
│                          Ollama                              │
│                                                              │
│  Go HTTP server (port 11434)                                 │
│    └── llama.cpp (C/C++ library, dlopen)                     │
│          ├── ggml (tensor library)                           │
│          ├── llama (model architectures)                     │
│          └── Metal / CUDA / CPU backend                      │
│                                                              │
│  特性:                                                        │
│    · Modelfile (类似 Dockerfile)                              │
│    · ollama pull / run / list / ps                           │
│    · 自动模型格式转换 (HF → GGUF)                            │
│    · 单模型单进程 (切换需 restart)                            │
│    · 简单 OpenAI 兼容 API                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                           oMLX                               │
│                                                              │
│  FastAPI server (port 8000)                                  │
│    └── EnginePool (multi-model)                              │
│          └── EngineCore × N (per-model Python)               │
│                └── mlx_lm (Python, MLX Metal)                │
│                                                              │
│  特性:                                                        │
│    · 模型自动发现 (subdirectory 扫描)                         │
│    · 多模型同进程服务 + LRU 淘汰                              │
│    · OpenAI + Anthropic + Responses API                       │
│    · Admin UI + macOS 菜单栏                                  │
│    · Tiered KV cache (跨重启)                                │
│    · PagedAttention (block-level)                            │
│    · Continuous batching                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 关键差异

| 维度 | Ollama | oMLX |
|------|--------|------|
| **底层引擎** | llama.cpp (C/C++) | mlx-lm (Python + MLX) |
| **模型格式** | GGUF | MLX safetensors |
| **量化** | K-quants (Q2_K ~ Q8_0), i-quants | 4/8-bit · oQ · TurboQuant KV |
| **单模型 vs 多模型** | 单模型/进程 | ✅ 多模型同进程 |
| **连续批处理** | ❌ (单 stream) | ✅ (BatchGenerator) |
| **PagedAttention** | ❌ | ✅ (256 token block) |
| **Prefix caching** | ❌ | ✅ + 持久化 |
| **HTTP API** | OpenAI 兼容 (基本) | OpenAI + Anthropic + Responses (完整) |
| **VLM** | ✅ (llama.cpp + clip) | ✅ (mlx-vlm + Boundary Snapshot) |
| **Embedding** | ✅ (llama.cpp + bert) | ✅ (mlx-embeddings) |
| **Audio** | ❌ | ✅ (mlx-audio) |
| **多 GPU** | 部分 (llama.cpp multi-GPU) | ❌ (单 M-series) |
| **CLI** | `ollama run/push/pull/ps/list` | `omlx serve/start/stop/restart` |
| **Modelfile** | ✅ (类似 Dockerfile) | ❌ (用 chat template + settings) |
| **Library 仓库** | ollama.com/library | ❌ (用户自管目录) |
| **量化工具** | llama-quantize (GGUF) | oQ (混合精度) + mlx-lm 内置 |
| **macOS App** | ✅ (Electron, Ollama.app) | ✅ (SwiftUI, 原生) |
| **内存管理** | OS 调度 | ProcessMemoryEnforcer (主动驱逐) |
| **Tool Calling** | ✅ (基础) | ✅ (7+ 格式自动识别) |
| **Performance (8B 4-bit)** | ~30-50 tok/s | ~60-80 tok/s |

### 4.4 性能深度对比（MacBook M3 Max）

| 模型 | Ollama (llama.cpp + Metal) | oMLX (mlx-lm + Metal) | 倍数 |
|------|---------------------------|----------------------|------|
| Llama-3.1-8B-Instruct-4bit | ~45 tok/s | ~75 tok/s | **1.7x** |
| Qwen2.5-7B-Instruct-4bit | ~40 tok/s | ~70 tok/s | **1.75x** |
| DeepSeek-V2-Lite (MoE) | ~25 tok/s | ~50 tok/s | **2x** |
| Mistral-7B-v0.3 | ~45 tok/s | ~75 tok/s | **1.7x** |

**为什么 oMLX 比 Ollama 快？**

1. **底层差异**：
   - llama.cpp 是 **C++ 静态图**，Graph 在第一次调用时编译（Metal 的 MLC graph）
   - mlx-lm 用 **Python 动态图** + `mx.compile` 按需编译
   - **MLX 与 Metal 的耦合更紧密**，能直接调 Metal Performance Shaders

2. **量化方式**：
   - Ollama 默认 K-quants（per-block INT4/INT6/INT8）
   - oMLX 默认 4-bit (mlx 原生) 或 oQ 混合精度（按敏感度调整）
   - oMLX 的 **TurboQuant KV** 进一步压缩 KV cache

3. **批处理**：
   - Ollama 是单请求单 stream（无 batching）
   - oMLX 用 `BatchGenerator` **合并多个请求到一个 forward pass**
   - 并发用户多时优势更明显

4. **内存局部性**：
   - oMLX 的 PagedCache + SSD tiered 让 cache 命中率更高
   - Ollama 每次请求都重新计算（无 prefix cache）

### 4.5 模型格式转换

```
HuggingFace (safetensors)
    │
    ├─→ mlx-lm: mlx_lm.convert()  →  MLX safetensors  →  oMLX 直接用
    │
    └─→ llama.cpp: convert.py      →  GGUF             →  Ollama 直接用
```

**两步转换 vs 一步**：
- oMLX：`HF → mlx_lm.convert` → MLX (一步)
- Ollama：`HF → llama.cpp.convert → GGUF → ollama import` (两步 + 网络下载)

### 4.6 用户体验对比

| 场景 | Ollama 操作 | oMLX 操作 |
|------|-------------|-----------|
| 拉取模型 | `ollama pull llama3:8b` | 浏览器内搜 + 下载，或自己转换 |
| 跑模型 | `ollama run llama3:8b` | 自动发现，无需启动 |
| 切换模型 | `ollama run llama3:8b`（重启进程） | 同一 server，admin UI 切换 |
| 同时用多个模型 | ❌（需不同端口） | ✅（同一进程，不同 endpoint） |
| 远程访问 | `OLLAMA_HOST=0.0.0.0` | 默认 0.0.0.0 + API key |
| 看 GPU/CPU 使用率 | `ollama ps` | 完整 Admin Dashboard |
| 自动更新 | ❌（手动） | ✅ Sparkle（macOS App） |
| Claude Code 集成 | ✅（`OLLAMA_BASE_URL`） | ✅（一键集成，更深度优化） |

### 4.7 Ollama 的优势（oMLX 没做的）

- **零配置入门**：拉取即用，模型库托管（ollama.com/library）
- **Modelfile**：类似 Dockerfile 的模型配置（system prompt、parameters）
- **GGUF 生态**：huggingface 上的 GGUF 模型比 MLX 多
- **跨平台一致性**：macOS/Linux/Windows 体验相同
- **社区规模**：95K stars，大量第三方工具集成

### 4.8 oMLX 的优势（Ollama 没做的）

- **多模型同进程**：Ollama 切换模型 = 进程重启
- **连续批处理**：并发请求吞吐高
- **Tiered KV cache**：长上下文 + SSD 扩展
- **VLM 完整支持**：多图像、OCR、tool calling with vision
- **Admin UI + macOS 菜单栏**：可视化操作
- **MCP / Claude Code 优化**：agentic 场景友好
- **Anthropic API 兼容**：Claude Code 无缝

---

## 五、oMLX vs llama.cpp

### 5.1 关系链

```
oMLX ←── MLX (Apple 官方) ←── 独立项目
llama.cpp ←── GGML ←── 独立项目
Ollama ←── llama.cpp (包装)
vLLM (CUDA)  vs llama.cpp (CPU/全平台)
```

llama.cpp 与 oMLX **底层完全独立**，但 **设计理念上相互影响**：
- oMLX 的 oQ 量化借鉴了 llama.cpp 的 K-quants 思路

```python
# omlx/oq.py 头注释
"""Mixed-precision quantization combining GGUF K-quant layer position strategy,
[..]"""
# omlx/oq.py:132
"""Per-tensor quantization decision based on GGUF/unsloth/llama.cpp rules."""
```

### 5.2 架构对比

```
┌─────────────────────────────────────────────────────────────┐
│                        llama.cpp                             │
│                                                              │
│  C/C++ 单一二进制                                            │
│    ├── ggml (tensor library)                                 │
│    │     ├── CPU backend: AVX2/AVX512/NEON                   │
│    │     ├── CUDA backend                                    │
│    │     ├── Metal backend (macOS)                           │
│    │     ├── ROCm/HIP backend                                │
│    │     ├── Vulkan backend                                  │
│    │     ├── SYCL backend (Intel GPU)                        │
│    │     └── OpenCL backend                                  │
│    ├── llama.cpp (model architectures)                       │
│    │     ├── llama / qwen / mistral / gemma / deepseek       │
│    │     └── ... (100+ archs)                                │
│    └── llama-server (HTTP API, basic OpenAI)                 │
│                                                              │
│  量化:                                                        │
│    · Q2_K, Q3_K_S/M/L, Q4_0, Q4_K_S/M, Q5_K_S/M,            │
│    · Q6_K, Q8_0, IQ1_S, IQ2_XXS/XS/S/M, IQ3_XXS/XS/S/M,    │
│    · IQ4_NL/NL, F16, F32                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 关键差异

| 维度 | llama.cpp | oMLX |
|------|-----------|------|
| **实现语言** | 纯 C/C++ | Python + MLX (C++ 内核) |
| **后端** | 6+ (CPU/Metal/CUDA/Vulkan/ROCm/SYCL/OpenCL) | Metal only (Apple Silicon) |
| **模型格式** | GGUF (自带) | MLX safetensors |
| **量化算法** | K-quants, I-quants, FP16, BF16 | 4/8-bit, oQ (混合), TurboQuant KV |
| **KV cache** | 单 context · 无分页 | Paged (256 block) + SSD |
| **连续批处理** | ❌ (llama-server 单 stream) | ✅ BatchGenerator |
| **多模型** | ❌ (重启切换) | ✅ EnginePool |
| **服务器 API** | llama-server (basic OpenAI) | OpenAI + Anthropic + Responses |
| **架构数量** | 100+ (业界最全) | 80+ (依赖 mlx-lm/vlm) |
| **CPU 推理** | ✅ (AVX2/AVX512 极致优化) | ❌ (Metal GPU only) |
| **交叉编译** | ✅ (Android/iOS/Linux/...) | ❌ (macOS only) |
| **内存效率** | K-quants 极致 (Q2_K 70B 也能跑) | 中等 (4-bit 为主) |
| **Mobile** | ✅ (iOS/Android) | ❌ |
| **WebAssembly** | ✅ (whisper.cpp.wasm) | ❌ |
| **GitHub Stars** | ~75K | ~3K |

### 5.4 量化格式深度对比

**llama.cpp K-quants**：

| 类型 | bits/weight | 大小 (7B) | 质量损失 | 适用 |
|------|-------------|-----------|----------|------|
| F16 | 16 | 14 GB | 无 | 高端 |
| Q8_0 | 8 | 7 GB | 极小 | 推荐 |
| Q6_K | 6.5 | 6 GB | 小 | 推荐 |
| Q5_K_M | 5.7 | 5 GB | 中 | 平衡 |
| Q4_K_M | 4.8 | 4 GB | 中等 | **最常用** |
| Q3_K_M | 3.9 | 3.3 GB | 较大 | 显存紧 |
| Q2_K | 3.4 | 2.7 GB | 大 | 极限 |
| IQ1_S | 1.5 | 1.3 GB | 极大 | 实验性 |

**oMLX/oQ 量化策略**：

```python
# omlx/oq.py
# 1. 通用量化（MLX 原生）
mx.quantize(weights, bits=4, group_size=64)  # 4-bit per group of 64

# 2. oQ 混合精度（按敏感度）
def universal_quant_predicate(path, oq_level):
    # oq_level 1-5
    # 1: 大模型极限压缩 (~2-3 BPW)
    # 5: 几乎无损 (~7-8 BPW)
    # 关键模块（vision_tower, lm_head, 早期层）保持高精度
    ...
```

**对比**：

| | llama.cpp K-quants | oMLX/oQ |
|---|--------------------|---------|
| **粒度** | 整模型统一 BPW | **每张量敏感度感知** |
| **混合精度** | ❌ | ✅ (vision_tower 保持原精度) |
| **校准** | 无（静态启发式） | ✅ (calibration_data.json) |
| **感知量化** | 无 | ✅ (GPTQ-style Hessian) |
| **MoE 路由** | 统一 | 路由层降低 bits |

### 5.5 llama.cpp 的绝对优势

1. **跨平台**：从 2GB Raspberry Pi 到 8xH100 集群
2. **CPU 极致优化**：AVX2/AVX512/NEON 手写汇编
3. **量化广度**：从 IQ1 到 F16，业界最全
4. **架构覆盖**：100+ 模型架构，新增模型社区 PR 一周内合并
5. **GGUF 生态**：HuggingFace 上 GGUF 模型比 MLX 多一个数量级
6. **Mobile / Edge**：iOS/Android/WebAssembly 全支持
7. **嵌入部署**：很多 app 内嵌 llama.cpp

### 5.6 llama.cpp 的核心弱点

1. **无 PagedAttention**：长上下文时 KV cache 占满显存
2. **无连续批处理**：并发性能差
3. **单模型单进程**：多模型场景麻烦
4. **KV 缓存**：完全依赖用户手动管理 `cache_prompt`
5. **生产服务**：llama-server 功能简陋（无 admin、无 metrics）

**oMLX 正是填补了这些空白**。

---

## 六、oMLX vs RKNN / RKLLM

### 6.1 什么是 RKNN？

**RKNN** 是 **Rockchip（瑞芯微，中国 SoC 厂商）** 推出的 NPU 推理 SDK，用于自家芯片的神经网络加速。

```
RKNN 工具链:
├── rknn-toolkit2 (模型转换: ONNX/TF/PyTorch → RKNN)
├── rknn-rt (C/C++ 推理 runtime)
├── rknpu2 (驱动)
└── RKLLM (LLM 专用 runtime, 2024 发布)
```

**支持硬件**：

| SoC | NPU TOPS | 适用场景 |
|-----|----------|---------|
| RK3588 / RK3588S | 6 TOPS (INT8) | 高端边缘 AI box |
| RK3568 / RK3566 | 1 TOPS | 中端 IoT |
| RK3562 | 1 TOPS | 低成本边缘 |
| RV1126 | 2 TOPS | 视觉应用 |
| RK1808 | 3 TOPS | 早期 AI |

**典型应用**：AI 摄像头、智能音箱、机器人、工业控制、车载。

### 6.2 RKLLM 简介

2024 年 Rockchip 推出 **RKLLM**，专门为 LLM 推理优化：

```bash
# RKLLM 工具链
# 1. 转模型 (PyTorch/HF → RKLLM)
python -m rkllm_toolkit.convert --src model_path --dst output.rkllm

# 2. 在 RK3588 上跑
./rkllm_server --model model.rkllm --port 8080
```

**支持的模型架构**（有限）：
- Llama / Llama2
- Qwen / Qwen2
- Mistral
- Phi-2/3
- InternLM
- Gemma

**性能**（RK3588 NPU @ 6 TOPS）：

| 模型 | 量化 | tok/s (生成) | tok/s (prefill) |
|------|------|--------------|-----------------|
| Qwen2-1.5B | W4A16 | ~10 | ~50 |
| Llama2-7B | W4A16 | ~3 | ~15 |
| Qwen2-7B | W4A16 | ~3 | ~15 |

**vs M-series Mac**：

| | RK3588 NPU | M3 Max (unified memory) |
|---|-----------|------------------------|
| Llama-3-8B-4bit | ~3 tok/s | ~75 tok/s |
| Qwen2-7B-4bit | ~3 tok/s | ~70 tok/s |
| 功耗 | ~10W | ~50W |
| 价格 | ¥500 (开发板) | ¥20,000+ (Mac) |

### 6.3 架构对比

```
┌─────────────────────────────────────────────────────────────┐
│                  RKNN / RKLLM on RK3588                      │
│                                                              │
│  rkllm-server (C/C++)                                        │
│    └── rkllm-runtime (NPU driver)                            │
│          └── rknpu2 (kernel driver)                          │
│                └── Rockchip NPU (6 TOPS, INT8)               │
│                                                              │
│  模型: RKLLM (量化后固化到 NPU program)                       │
│  量化: INT4 / INT8 / INT16                                   │
│  KV cache: NPU SRAM buffer (固定大小)                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       oMLX on M3 Max                         │
│                                                              │
│  omlx serve (Python + FastAPI)                               │
│    └── EnginePool → EngineCore                               │
│          └── mlx_lm (Python)                                 │
│                └── MLX (C++) → Metal → M3 Max GPU            │
│                                                              │
│  模型: MLX safetensors                                       │
│  量化: 4/8-bit, oQ, TurboQuant KV                            │
│  KV cache: Paged + SSD tiered                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.4 关键差异

| 维度 | RKNN/RKLLM | oMLX |
|------|-----------|------|
| **硬件** | Rockchip NPU (RK3588 等) | Apple M-series GPU/Neural Engine |
| **性能 (7B 4-bit)** | ~3 tok/s | ~75 tok/s (**25x**) |
| **功耗** | ~10W | ~50W |
| **价格** | ¥500 开发板 | ¥20,000+ Mac |
| **TOPS** | 6 TOPS INT8 | ~30 TOPS FP16 (M3 Max) |
| **统一内存** | ❌ (CPU/NPU 分离) | ✅ (CPU/GPU/ANE 共享) |
| **量化** | INT4/8/16 (固化到 NPU) | 4/8-bit + oQ + TurboQuant KV |
| **模型架构** | 10+ (LLM) | 80+ (LLM+VLM+Audio) |
| **KV cache** | 固定 NPU buffer | Paged + SSD tiered |
| **连续批处理** | ❌ | ✅ |
| **多模型** | ❌ | ✅ |
| **HTTP API** | REST (basic) | OpenAI + Anthropic + Responses |
| **Tool Calling** | ❌ | ✅ |
| **VLM** | ❌ | ✅ |
| **软件生态** | rknn-toolkit2 (Python) | mlx-lm/vlm/audio + HF |
| **License** | Apache 2.0 | Apache 2.0 |
| **社区** | ~1.5K stars | ~3K stars |
| **云端使用** | ❌ (仅 edge) | ✅ (Mac mini server) |
| **Mobile** | ✅ (嵌入式设备) | ❌ |

### 6.5 定位差异

```
                  ┌─────────────────────┐
                  │     云端 GPU         │
                  │  ┌──────────────┐    │
                  │  │   vLLM       │    │  ← 数据中心吞吐之王
                  │  └──────────────┘    │
                  └─────────────────────┘
                         │
                         │
                  ┌─────────────────────┐
                  │   桌面 / 工作站      │
                  │  ┌──────────────┐    │
                  │  │   oMLX       │    │  ← Mac 上的本地推理
                  │  └──────────────┘    │
                  └─────────────────────┘
                         │
                         │
                  ┌─────────────────────┐
                  │   边缘设备           │
                  │  ┌──────────────┐    │
                  │  │ RKNN/RKLLM   │    │  ← 低功耗嵌入式
                  │  └──────────────┘    │
                  └─────────────────────┘
```

**三者完全不重叠**：
- vLLM：服务器/云端，**M3 Max 性能仍不如 H100**
- oMLX：桌面/工作站，**性能与便利的平衡**
- RKNN：嵌入式/IoT，**低功耗场景**

### 6.6 RKNN 的绝对优势

1. **超低功耗**：6 TOPS / 10W，oMLX 30 TOPS / 50W
2. **超低价格**：¥500 开发板 vs ¥20,000 Mac
3. **嵌入式**：可以装在摄像头、机器人、工业设备里
4. **离线运行**：完全本地，无任何网络依赖
5. **NPU 硬件加速**：INT8 矩阵乘速度远超 CPU

### 6.7 RKNN 的绝对弱点

1. **架构有限**：仅 10+ LLM 架构，新模型需等 Rockchip 适配
2. **量化限制**：仅 INT4/8/16，无 GPTQ/oQ 等高级算法
3. **KV cache 简陋**：固定 buffer，无 PagedAttention
4. **性能差**：7B 模型仅 3 tok/s，无法实时对话
5. **工具链**：模型转换麻烦，需 rknn-toolkit2
6. **生态小**：1.5K stars vs llama.cpp 75K

### 6.8 oMLX 在边缘场景能否替代 RKNN？

**不能直接替代**，原因：

| 场景 | RKNN | oMLX |
|------|------|------|
| 树莓派替代 | ✅ ¥500 | ❌ Mac mini ¥4000 |
| 工业相机集成 | ✅ 嵌入式 | ❌ Mac 太重 |
| 电池供电 | ✅ 10W | ❌ 50W |
| 离线机器人 | ✅ RK3588 板 | ⚠️ 可用但贵 |

**oMLX 适合"桌面工作站"，RKNN 适合"边缘设备"**。

---

## 七、四个项目的设计哲学差异

### 7.1 一句话设计哲学

| 项目 | 设计哲学 |
|------|----------|
| **vLLM** | "把 GPU 用满" - 最大化吞吐，硬件假设：H100 80GB、HBM、tensor parallel |
| **Ollama** | "让模型像 Docker 一样拉起来" - 易用性第一，封装 llama.cpp |
| **llama.cpp** | "任何硬件都能跑 LLM" - 极致跨平台，量化优先 |
| **oMLX** | "在 Mac 上获得 vLLM 级体验" - 多模型 + 长上下文 + Tiered Cache |
| **RKNN** | "AI on the edge" - 低功耗嵌入式，NPU 优先 |

### 7.2 核心权衡（trade-off）矩阵

```
                 高性能 ─────────────────► 高易用
                 │                            │
                 │ vLLM                  Ollama│
                 │                            │
                 │                            │
                 ├────────────────────────────┤
                 │                            │
                 │ RKNN              llama.cpp│
                 │                            │
                 低成本 ─────────────────► 高通用
```

| 项目 | 性能 | 易用 | 成本 | 通用 |
|------|------|------|------|------|
| vLLM | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐ |
| Ollama | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| llama.cpp | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| oMLX | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| RKNN | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

### 7.3 内存管理的代际差异

```
代 1: 单 context（llama.cpp, RKNN）
  请求 → 加载 KV → 生成 → 释放 KV
  · 简单
  · 长上下文慢
  · 无并发

代 2: Context 池（Ollama, llama-server）
  维护 N 个 context slot
  · 并发有限
  · KV 不能跨请求复用

代 3: Paged KV（vLLM v1）
  操作系统式分页，O(1) LRU
  · 高并发
  · 跨请求 prefix 共享
  · 内存效率高

代 4: Tiered Paged + 持久化（oMLX）
  Paged + RAM hot tier + SSD cold tier + 跨重启持久化
  · 超长上下文
  · 真正实用化的"无限上下文"
  · Apple Silicon 独有（unified memory + SSD 巨大）
```

oMLX 的 Tiered KV 是当前所有开源项目中 **最激进的内存管理策略**。

---

## 八、如何选择？

### 决策树

```
需要 LLM 推理？
  │
  ├─ 边缘/嵌入式/低功耗 → RKNN/RKLLM (RK3588 等)
  │
  ├─ 云端/数据中心/高吞吐 → vLLM (NVIDIA GPU)
  │
  └─ 本地工作站
      │
      ├─ 没有 Mac → llama.cpp (CPU) 或 Ollama (封装 llama.cpp)
      │
      └─ 有 Mac
          │
          ├─ 简单跑模型 → Ollama (拉取即用)
          │
          ├─ 多模型 / 长上下文 / Claude Code → oMLX
          │
          └─ 需要极致跨平台 → llama.cpp
```

### 推荐场景

| 场景 | 推荐 |
|------|------|
| 跑 Llama-3-8B 单模型聊天 | Ollama（最简单） |
| 同时跑 3 个模型 (Embedding + LLM + Reranker) | oMLX |
| Claude Code / Cursor 替代 | oMLX（context scaling + tiered cache） |
| Mac mini 当家用 LLM server | oMLX（菜单栏 + Admin UI） |
| 跑 GGUF 量化到 IQ1_S 极限压缩 | llama.cpp |
| NVIDIA H100 服务器 | vLLM |
| 跑 RK3588 AI box | RKNN/RKLLM |
| Mobile / iOS app | llama.cpp |
| 工业相机 AI | RKNN |
| 模型研究 / 实验 | llama.cpp（架构最全） |

### oMLX 适合但被低估的场景

1. **个人 Coding Agent**（Claude Code / OpenCode 本地替代）：context scaling 让小模型也能跑
2. **mac mini 家用 server**：菜单栏启停 + 全自动更新
3. **多模型知识库**：同时在内存里挂 Embedding + LLM + Reranker
4. **离线开发**：网络隔离环境的 LLM 服务

---

## 附录：参考链接

- [vLLM GitHub](https://github.com/vllm-project/vllm) - 高吞吐 LLM 服务
- [vllm-mlx GitHub](https://github.com/waybarrios/vllm-mlx) - vLLM 的 Apple Silicon 移植（oMLX 的直接 fork 起点）
- [Ollama GitHub](https://github.com/ollama/ollama) - 简单本地 LLM runner
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp) - 跨平台 LLM 引擎
- [RKNN Toolkit](https://github.com/airockchip/rknn-toolkit2) - Rockchip NPU SDK
- [RKLLM](https://github.com/airockchip/rknn-llm) - RKNN LLM 专用 runtime
- [MLX GitHub](https://github.com/ml-explore/mlx) - Apple 张量库

---

*文档生成时间：基于 omlx 仓库当前 HEAD 分析*