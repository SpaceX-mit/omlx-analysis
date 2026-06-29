# 400 · 多模型推理全链路：LLM/VLM/OCR/ASR/TTS/STS/Embedding/Reranker 端到端管道

> **文档编号**：`analysis/400-multi-model-inference-pipeline.md`
> **主题**：从 HTTP 请求进入到 Metal GPU 执行的完整多模型推理管道
> **范围**：模型发现、EnginePool 调度、各类 Engine 实现、HTTP 路由、SSE 流式响应
> **前置阅读**：[001-omlx-project-overview.md](./001-omlx-project-overview.md) · [200-batch-generator-and-continuous-batching.md](./200-batch-generator-and-continuous-batching.md)

> 📁 本文档属于 [analysis/](./README.md) 目录。

---

## 目录

- [一、支持的模型类型](#一支持的模型类型)
- [二、整体架构总览](#二整体架构总览)
- [三、阶段 1：模型发现与分类](#三阶段-1模型发现与分类)
- [四、阶段 2：HTTP 路由与 API 适配](#四阶段-2http-路由与-api-适配)
- [五、阶段 3：EnginePool 多模型调度](#五阶段-3enginepool-多模型调度)
- [六、阶段 4：Engine 创建与加载](#六阶段-4engine-创建与加载)
- [七、阶段 5：推理执行（按引擎类型分支）](#七阶段-5推理执行按引擎类型分支)
- [八、阶段 6：流式输出（SSE / WebSocket / multipart）](#八阶段-6流式输出sse--websocket--multipart)
- [九、阶段 7：缓存存储与异步清理](#九阶段-7缓存存储与异步清理)
- [十、阶段 8：内存管理与 LRU 驱逐](#十阶段-8内存管理与-lru-驱逐)
- [十一、各类模型的差异点对照](#十一各类模型的差异点对照)
- [十二、为什么 oMLX 不支持 YOLO](#十二为什么-omlx-不支持-yolo)
- [十三、端到端时序示例](#十三端到端时序示例)
- [十四、总结](#十四总结)

---

## 一、支持的模型类型

| 类型 | 引擎类 | 底层库 | 主要场景 |
|------|--------|--------|----------|
| **LLM** (大语言模型) | `BatchedEngine` | `mlx_lm` | 文本对话、Completion、Tool Calling |
| **VLM** (视觉语言模型) | `VLMBatchedEngine` | `mlx_vlm` | 多图像对话、视觉问答、OCR |
| **OCR** (文本识别) | `VLMBatchedEngine` (专用) | `mlx_vlm` | DeepSeek-OCR / DOTS-OCR / GLM-OCR |
| **Embedding** (文本嵌入) | `EmbeddingEngine` | `mlx_embeddings` | RAG、语义检索 |
| **Reranker** (重排序) | `RerankerEngine` | 自实现 + mlx_embeddings | 检索结果排序 |
| **ASR / STT** (语音转文字) | `STTEngine` | `mlx_audio` | Whisper、NeMo |
| **TTS** (文字转语音) | `TTSEngine` | `mlx_audio` | LFM2、CSM |
| **STS** (语音转语音) | `STSEngine` | `mlx_audio` | LFM2-Audio |
| **DFlash** (Block Diffusion) | `DFlashEngine` | `dflash_mlx` | 推测解码加速 |

> ⚠️ **注意**：oMLX **不直接支持 YOLO**（计算机视觉目标检测模型）。原因见 [十二节](#十二为什么-omlx-不支持-yolo)。

---

## 二、整体架构总览

### 2.1 三层架构

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1: HTTP API 路由层  (FastAPI)                                  │
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐         │
│ │ /v1/chat   │ │ /v1/messages│ │ /v1/embed  │ │ /v1/rerank │         │
│ │ /v1/audio  │ │ /v1/audio/  │ │ /admin     │ │ /health    │         │
│ │ /responses │ │ transcriptions│ │            │ │            │         │
│ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘         │
│       │              │              │              │                │
│       └──────────────┴──────────────┴──────────────┘                │
│                            ↓                                        │
│              ┌─────────────────────────────┐                        │
│              │  api/adapters/              │ (格式转换)              │
│              │  openai / anthropic         │                        │
│              └─────────────┬───────────────┘                        │
└────────────────────────────┼────────────────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────────────┐
│ Layer 2: EnginePool 调度层                                          │
│ ┌─────────────────────────────────────────────────────────────┐   │
│ │  EnginePool                                                   │   │
│ │    ├── LRU 驱逐 (LRU eviction)                                │   │
│ │    ├── Pin / TTL (admin 面板设置)                              │   │
│ │    ├── Pre-load 内存检查 (Pre-flight eviction)                │   │
│    ├── Lease 机制 (in_use count)                                │   │
│    ├── Alias / Profile 解析                                    │   │
│    └── 默认模型 fallback                                        │   │
│                                                                   │   │
│    EngineEntry[] = { llama-3b, qwen3-vl, bge-m3, whisper-large }   │
│         ↓         ↓           ↓           ↓                          │
│    Engine       Engine       Engine       Engine                     │
└────┼─────────────┼───────────┼───────────┼───────────────────────────┘
     ↓             ↓           ↓           ↓
┌────────────────────────────────────────────────────────────────────┐
│ Layer 3: Engine 执行层  (8 类引擎)                                  │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │ BatchedEngine   │  │ VLMBatchedEngine│  │ EmbeddingEngine │    │
│  │ mlx_lm          │  │ mlx_vlm         │  │ mlx_embeddings  │    │
│  │ AsyncEngineCore │  │ AsyncEngineCore │  │ 简单 forward    │    │
│  │ Scheduler       │  │ Scheduler       │  │ batch_size=N    │    │
│  │ Prefix Cache    │  │ Boundary Snap   │  │ no streaming    │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │ RerankerEngine  │  │ STTEngine       │  │ TTSEngine       │    │
│  │ SequenceClassif │  │ mlx_audio.stt   │  │ mlx_audio.tts   │    │
│  │ yes/no logits   │  │ Whisper/NeMo    │  │ LFM2/CSM        │    │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘    │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐                          │
│  │ STSEngine       │  │ DFlashEngine    │                          │
│  │ mlx_audio.sts   │  │ dflash_mlx      │                          │
│  │ LFM2-Audio      │  │ 推测解码         │                          │
│  └─────────────────┘  └─────────────────┘                          │
└─────────────────────────┬──────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────────┐
│ Layer 0: 硬件抽象层  (MLX)                                          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Apple MLX Framework                                          │   │
│  │   - mlx.core (张量)                                            │   │
│  │   - mlx.nn (Module)                                           │   │
│  │   - mx.fast.scaled_dot_product_attention (Flash Attention)    │   │
│  │   - mx.quantize / mx.quantized_matmul (量化)                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ libmlx.dylib (C++)                                            │   │
│  │   - Metal Command Queue                                       │   │
│  │   - Compile Cache (thread_local)                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Apple Metal GPU (M-series Unified Memory)                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流总览

```
HTTP Request
  → API Adapter (格式转换)
    → EnginePool.get_engine() (调度)
      → Engine.start() (如果未加载)
        → Model loading (mlx_lm.load / mlx_vlm.load / mlx_audio.load)
      → Engine.execute() (推理)
        → Model forward pass
        → Output processing
      → SSE / multipart streaming (流式)
    → Engine release (lease 释放)
  → HTTP Response
```

---

## 三、阶段 1：模型发现与分类

### 3.1 触发时机

- **服务器启动时**：扫描 `--model-dir` 目录
- **Admin UI 添加模型时**：手动指定路径
- **下载完成后**：HF Downloader 下载到指定路径

### 3.2 发现流程

```python
# omlx/model_discovery.py:discovers_models()
def discover_models(model_dir: Path) -> list[DiscoveredModel]:
    """
    Recursively scan model_dir for valid model subdirectories.
    
    Each subdirectory with config.json + *.safetensors is a model.
    Two-level organization (mlx-community/model-name/) is also supported.
    """
    discovered = []
    for path in walk(model_dir):
        if not is_valid_model(path):  # has config.json + safetensors
            continue
        if _is_unsupported_model(path):
            continue
        
        # 分类：核心调用
        model_type = detect_model_type(path)
        engine_type = derive_engine_type(model_type)
        
        # 估算内存
        estimated_size = estimate_size(path)
        
        discovered.append(DiscoveredModel(
            model_id=path.name,
            model_path=str(path),
            model_type=model_type,    # "llm"/"vlm"/"embedding"/...
            engine_type=engine_type,  # "batched"/"vlm"/"embedding"/...
            estimated_size=estimated_size,
            ...
        ))
    return discovered
```

### 3.3 detect_model_type 决策树

```python
# omlx/model_discovery.py:detect_model_type()
def detect_model_type(model_path: Path) -> ModelType:
    """
    决策顺序（重要！）：
    1. architectures 包含 SequenceClassification → "reranker"
    2. CausalLM + 目录名包含 "rerank" → "reranker"  (Qwen3-Reranker)
    3. CausalLM + 目录名包含 "embed" → "embedding"  (Qwen3-Embedding)
    4. multimodal reranker/embedding (VLM-based)
    5. sentence-transformers pipeline → "embedding"
    6. architectures 包含 *ForMaskedLM / BertLM → "embedding"
    7. model_type 字段匹配 EMBEDDING_MODEL_TYPES → "embedding"
    8. model_type 是 VLM_NATIVE_TEXT_MODEL_TYPES → "vlm"
    9. 视觉子配置存在 (vision_config/vit_config/mm_vision_tower) → "vlm"
    10. audio_config / tts_config / stt_config 存在 → "audio_*"
    11. 否则 → "llm"
    """
```

### 3.4 模型类型常量

```python
# omlx/model_discovery.py:28
ModelType = Literal[
    "llm",
    "vlm",
    "embedding",
    "reranker",
    "audio_stt",   # 语音转文字 (Whisper, NeMo)
    "audio_tts",   # 文字转语音 (LFM2-TTS, CSM)
    "audio_sts",   # 语音转语音 (LFM2-Audio)
]
```

### 3.5 Engine Type 映射

```python
# omlx/model_discovery.py:282
@dataclass
class DiscoveredModel:
    model_type: ModelType        # 用于 routing
    engine_type: EngineType      # 用于 EnginePool 实例化
```

```python
# 映射关系
"llm"        → "batched"   → BatchedEngine
"vlm"        → "vlm"       → VLMBatchedEngine
"embedding"  → "embedding" → EmbeddingEngine
"reranker"   → "reranker"  → RerankerEngine
"audio_stt"  → "audio_stt" → STTEngine
"audio_tts"  → "audio_tts" → TTSEngine
"audio_sts"  → "audio_sts" → STSEngine
```

---

## 四、阶段 2：HTTP 路由与 API 适配

### 4.1 路由总览

```python
# omlx/server.py 中的 endpoint（已 grep 确认）
@app.post("/v1/chat/completions")         # OpenAI chat
@app.post("/v1/completions")              # OpenAI completion
@app.post("/v1/embeddings")               # OpenAI embedding
@app.post("/v1/rerank")                   # OpenAI rerank
@app.post("/v1/messages")                 # Anthropic Messages
@app.post("/v1/messages/count_tokens")    # Anthropic token count
@app.post("/v1/responses")                # OpenAI Responses
@app.get("/v1/responses/{response_id}")
@app.get("/v1/models")                    # 列出模型
@app.post("/v1/audio/transcriptions")     # Whisper API
@app.post("/v1/audio/translations")       # Whisper API
@app.post("/v1/audio/speech")             # TTS API
@app.get("/health")
@app.get("/api/status")                   # Admin dashboard
```

### 4.2 LLM 聊天请求全链路（最复杂场景）

```
HTTP POST /v1/chat/completions
  Headers: Authorization: Bearer sk-xxx, Content-Type: application/json
  Body: {
    "model": "llama-3b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": true
  }
```

### 4.3 API Adapter 转换

```python
# omlx/api/adapters/openai.py
class OpenAIAdapter(BaseAdapter):
    def parse_request(self, request: ChatCompletionRequest) -> InternalRequest:
        """OpenAI ChatCompletionRequest → InternalRequest"""
        # 转换 messages 格式
        internal_messages = [
            InternalMessage(role=msg.role, content=extract_text(msg.content))
            for msg in request.messages
        ]
        # 转换 tools
        tools = convert_tools_for_template(request.tools)
        return InternalRequest(
            messages=internal_messages,
            tools=tools,
            sampling_params=...,
        )
```

### 4.4 Anthropic Adapter 转换

```python
# omlx/api/adapters/anthropic.py
class AnthropicAdapter(BaseAdapter):
    def parse_request(self, request: MessagesRequest) -> InternalRequest:
        """Anthropic MessagesRequest → InternalRequest"""
        # 1. 提取 system message（Anthropic 单独字段）
        system = request.system
        # 2. 转换 content blocks（text/image/tool_use/tool_result）
        messages = convert_anthropic_to_internal(request)
        # 3. 转换 tools（Anthropic input_schema → OpenAI parameters）
        tools = convert_anthropic_tools_to_internal(request.tools)
        return InternalRequest(messages=..., tools=..., sampling_params=...)
```

### 4.5 SSE 协议输出

```python
# omlx/api/adapters/openai.py 流式响应
async def stream_response(generator):
    async for chunk in generator:
        sse_event = {
            "id": chunk.id,
            "object": "chat.completion.chunk",
            "created": chunk.created,
            "model": chunk.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk.text},
                    "finish_reason": chunk.finish_reason,
                }
            ],
        }
        yield f"data: {json.dumps(sse_event)}\n\n"
    
    # 最终 chunk
    yield "data: [DONE]\n\n"
```

---

## 五、阶段 3：EnginePool 多模型调度

### 5.1 核心数据结构

```python
# omlx/engine_pool.py:54
class EngineEntry:
    """每个模型一份状态。"""
    model_id: str                      # 目录名（如 "llama-3b"）
    model_path: str                    # 绝对路径
    model_type: ModelType              # "llm"/"vlm"/"embedding"/...
    engine_type: EngineType            # "batched"/"vlm"/"embedding"/...
    estimated_size: int                # safetensors 估算大小（bytes）
    actual_size: int | None = None     # 加载后实测
    
    # 运行时状态
    engine: BaseEngine | None = None   # 实例（未加载时 None）
    last_access: float = 0.0           # LRU 时间戳
    is_loading: bool = False           # 防止并发加载
    is_pinned: bool = False            # 不被 LRU 驱逐
    in_use: int = 0                    # lease 引用计数
    abort_requested: bool = False      # 硬压力下请求 abort
    
    # Per-model 配置
    thinking_default: bool | None
    model_context_length: int | None

class EnginePool:
    def __init__(self):
        self._entries: dict[str, EngineEntry] = {}  # model_id → entry
        self._lock = asyncio.Lock()                  # 异步锁
        self._current_model_memory = 0               # 当前已加载模型总内存
```

### 5.2 get_engine 主入口

```python
# omlx/server.py:839
async def get_engine(
    model_id: str | None = None,
    engine_type: EngineType = EngineType.LLM,
    _lease: bool = False,
) -> Union[BaseEngine, EmbeddingEngine, RerankerEngine]:
    """
    统一引擎获取入口。
    
    流程：
    1. 解析 alias / profile (e.g. "qwen3-8b:thinking")
    2. 默认模型 fallback
    3. EnginePool.get_engine()  ← 实际加载逻辑
    4. 返回 engine 实例
    """
```

### 5.3 EnginePool.get_engine 详细流程

```python
# omlx/engine_pool.py
async def get_engine(self, model_id: str, *, _lease=False, runtime_settings=None):
    async with self._lock:
        entry = self._entries.get(model_id)
        
        # 1. 检查是否已加载
        if entry.engine is not None:
            entry.last_access = time.time()
            if _lease:
                entry.in_use += 1
            return entry.engine
        
        # 2. 未加载 → 触发加载流程
        return await self._load_engine(entry, _lease=_lease, runtime_settings=runtime_settings)
```

### 5.4 _load_engine 完整流程

```python
async def _load_engine(self, entry, _lease=False, runtime_settings=None):
    # 1. Pre-load 内存检查（关键！避免 OOM）
    await self._pre_load_memory_check(entry)
    
    # 2. 标记加载中（防止并发加载）
    entry.is_loading = True
    entry.loading_started_at = time.time()
    
    try:
        # 3. 选择 Engine 实现（基于 entry.engine_type）
        engine = await self._create_engine_instance(entry, runtime_settings)
        
        # 4. 加载模型（异步、在 MLX executor 上）
        await engine.start()
        
        # 5. 更新 entry
        entry.engine = engine
        entry.actual_size = compute_actual_size(entry)
        self._current_model_memory += entry.actual_size
        
        # 6. 触发 LRU 检查
        await self._maybe_evict_for_memory()
        
        # 7. Lease
        if _lease:
            entry.in_use += 1
        
        return engine
    finally:
        entry.is_loading = False
```

### 5.5 _pre_load_memory_check（避免 OOM）

```python
async def _pre_load_memory_check(self, entry):
    """
    加载前估算 + 必要时预驱逐其他模型。
    
    流程：
    1. 估算当前可用内存
    2. 估算新模型所需内存
    3. 如果不够 → 找到 LRU 候选（最久未用 + 非 pinned + 无 in_use）
    4. 卸载候选 → 重新估算
    5. 仍不够 → 抛 InsufficientMemoryError / ModelTooLargeError
    """
    ceiling = self._current_ceiling()
    needed = entry.estimated_size
    
    while True:
        available = ceiling - self._current_model_memory
        if available >= needed:
            return  # 足够
        
        candidate = self._find_lru_eviction_candidate()
        if candidate is None:
            raise InsufficientMemoryError(...)
        
        # 预驱逐
        await self._unload_engine(candidate.model_id)
```

### 5.6 EngineType 路由

```python
# omlx/engine_pool.py:1322
async def _create_engine_instance(self, entry, runtime_settings):
    effective_type = entry.engine_type
    
    # 检查是否有 DFlash 替代
    if model_settings is not None and model_settings.dflash_enabled:
        # 尝试用 DFlashEngine
        ...
    
    # 创建 Engine（按 engine_type 分支）
    if effective_type == "embedding":
        return EmbeddingEngine(
            model_name=entry.model_path,
            trust_remote_code=trc,
            scheduler_config=self._scheduler_config,
        )
    elif effective_type == "reranker":
        return RerankerEngine(model_name=entry.model_path, ...)
    elif effective_type == "vlm":
        return VLMBatchedEngine(
            model_name=entry.model_path,
            trust_remote_code=trc,
            scheduler_config=self._scheduler_config,
            model_settings=model_settings,
            prefill_eviction_callback=prefill_eviction_callback,
        )
    elif entry.engine_type == "audio_stt":
        return STTEngine(model_name=entry.model_path)
    elif entry.engine_type == "audio_tts":
        return TTSEngine(model_name=entry.model_path)
    elif entry.engine_type == "audio_sts":
        return STSEngine(model_name=entry.model_path, config_model_type=...)
    else:  # "batched" or "simple"
        return BatchedEngine(
            model_name=entry.model_path,
            trust_remote_code=trc,
            scheduler_config=self._scheduler_config,
            model_settings=model_settings,
            prefill_eviction_callback=prefill_eviction_callback,
        )
```

---

## 六、阶段 4：Engine 创建与加载

### 6.1 通用模式

每种 Engine 都遵循 `start()` 异步加载 + `stop()` 异步卸载的模式。

```python
# 通用加载模式
async def start(self):
    if self._model is not None:
        return  # 已加载
    
    logger.info(f"Starting {self.__class__.__name__}: {self._model_name}")
    
    # 关键：Model loading 在 MLX executor 上运行
    # 原因：避免 Metal command buffer races (issue #85)
    loop = asyncio.get_running_loop()
    self._model = await loop.run_in_executor(
        get_mlx_executor(),           # ← 全局单线程 MLX executor
        lambda: load_model(self._model_name, ...)  # 同步加载
    )
    
    logger.info(f"{self.__class__.__name__} started: {self._model_name}")
```

### 6.2 LLM 加载（BatchedEngine）

```python
# omlx/engine/batched.py:228
async def start(self):
    if self._loaded:
        return
    
    import asyncio
    from mlx_lm import load
    from ..engine_core import AsyncEngineCore, EngineConfig
    from ..scheduler import SchedulerConfig
    from ..utils.model_loading import (
        maybe_apply_pre_load_patches,        # ← DeepSeek V4 等 monkey-patch
        maybe_load_custom_quantization,      # ← paroquant 等
    )
    
    # Pre-load monkey-patches（必要时向 sys.modules 注入新模型）
    maybe_apply_pre_load_patches(self._model_name, model_settings=self._model_settings)
    
    def _load_model_sync():
        custom_loaded = maybe_load_custom_quantization(self._model_name, is_vlm=False)
        if custom_loaded is not None:
            model, processor = custom_loaded
            return model, getattr(processor, "tokenizer", processor)
        
        return load(                      # ← mlx_lm.load
            self._model_name,
            tokenizer_config=tokenizer_config,
            trust_remote_code=self._trust_remote_code,
        )
    
    loop = asyncio.get_running_loop()
    self._model, self._tokenizer = await loop.run_in_executor(
        get_mlx_executor(), _load_model_sync
    )
    
    # Post-load transforms（如 IndexCache for DSA）
    apply_post_load_transforms(self._model, self._model_settings)
    
    # Materialize lazy state（避免跨线程 stream 错误）
    materialize_lazy_state(self._model)
    
    # 构造 AsyncEngineCore
    self._engine = AsyncEngineCore(
        model=self._model,
        tokenizer=self._tokenizer,
        config=EngineConfig(...),
    )
    
    self._loaded = True
    await self._engine.start()  # 启动 asyncio 推理循环
```

### 6.3 VLM 加载（VLMBatchedEngine）

```python
# omlx/engine/vlm.py:871
class VLMBatchedEngine(BaseEngine):
    async def start(self):
        if self._loaded:
            return
        
        from mlx_vlm import load as mlx_vlm_load
        
        def _load_model_sync():
            return mlx_vlm_load(    # ← mlx_vlm.load
                self._model_name,
                processor_kwargs=...,
            )
        
        loop = asyncio.get_running_loop()
        self._vlm_model, self._processor = await loop.run_in_executor(
            get_mlx_executor(), _load_model_sync
        )
        
        # VLM 特有：detokenizer runtime
        _attach_vlm_tokenizer_runtime(self._tokenizer, model_path, eos_token_id)
        
        # 构造 Adapter（用于注入 vision embeddings）
        self._adapter = VLMModelAdapter(self._vlm_model, self._processor)
        
        # AsyncEngineCore（同 LLM）
        self._engine = AsyncEngineCore(...)
```

### 6.4 Embedding 加载（EmbeddingEngine）

```python
# omlx/engine/embedding.py:100
async def start(self):
    if self._model is not None:
        return
    
    self._model = MLXEmbeddingModel(self._model_name, trust_remote_code=...)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(get_mlx_executor(), self._model.load)
```

```python
# omlx/models/embedding.py:51
class MLXEmbeddingModel:
    def load(self):
        """
        加载 HF AutoModel + AutoTokenizer，转换为 MLX。
        包装 mlx-embeddings 的能力。
        """
        from mlx_embeddings import load as mlx_emb_load
        
        # mlx-embeddings 是 MLX 原生 embedding 模型库
        # 支持 BGE-M3、ModernBERT 等
        self.model, self.processor = mlx_emb_load(self.model_path)
        self._loaded = True
```

### 6.5 Reranker 加载（RerankerEngine）

```python
# omlx/engine/reranker.py:79
async def start(self):
    if self._model is not None:
        return
    
    self._model = MLXRerankerModel(self._model_name, ...)
    await loop.run_in_executor(get_mlx_executor(), self._model.load)
```

```python
# omlx/models/reranker.py（自实现）
class MLXRerankerModel:
    """
    两种模式：
    1. SequenceClassification 模型（BERT-style）：直接用 logits
    2. CausalLM Reranker（Qwen3-Reranker）：用 yes/no logits
    
    通过架构识别 + 目录名启发式区分。
    """
    def load(self):
        if self._is_causal_lm_reranker:
            # 用 mlx_lm.load 加载，然后自定义 forward
            from mlx_lm import load
            self.model, self.tokenizer = load(self.model_path)
        else:
            # 用 mlx_embeddings / HF AutoModel 加载
            self.model = AutoModelForSequenceClassification.from_pretrained(...)
```

### 6.6 STT 加载（STTEngine）

```python
# omlx/engine/stt.py:145
async def start(self):
    if self._model is not None:
        return
    
    from mlx_audio.stt.utils import load_model as _load_model
    
    def _load_sync():
        return _load_model(self._model_name)
    
    loop = asyncio.get_running_loop()
    self._model = await loop.run_in_executor(get_mlx_executor(), _load_sync)
```

### 6.7 TTS / STS 加载

```python
# omlx/engine/tts.py:50 / omlx/engine/sts.py
async def start(self):
    from mlx_audio.tts.utils import load_model as _load_tts_model
    
    self._model = await loop.run_in_executor(
        get_mlx_executor(),
        lambda: _load_tts_model(self._model_name)
    )
```

### 6.8 加载过程共性

| 步骤 | 说明 |
|------|------|
| 1. lazy import | 在 start() 内 import 对应库的 load 函数（避免未安装时的 import error） |
| 2. loop.run_in_executor | 在全局 MLX executor 线程上加载（避免 Metal 竞争） |
| 3. return loaded model | 通常返回 (model, tokenizer/processor) tuple |
| 4. materialize lazy state | 物化 MLX Array 树（避免跨线程 stream 错误） |

---

## 七、阶段 5：推理执行（按引擎类型分支）

### 7.1 LLM 推理（BatchedEngine）— 最复杂

```python
# omlx/engine/batched.py
async def stream_chat(self, messages, max_tokens=256, temperature=0.7, ..., tools=None):
    # 1. Apply chat template（生成 prompt string）
    prompt = self._tokenizer.apply_chat_template(messages, tools=tools, ...)
    
    # 2. Preflight 内存检查
    await self.preflight_chat(messages, tools=tools, request_id=request_id, ...)
    
    # 3. 调用 AsyncEngineCore.generate()
    async for response in self._engine.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        ...
    ):
        yield response  # 每个 token 一个 SSE 事件
```

**底层走 BatchGenerator**（详见 [200-batch-generator-and-continuous-batching.md](./200-batch-generator-and-continuous-batching.md)）：

```
BatchedEngine.stream_chat
  → AsyncEngineCore.generate
    → Scheduler.add_request (FCFS 队列)
    → Scheduler.step()
      → mlx_lm.BatchGenerator.step()
        → 模型 forward (mx.array)
        → mx.fast.scaled_dot_product_attention
        → Sampling
        → 输出 1 token
      → StoreCache (RAM hot tier + SSD demote)
    → SSE 发送 token
```

### 7.2 VLM 推理（VLMBatchedEngine）— 多图像 + 边界快照

```python
# omlx/engine/vlm.py
class VLMBatchedEngine(BaseEngine):
    async def stream_chat(self, messages, images=None, ...):
        # 1. 提取 images from messages
        #    (OpenAI 格式: {"type": "image_url", "image_url": {"url": "data:..."}})
        images = extract_images_from_messages(messages)
        
        # 2. VLM processor 处理（图像编码 → patch embeddings）
        #    mlx_vlm.process_vision_info(images) → pixel_values
        pixel_values = self._processor.image_processor(images)
        
        # 3. Apply chat template with image tokens
        prompt = self._tokenizer.apply_chat_template(messages, images=images)
        
        # 4. BoundarySnapshot lookup
        #    如果同一图像之前用过，复用其 KV snapshot
        cached_snapshot = self._boundary_snapshot_store.lookup(images_hash)
        
        # 5. Preflight + 调 BatchedEngine 类似流程
        async for token in self._engine.generate(prompt, ...):
            yield token
```

**VLM 特有的 BoundarySnapshotStore**：

```
第一次: 用户上传图像 A → 提取 patch embeddings → 注入 prompt
        → prefill → KV cache → 在图像边界位置保存 snapshot
第二次: 同一图像 A → 检测 hash 命中 → 跳过 prefill → 直接 decode
        → 大幅加速！
```

### 7.3 Embedding 推理（EmbeddingEngine）— 简单

```python
# omlx/engine/embedding.py:135
async def embed(self, texts, max_length=None, padding=True, truncation=True):
    if self._model is None:
        raise RuntimeError("Engine not started.")
    
    model = self._model
    input_items = [texts] if isinstance(texts, str) else list(texts)
    
    batch_size = self._batch_size  # 默认 32
    embeddings = []
    total_tokens = 0
    
    for start in range(0, len(input_items), batch_size):
        batch = input_items[start:start + batch_size]
        
        def _embed_sync():
            return model.embed(inputs=batch, max_length=max_length, ...)
        
        output = await loop.run_in_executor(get_mlx_executor(), _embed_sync)
        embeddings.extend(output.embeddings)
        total_tokens += output.total_tokens
    
    return EmbeddingOutput(embeddings=embeddings, total_tokens=total_tokens, ...)
```

**底层**（`omlx/models/embedding.py:489`）：

```python
class MLXEmbeddingModel:
    def embed(self, inputs, max_length=None, ...):
        # 1. Tokenize（HF tokenizer）
        encoded = self.processor(input_texts, padding=True, truncation=True, ...)
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])
        
        # 2. Forward（MLX 模型）
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 3. Pooling（CLS token / mean pooling）
        embeddings_array = self._extract_embeddings_array(outputs)
        
        return EmbeddingOutput(
            embeddings=embeddings_array.tolist(),  # 转 Python list
            total_tokens=int(attention_mask.sum()),
            dimensions=embeddings_array.shape[-1],
        )
```

**特点**：
- **无 streaming**：单次 forward 返回所有结果
- **无 chat 接口**：没有 chat template 概念
- **批处理**：一次可处理多个文本（默认 batch_size=32）

### 7.4 Reranker 推理（RerankerEngine）— 更简单

```python
# omlx/engine/reranker.py:107
async def rerank(self, query, documents, top_n=None):
    if self._model is None:
        raise RuntimeError("Engine not started.")
    
    model = self._model
    loop = asyncio.get_running_loop()
    
    def _rerank_sync():
        try:
            return model.rerank(
                query=query,
                documents=documents,
                top_n=top_n,
            )
        finally:
            mx.synchronize()
            mx.clear_cache()
    
    return await loop.run_in_executor(get_mlx_executor(), _rerank_sync)
```

**底层两种模式**：

```python
# omlx/models/reranker.py
class MLXRerankerModel:
    def rerank(self, query, documents, top_n=None):
        if self._is_causal_lm_reranker:
            return self._causal_lm_rerank(query, documents)
        else:
            return self._classification_rerank(query, documents)
    
    def _classification_rerank(self, query, documents):
        # SequenceClassification 模型（BERT-style）
        # 1. 拼接 query + document
        # 2. Forward → logits [batch, num_labels]
        # 3. 取 label=1 的分数作为 relevance score
        ...
    
    def _causal_lm_rerank(self, query, documents):
        # Qwen3-Reranker 模式
        # 1. 提示模型："Is this document relevant? Output yes/no"
        # 2. Forward → logits at "yes"/"no" token positions
        # 3. P(yes) / (P(yes) + P(no)) 作为 relevance score
        ...
```

### 7.5 ASR 推理（STTEngine）

```python
# omlx/engine/stt.py:225
async def transcribe(self, audio_path, language=None, ...):
    model = self._model
    loop = asyncio.get_running_loop()
    
    def _transcribe_sync():
        # mlx_audio.stt 提供 transcribe 接口
        return model.transcribe(audio_path, language=language, ...)
    
    return await loop.run_in_executor(get_mlx_executor(), _transcribe_sync)
```

**流程**：

```
音频文件 (WAV/MP3)
  → AudioFeatureExtractor (log-mel spectrogram)
  → Encoder forward (Whisper encoder)
  → Decoder (autoregressive text generation)
  → Output: text + timestamps + language
```

### 7.6 TTS 推理（TTSEngine）

```python
# omlx/engine/tts.py
async def synthesize(self, text, voice=None, ...):
    loop = asyncio.get_running_loop()
    
    def _synthesize_sync():
        return self._model.generate(text=text, voice=voice, ...)
    
    audio_array, sample_rate = await loop.run_in_executor(
        get_mlx_executor(), _synthesize_sync
    )
    
    # 编码为 WAV bytes 返回
    return encode_wav(audio_array, sample_rate)
```

**流程**：

```
text
  → TextTokenizer (BPE/SentencePiece)
  → ProsodyPredictor (LFM2-TTS)
  → AcousticModel / CodecModel (MLX forward)
  → Audio decoder (vocoder)
  → Output: audio waveform (numpy array / bytes)
```

### 7.7 STS 推理（STSEngine）

```
input audio (speech)
  → STT part (转文本)
  → LLM/Embedding part (理解 + 生成响应文本)
  → TTS part (合成语音)
  → output audio (response speech)
```

→ **组合多个模型**，是更复杂的 pipeline。

### 7.8 DFlash 推测解码（DFlashEngine）

```python
# omlx/engine/dflash.py
class DFlashEngine(BaseEngine):
    """
    Block diffusion speculative decoding on Apple Silicon.
    
    思想：用一个小的"draft"模型生成 k 个候选 token，
         然后用主模型并行验证（一次 forward）。
    """
    async def generate(self, prompt, ...):
        # 1. 主模型生成 1 个 token
        # 2. Draft 模型生成 k 个候选
        # 3. 主模型一次 forward 验证 k 个候选
        # 4. 接受正确部分，拒绝错误部分
        ...
```

---

## 八、阶段 6：流式输出（SSE / WebSocket / multipart）

### 8.1 LLM 流式响应（SSE）

```python
# omlx/server.py:3030
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, ...):
    engine = await get_engine_for_model(request.model, lease=lease)
    
    # 流式响应包装
    async def generate_sse():
        async for chunk in engine.stream_chat(messages, ...):
            yield f"data: {json.dumps({
                'id': chunk.id,
                'object': 'chat.completion.chunk',
                'choices': [{
                    'delta': {'content': chunk.text},
                    'finish_reason': chunk.finish_reason,
                }],
            })}\n\n"
        
        # SSE keep-alive（防止长 prefill 期间 read timeout）
        # 见 sse-keepalive-mode CLI 参数
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate_sse(), media_type="text/event-stream")
```

### 8.2 SSE Keep-alive 机制

```python
# 关键设计：避免 Claude Code 等长 prefill 时 SSE read timeout
# chunk 模式：发出符合协议的无操作事件，兼容严格客户端（OpenClaw / WorkBuddy）
# comment 模式：传统 SSE ': keep-alive' 注释
# off 模式：禁用

# 在 omlx/cli.py 解析
serve_parser.add_argument(
    "--sse-keepalive-mode",
    type=str,
    choices=["chunk", "comment", "off"],
    default="chunk",
)
```

### 8.3 Anthropic SSE 格式

```python
# omlx/api/adapters/anthropic.py:create_text_delta_event
def create_text_delta_event(index: int, text: str) -> dict:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }

# event 流：
event: message_start
data: {"type": "message_start", ...}

event: content_block_start
data: {"type": "content_block_start", "index": 0, ...}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}

event: message_stop
data: {"type": "message_stop"}
```

### 8.4 Audio API 输出（multipart）

```python
# omlx/api/audio_routes.py
@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    engine = await get_engine_for_model(request.model, lease=lease)
    
    audio_bytes = await engine.synthesize(text=request.input, voice=request.voice)
    
    # 返回 audio/wav 或 audio/mpeg
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )

@app.post("/v1/audio/transcriptions")
async def create_transcription(request: TranscriptionsRequest, file: UploadFile):
    engine = await get_engine_for_model(request.model, lease=lease)
    
    # 1. 保存上传文件
    audio_path = await save_upload(file)
    
    # 2. STT 推理
    result = await engine.transcribe(audio_path=audio_path, language=request.language)
    
    # 3. 返回 JSON 或 text/plain 或 srt/vtt
    return {"text": result.text}
```

### 8.5 流式 vs 非流式对比

| 引擎 | 流式 | 非流式 | 备注 |
|------|------|--------|------|
| LLM / VLM | ✅ SSE | ✅ JSON | 主用流式 |
| Embedding | ❌ | ✅ JSON | 简单 forward |
| Reranker | ❌ | ✅ JSON | 简单 forward |
| STT | 部分 | ✅ JSON/text/srt/vtt | 部分模型支持 streaming |
| TTS | 部分 | ✅ audio/wav | 部分模型支持 streaming |
| STS | ❌ | ✅ audio/wav | 组合复杂 |

---

## 九、阶段 7：缓存存储与异步清理

### 9.1 LLM 完成后的缓存存储

```
LLM request 完成
  → Scheduler._cleanup_finished(uid)
    → 调用 cache.store_cache(uid, tokens, kv_cache_data)
      → BlockAwarePrefixCache.store_cache()
        → 提取每个 block 的 KV tensor
        → 写入 HotCache (RAM)
        → 异步 demote 到 PagedSSDCache (SSD)
```

### 9.2 异步 Demote 队列

```python
# omlx/cache/paged_ssd_cache.py
class PagedSSDCacheManager:
    """
    后台 writer thread 持续消费 pending_writes 队列。
    """
    def __init__(self):
        self._pending_writes = queue.Queue(maxsize=compute_max_pending_writes())
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
    
    def save_block(self, block_hash, kv_tensors):
        # 1. 提取 bytes (copy 到 host buffer)
        bytes_payload = _extract_tensor_bytes(kv_tensors)
        
        # 2. 放入 pending 队列（非阻塞 + 超时）
        try:
            self._pending_writes.put(bytes_payload, timeout=1.0)
        except queue.Full:
            logger.warning("SSD cache write queue full; dropping")
    
    def _writer_loop(self):
        """后台线程：从队列取数据 → 写 SSD → 索引到 LRU"""
        while True:
            payload = self._pending_writes.get()
            safetensors.save_file(payload.tensors, payload.path, metadata={...})
            # 更新 LRU 索引、统计等
```

### 9.3 异步清理 + 边界快照

```python
# omlx/scheduler.py
def _drain_pending_async_removes(self) -> bool:
    """
    异步 store_cache 完成后，调用 BatchGenerator.remove(uid)
    让 BatchGenerator 释放该序列的资源。
    """
    for completed in self._async_store_cache_results:
        if completed.success:
            self.batch_generator.remove(completed.uid)
```

---

## 十、阶段 8：内存管理与 LRU 驱逐

### 10.1 三层内存监控

```python
# omlx/process_memory_enforcer.py
class ProcessMemoryEnforcer:
    """
    后台 asyncio 任务，每 1s 轮询：
    - 主机 RSS（psutil）
    - MLX active memory（mx.get_active_memory）
    - HotCache bytes
    - Paged SSD 元数据
    
    计算 pressure level：
    - ok (current < 90% of ceiling)
    - soft (90-95%) → LRU evict non-pinned + pause admission
    - hard (>=95%) → abort in-flight + abort loading
    """
```

### 10.2 LRU 驱逐算法

```python
# omlx/engine_pool.py
def _find_lru_eviction_candidate(self) -> EngineEntry | None:
    """
    找最久未用 + 可驱逐的 entry：
    - 不是 pinned
    - in_use == 0 (没有活跃 lease)
    - is_loading == False (没在加载中)
    
    排除规则：
    - 正在被请求使用的
    - pinned 的
    - 加载中的
    """
    candidates = [
        e for e in self._entries.values()
        if not e.is_pinned
        and e.in_use == 0
        and not e.is_loading
        and e.engine is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda e: e.last_access)
```

### 10.3 卸载流程

```python
async def _unload_engine(self, model_id: str):
    entry = self._entries[model_id]
    engine = entry.engine
    
    # 1. 等待所有活跃请求完成（或 abort）
    #    lease 机制保证 in_use > 0 时不卸载
    while entry.in_use > 0:
        await asyncio.sleep(0.05)
    
    # 2. 异步 stop engine
    await engine.stop()
    
    # 3. 释放内存
    entry.engine = None
    self._current_model_memory -= entry.actual_size
    
    # 4. 触发 MLX 内存回收
    await loop.run_in_executor(
        get_mlx_executor(),
        lambda: (mx.synchronize(), mx.clear_cache())
    )
```

---

## 十一、各类模型的差异点对照

| 维度 | LLM | VLM | Embedding | Reranker | STT | TTS |
|------|-----|-----|-----------|----------|-----|-----|
| **Engine 类** | `BatchedEngine` | `VLMBatchedEngine` | `EmbeddingEngine` | `RerankerEngine` | `STTEngine` | `TTSEngine` |
| **底层库** | `mlx_lm` | `mlx_vlm` | `mlx_embeddings` | 自实现 + mlx | `mlx_audio.stt` | `mlx_audio.tts` |
| **流式** | ✅ SSE | ✅ SSE | ❌ | ❌ | 部分 | 部分 |
| **聊天接口** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Prefix Cache** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Tool Calling** | ✅ 7+ 格式 | ✅ | ❌ | ❌ | ❌ | ❌ |
| **结构化输出** | ✅ JSON schema | ✅ | ❌ | ❌ | ❌ | ❌ |
| **并发批处理** | ✅ BatchGenerator | ✅ | ❌ | ❌ | ❌ | ❌ |
| **输入** | text | text + images | text | query + docs | audio file | text |
| **输出** | tokens | tokens | vectors | scores | text | audio |
| **HTTP 路径** | `/v1/chat/completions` | `/v1/chat/completions` | `/v1/embeddings` | `/v1/rerank` | `/v1/audio/transcriptions` | `/v1/audio/speech` |
| **API 适配** | OpenAI/Anthropic | OpenAI/Anthropic | OpenAI | OpenAI | Whisper API | OpenAI Speech |
| **Lease** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **LRU 驱逐** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Memory Monitor** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **持续时间** | 长（multi-turn） | 长 | 短 | 短 | 中 | 中 |

---

## 十二、为什么 oMLX 不支持 YOLO

### 12.1 根本原因

**oMLX 是 LLM 推理服务器，不是通用 CV 服务器**。

```bash
# 全仓搜索 yolo 结果：0 命中
$ grep -rn "yolo\|YOLO" /home/bianbu/aiws/omlx/omlx/ --include="*.py"
(no output)
```

### 12.2 设计差异

| | LLM | YOLO |
|---|-----|------|
| **任务** | 生成文本序列 | 检测目标边界框 |
| **输出格式** | token sequence | bbox + class + confidence |
| **模型架构** | Transformer (decoder) | CNN (YOLOv8) / Vision Transformer |
| **输入** | text tokens | image tensor |
| **推理特性** | autoregressive（需要 KV cache） | single forward（无 KV cache） |
| **后处理** | detokenize | NMS + bbox decode |
| **API 范式** | streaming chat | image upload + JSON bbox list |

### 12.3 如果要支持 YOLO 需要做的事

```python
# 假设实现
class YOLOEngine(BaseNonStreamingEngine):
    async def start(self):
        # 1. 加载模型
        from ultralytics import YOLO  # 外部依赖
        # 或者用 MLX 实现（mlx-yolo）
        self._model = load_yolo_mlx(self._model_name)
    
    async def detect(self, image_path: bytes, conf_threshold=0.5):
        # 1. 预处理图像
        image = preprocess(image_path)
        
        # 2. Forward
        predictions = self._model(image)
        
        # 3. NMS + bbox decode
        boxes, scores, classes = nms(predictions, conf_threshold)
        
        return DetectionResult(
            boxes=boxes,
            scores=scores,
            classes=classes,
        )
```

### 12.4 类似地不支持

oMLX 也**不直接支持**：
- **Whisper.cpp-style 图像分类**（ImageNet 类）
- **Stable Diffusion 文生图**（有 mlx-stable-diffusion 但 oMLX 未集成）
- **语音合成（MusicGen）**
- **蛋白质结构预测（AlphaFold）**

**理由**：oMLX 专注 **LLM/VLM/Audio** 这条线，没有扩张到通用 CV 模型的计划。

---

## 十三、端到端时序示例

### 13.1 典型 LLM Chat 请求全链路

```
时间(s)   组件                 动作
─────────────────────────────────────────────────────────
0.000     HTTP Client          POST /v1/chat/completions
0.001     FastAPI              路由到 create_chat_completion
0.002     verify_api_key       校验 Bearer token
0.003     OpenAIAdapter        解析 Request → InternalRequest
0.005     get_engine_for_model 调 EnginePool.get_engine("llama-3b")
0.005     EnginePool           检查 entry.engine (None → 加载)
0.006     EnginePool           _pre_load_memory_check
0.007     EnginePool           _load_engine:
                                    - 创建 BatchedEngine
                                    - await engine.start()
0.010     BatchedEngine.start  → loop.run_in_executor(MLX executor)
0.100     MLX executor         mlx_lm.load(model_path)
            ├ 读取 config.json
            ├ 读取 *.safetensors → mx.array
            ├ 构建 nn.Module 树
            └ materialize lazy state
0.500     EnginePool           entry.engine = engine
0.501     EnginePool           _maybe_evict_for_memory (LRU 检查)
0.502     BatchedEngine        preflight_chat (内存预算检查)
0.503     AsyncEngineCore     generate() → Scheduler.add_request
0.504     Scheduler            tokenize(prompt) → token_ids
0.505     Scheduler            prefill_or_raise (内存预算)
0.506     BatchGenerator       insert(token_ids)
0.507     BatchGenerator       prefill (sync MLX forward)
0.700     BatchGenerator       第 1 个 token 生成
0.701     Scheduler            异步返回 SSE chunk
0.702     HTTP                 "data: {json}\n\n" 发送给 client
...        (循环)
1.000     BatchGenerator       第 N 个 token 生成
1.001     Scheduler            EOS 检出 → finish
1.002     Scheduler            store_cache (写 HotCache + 异步 SSD)
1.005     BatchGenerator       remove(uid)
1.010     HTTP                 "data: [DONE]\n\n"
1.011     EnginePool           lease.release()
1.012     HTTP                 关闭连接
─────────────────────────────────────────────────────────
```

### 13.2 多模型并发场景

```
T0:    Client A: POST /v1/chat (llama-3b)
T1:    Client B: POST /v1/embeddings (bge-m3)
T2:    Client C: POST /v1/audio/transcriptions (whisper)

T0.000 EnginePool.get_engine("llama-3b")     → 加载/复用 LLM engine
T1.000 EnginePool.get_engine("bge-m3", EMBEDDING) → 加载 EmbeddingEngine
T2.000 EnginePool.get_engine("whisper", AUDIO_STT) → 加载 STTEngine

T0.500 [LLM executor 线程]    跑 Llama forward
T1.500 [EmbeddingEngine]      跑 bge-m3 forward (单 forward)
T2.500 [STTEngine]            跑 whisper forward + decoder

→ 三个模型并行运行在不同引擎实例上
→ 通过 MLX Stream 隔离避免 Metal 竞争
```

---

## 十四、总结

### 14.1 完整链路一句话

> **从 HTTP Request 到 Metal GPU 执行，oMLX 经历 8 个阶段：模型发现 → HTTP 路由 → API 适配 → EnginePool 调度 → Engine 加载 → 模型 forward → SSE 流式 → 缓存存储 → 内存守护。**

### 14.2 各类模型的核心差异

| 模型类型 | 核心差异点 |
|---------|-----------|
| **LLM** | 最复杂：BatchGenerator + Prefix Cache + Tool Calling + 结构化输出 |
| **VLM** | 图像预处理 + Boundary Snapshot + 多模态 Chat Template |
| **OCR** | VLM 专用子类型，自动提示词优化 |
| **Embedding** | 最简单：单 forward + pooling |
| **Reranker** | 两种模式（Classification / CausalLM yes/no） |
| **STT** | 音频特征提取 + encoder-decoder |
| **TTS** | 文本 tokenizer + prosody + vocoder |
| **STS** | STT + LLM + TTS 组合 |
| **DFlash** | 推测解码，verifier-drafter 协作 |

### 14.3 共同的"骨架"

```
所有 Engine 都遵循：
1. async start() ─ 加载模型（MLX executor 上）
2. async generate/embed/rerank/transcribe/synthesize ─ 推理
3. async stop() ─ 释放内存（MLX executor 上）
4. get_stats() / get_cache_stats() ─ 监控

所有 Engine 都被 EnginePool 管理：
- LRU 驱逐
- Pin / TTL
- Lease 机制
- Pre-load 内存检查
- Alias / Profile

所有 Engine 都被 ProcessMemoryEnforcer 监控：
- RSS + MLX active + HotCache = 总内存
- soft/hard threshold 触发不同动作
```

### 14.4 oMLX 的"宽度"vs"深度"

- **宽度**：8 种模型类型（LLM/VLM/Embedding/Reranker/ASR/TTS/STS/DFlash）
- **深度**：LLM/VLM 走完整 server stack（prefix cache、paged cache、continuous batching、tool calling）
- **简单度**：Embedding/Reranker 是单 forward
- **中等**：ASR/TTS 是 encoder-decoder

→ **oMLX 不是单一模型的优化，而是"多模型协同 + 单一 stack 复用"的工程化实践**。

---

## 附录：相关源码位置

| 文件 | 行数 | 作用 |
|------|------|------|
| `omlx/server.py` | 6569 | FastAPI 路由 + Adapter 集成 |
| `omlx/engine_pool.py` | 1717 | 多模型调度核心 |
| `omlx/model_discovery.py` | 1221 | 模型发现与分类 |
| `omlx/engine/batched.py` | 885 | LLM 引擎 |
| `omlx/engine/vlm.py` | 2583 | VLM 引擎（含 Boundary Snapshot） |
| `omlx/engine/embedding.py` | 200+ | Embedding 引擎 |
| `omlx/engine/reranker.py` | 150+ | Reranker 引擎 |
| `omlx/engine/stt.py` | 250+ | STT 引擎 |
| `omlx/engine/tts.py` | 200+ | TTS 引擎 |
| `omlx/engine/sts.py` | 350+ | STS 引擎 |
| `omlx/engine/dflash.py` | 500+ | DFlash 推测解码 |
| `omlx/engine/base.py` | 491 | BaseEngine / BaseNonStreamingEngine 抽象 |
| `omlx/models/embedding.py` | 700+ | MLXEmbeddingModel 实现 |
| `omlx/models/reranker.py` | - | MLXRerankerModel 实现 |

---

*文档生成时间：基于 omlx 仓库当前 HEAD 分析*