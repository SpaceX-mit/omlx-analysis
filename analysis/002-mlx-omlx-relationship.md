# 002 · MLX 框架与 oMLX 的关系：从 GPU 内核到应用层

> **文档编号**：`analysis/002-mlx-omlx-relationship.md`
> **主题**：MLX (Apple ML 框架) 与 oMLX (本地 LLM 推理服务器) 的集成关系
> **核心问题**：
> 1. MLX 在 oMLX 项目中扮演什么角色？
> 2. oMLX 如何调用 MLX？
> 3. 二者的边界在哪里？
> **前置阅读**：[001-omlx-project-overview.md](./001-omlx-project-overview.md)

> 📁 本文档属于 [analysis/](./README.md) 目录。

---

## 目录

- [一、一句话总结](#一一句话总结)
- [二、MLX 家族层级关系](#二mlx-家族层级关系)
- [三、oMLX 依赖的 MLX 系列包](#三omlx-依赖的-mlx-系列包)
- [四、oMLX 调用 MLX 的七个层次](#四omlx-调用-mlx-的七个层次)
- [五、核心 API 调用清单](#五核心-api-调用清单)
- [六、oMLX 对 MLX 的关键适配](#六omlx-对-mlx-的关键适配)
- [七、为什么 oMLX 不直接用 MLX 而是要包一层](#七为什么-omlx-不直接用-mlx-而是要包一层)
- [八、与 MLX 的边界：oMLX 不做这些事](#八与-mlx-的边界omlx-不做这些事)
- [九、总结](#九总结)

---

## 一、一句话总结

**MLX 是 Apple 官方提供的"GPU 张量计算 + Metal 加速"底层框架**；**oMLX 是一个跑在 MLX 之上的"LLM 推理服务器"**——它不重写模型架构、不重写注意力计算，而是把 `mlx-lm` / `mlx-vlm` 这些 MLX 生态的高层封装嵌入到一个**多模型、连续批处理、KV 缓存分层管理、带 HTTP API 和 Admin UI**的服务器进程里。

类比：
- **MLX ≈ PyTorch**（张量库 + Metal CUDA 内核）
- **mlx-lm / mlx-vlm ≈ HuggingFace Transformers**（预训练模型架构 + 推理工具）
- **oMLX ≈ vLLM / TGI**（生产级推理服务器）

---

## 二、MLX 家族层级关系

```
┌────────────────────────────────────────────────────────────────┐
│ 应用层                                                          │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │                       oMLX                               │   │
│ │  - FastAPI HTTP server                                    │   │
│ │  - EnginePool 多模型调度                                   │   │
│ │  - PagedCacheManager / PagedSSDCacheManager              │   │
│ │  - ProcessMemoryEnforcer                                  │   │
│ │  - macOS SwiftUI 菜单栏 + Admin Web UI                   │   │
│ └────────────────────────┬─────────────────────────────────┘   │
│                          │ 调用                                  │
│ ┌────────────────────────▼─────────────────────────────────┐   │
│ │            高级封装 (MLX 生态, Apple & 社区)                │   │
│ │  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐   │   │
│ │  │   mlx-lm     │ │  mlx-vlm     │ │ mlx-embeddings  │   │   │
│ │  │ (LLM 模型)   │ │ (视觉模型)   │ │  (Embedding)    │   │   │
│ │  └──────────────┘ └──────────────┘ └─────────────────┘   │   │
│ │  ┌──────────────┐ ┌──────────────┐                       │   │
│ │  │  mlx-audio   │ │ dflash-mlx   │ (音频+推测解码)      │   │
│ │  └──────────────┘ └──────────────┘                       │   │
│ └────────────────────────┬─────────────────────────────────┘   │
│                          │ 调用                                  │
│ ┌────────────────────────▼─────────────────────────────────┐   │
│ │            MLX Core (Apple 官方, Python + C++)           │   │
│ │  ┌──────────────────┐  ┌──────────────────────────┐     │   │
│ │  │  mlx.core (mx)   │  │     mlx.nn (nn)           │     │   │
│ │  │ - 张量计算        │  │ - Module / Layers         │     │   │
│ │  │ - Metal kernels  │  │ - Linear / Attention      │     │   │
│ │  │ - SDPA / Flash   │  │ - Embedding / RoPE        │     │   │
│ │  │ - Quantization   │  │                          │     │   │
│ │  │ - Compile cache  │  │                          │     │   │
│ │  └──────────────────┘  └──────────────────────────┘     │   │
│ └────────────────────────┬─────────────────────────────────┘   │
│                          │ FFI                                   │
│ ┌────────────────────────▼─────────────────────────────────┐   │
│ │        libmlx.dylib (C++ 编译产物, Metal 后端)           │   │
│ │  - GPU kernel 调度                                         │   │
│ │  - Unified memory 管理                                      │   │
│ │  - CompilerCache (C++ thread_local)                       │   │
│ └──────────────────────────────────────────────────────────┘   │
│                          │                                      │
│ ┌────────────────────────▼─────────────────────────────────┐   │
│ │            Apple Metal GPU (M-series GPU)                │   │
│ └──────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

**oMLX 处在最顶层**——它甚至都不直接碰 `mlx.core`，而是透过 `mlx-lm` / `mlx-vlm` 这些已经写好的"模型 + 推理循环"包去用 MLX。

---

## 三、oMLX 依赖的 MLX 系列包

从 `pyproject.toml` 可以看出：

| 包 | 版本 | 来源 | 作用 |
|---|------|------|------|
| **mlx** | `>=0.31.2` | PyPI | Apple 官方张量库（GPU 内核） |
| **mlx-lm** | `v0.31.3 (2c008fd)` | git pin | LLM 模型架构 + BatchGenerator 推理循环 |
| **mlx-vlm** | PR #1374 (086ab9d) | git pin | VLM 模型架构（含 MiniMax M3、Gemma4 等） |
| **mlx-embeddings** | (32981fa) | git pin | Embedding 模型 |
| **mlx-audio** [可选] | (5175326) | git pin | STT / TTS / STS 语音模型 |
| **dflash-mlx** | v0.1.9 (5d70fae) | git pin | Block Diffusion 推测解码 |
| **paroquant** [可选] | 0.1.15 | PyPI | 另一种量化格式 loader |

**为什么要 git pin 到具体 commit？**

- `pyproject.toml` 注释明确说明：「mlx-lm from commit (2c008fd, v0.31.3) - trust_remote_code load gate, transformers 5.7.0 compatibility, and prior BatchGenerator/cache fixes.」
- 因为 `mlx-lm` 还在频繁变动，某些 commit 引入了 oMLX 必需的 bug fix（如 issue #85 的 MLX Stream 竞争、trust_remote_code 安全门）。
- oMLX 同时通过 `omlx/patches/` 目录对 `mlx-lm` 缺失的模型（如 DeepSeek V4）打 monkey-patch，避免修改 `mlx-lm` 源码导致分叉。

---

## 四、oMLX 调用 MLX 的七个层次

按调用深度从表层到底层：

### 层次 1：HTTP API → EnginePool

```python
# omlx/server.py
@app.post("/v1/chat/completions")
async def create_chat_completion(...):
    lease = _LLMEngineLease()
    engine = await get_engine_for_model(request.model, lease=lease)
    # ...
```

→ **不直接接触 MLX**，仅路由分发。

### 层次 2：EnginePool → EngineCore（多模型管理）

```python
# omlx/engine_pool.py
@dataclass
class EngineEntry:
    engine: BaseEngine | None = None
    # LRU eviction, Pin, TTL
```

→ **不直接接触 MLX**，仅做生命周期管理。

### 层次 3：BaseEngine / BatchedEngine.start()（模型加载）

```python
# omlx/engine/batched.py
async def start(self):
    from mlx_lm import load
    # ...
    self._model, self._tokenizer = await loop.run_in_executor(
        get_mlx_executor(),
        lambda: load(self._model_name, tokenizer_config=..., trust_remote_code=...)
    )
```

→ **第一次出现 MLX 调用**：`mlx_lm.load()` 是 MLX 生态的高层 API，它内部会：
1. 读取 `config.json` 决定模型架构
2. 读取 `*.safetensors` 加载权重到 MLX Array
3. 构建 `mlx.nn.Module` 树
4. 加载 tokenizer

### 层次 4：Scheduler.step() → mlx_lm.generate.BatchGenerator（推理核心）

```python
# omlx/scheduler.py
from mlx_lm.generate import (
    BatchGenerator,
    GenerationBatch,
    PromptProcessingBatch,
    SequenceStateMachine,
    generation_stream,
)
from mlx_lm.models.cache import (
    KVCache as _MLXKVCache,
    RotatingKVCache as _MLXRotatingKVCache,
    make_prompt_cache,
)
from mlx_lm.sample_utils import make_logits_processors
```

→ **核心依赖**：oMLX 的"连续批处理"完全是 `BatchGenerator` 的封装。

`mlx_lm.generate.BatchGenerator` 的工作原理：

```python
class BatchGenerator:
    """Multi-sequence batched generator."""
    def __init__(self, model, ...):
        # 为每个 sequence 创建 KVCache
        self.state = SequenceStateMachine(...)
    
    def step(self, batch):
        # 合并 prefill + decode 到一个 forward pass
        model(**batch.to_inputs())  # ← mx.array 计算
        # 采样 + 状态转移
```

### 层次 5：BatchGenerator 内部 → `mlx.nn.Module.__call__()`

模型的前向计算：

```python
# 在 mlx_lm/models/llama.py 中（oMLX 不修改）
class Model(nn.Module):
    def __call__(self, x):
        h = self.embed_tokens(x)
        for layer in self.layers:
            h = layer(h)  # ← mx.fast.scaled_dot_product_attention
        return self.lm_head(h)
```

→ `mx.fast.scaled_dot_product_attention` 是 MLX 内置的 Flash Attention 实现。

### 层次 6：MLX 操作 → Metal kernel

```python
# mlx.core 操作
x = mx.array([1.0, 2.0, 3.0])  # 分配 unified memory
y = mx.matmul(x, x.T)           # 提交 Metal command buffer
mx.eval(y)                      # 强制求值（lazy evaluation）
```

→ MLX 内部用 lazy evaluation：算子不立即执行，而是构建计算图，`mx.eval()` 触发实际 Metal kernel 提交。

### 层次 7：Metal → Apple GPU

```
mx.eval(y)
  → Metal command buffer 入队
  → GPU kernel 执行（矩阵乘法、注意力等）
  → 完成后 CPU 通过共享内存读到结果（unified memory，无 PCIe 拷贝）
```

---

## 五、核心 API 调用清单

在 oMLX 整个仓库里，`mlx.*` / `mlx_lm.*` / `mlx_vlm.*` 共出现 **267+ 处**。下面是按用途归类的关键 API：

### 5.1 张量与内存管理

| API | 出处 | 用途 |
|-----|------|------|
| `mx.array(...)` | `mlx.core` | 创建张量 |
| `mx.eval(...)` | `mlx.core` | 强制求值 lazy 计算图 |
| `mx.synchronize()` | `mlx.core` | 阻塞等待所有 Metal 命令完成 |
| `mx.clear_cache()` | `mlx.core` | 清空 MLX 缓存分配器（腾出 unified memory） |
| `mx.get_active_memory()` | `mlx.core` | 当前活跃内存（admin dashboard 显示） |
| `mx.get_cache_memory()` | `mlx.core` | 缓存池大小 |
| `mx.get_peak_memory()` | `mlx.core` | 峰值内存 |
| `mx.device_info()` | `mlx.core` | GPU 型号、显存上限 |
| `mx.set_wired_limit(bytes)` | `mlx.core` | 调整 iogpu.wired_limit（kernel 级 GPU 内存上限） |
| `mx.new_thread_local_stream(device)` | `mlx.core` | 创建线程本地 Metal stream（防止多线程竞争） |
| `mx.default_device()` | `mlx.core` | 获取默认设备（gpu/cpu） |
| `mx.metal.is_available()` | `mlx.core` | Metal 是否可用（启动时 sanity check） |

### 5.2 量化

| API | 用途 |
|-----|------|
| `mx.quantize(...)` | 权重量化（4-bit、8-bit） |
| `mx.dequantize(...)` | 反量化 |
| `mx.quantized_matmul(...)` | 量化矩阵乘法 |
| `mlx_vlm.turboquant.*` | TurboQuant KV 缓存（Codec 由 head_dim+bits+seed 重建） |

### 5.3 计算原语

| API | 用途 |
|-----|------|
| `mx.matmul(a, b)` | 矩阵乘法 |
| `mx.softmax(x, axis=...)` | softmax |
| `mx.fast.scaled_dot_product_attention(...)` | Flash Attention |
| `mx.embedding(...)` | Embedding 查表 |
| `mx.argmax`, `mx.argsort` | 采样 / 排序 |
| `mx.cumsum`, `mx.take_along_axis` | top-p / top-k 采样 |

### 5.4 编译与缓存

| API | 用途 |
|-----|------|
| `@mx.compile` | 函数级编译缓存（DeepSeek V4 等模型用） |
| `@partial(mx.compile, shapeless=True)` | shape 不固定版本 |
| `mlx.core.detail.compile_clear_cache()` | 通过 ctypes 清空 worker 线程的 `CompilerCache`（`omlx/utils/compile_cache.py` 实现了 ctypes 调用） |

### 5.5 模型 / 推理（高层封装）

| API | 出处 | 用途 |
|-----|------|------|
| `mlx_lm.load(path)` | `mlx_lm` | 加载 LLM 模型 + tokenizer |
| `mlx_lm.tokenizer_utils.load(path)` | `mlx_lm` | 仅加载 tokenizer |
| `mlx_lm.tokenizer_utils.NaiveStreamingDetokenizer` | `mlx_lm` | 流式 detokenize |
| `mlx_lm.generate.BatchGenerator` | `mlx_lm` | **连续批处理核心** |
| `mlx_lm.generate.GenerationBatch` | `mlx_lm` | Decode 阶段 batch |
| `mlx_lm.generate.PromptProcessingBatch` | `mlx_lm` | Prefill 阶段 batch |
| `mlx_lm.generate.SequenceStateMachine` | `mlx_lm` | 序列状态机 |
| `mlx_lm.models.cache.KVCache` | `mlx_lm` | 标准 KV 缓存 |
| `mlx_lm.models.cache.RotatingKVCache` | `mlx_lm` | 滑动窗口 KV 缓存 |
| `mlx_lm.models.cache.make_prompt_cache` | `mlx_lm` | 为 prompt 创建初始 cache |
| `mlx_lm.sample_utils.make_logits_processors` | `mlx_lm` | 创建 logits processors |
| `mlx_lm.models.base.create_attention_mask` | `mlx_lm` | 创建注意力 mask |
| `mlx_lm.utils._get_classes` | `mlx_lm` | 动态加载模型架构（oQ 量化时用） |
| `mlx_lm.quant.utils.load_data` | `mlx_lm` | 加载量化校准数据 |
| `mlx_vlm.speculative.load_drafter` | `mlx_vlm` | 加载 VLM 推测解码 drafter |
| `mlx_vlm.speculative.utils._mtp_rounds` | `mlx_vlm` | Multi-Token Prediction |
| `mlx_vlm.turboquant.TurboQuantKVCache` | `mlx_vlm` | TurboQuant 量化 KV cache |
| `mlx.utils.tree_flatten` | `mlx.utils` | 把 nn.Module 展平成 (name, array) 列表 |
| `mlx.utils.tree_map` | `mlx.utils` | 对 nn.Module 所有叶子做函数映射 |
| `mlx.nn.layers.distributed.*` | `mlx.nn` | 分布式训练相关（部分多卡场景用） |

### 5.6 关键模型加载代码

```python
# omlx/utils/model_loading.py
def load_text_model(model_name, tokenizer_config=None, model_settings=None):
    """Load an LLM model/tokenizer pair via mlx-lm."""
    maybe_apply_pre_load_patches(model_name, model_settings=model_settings)
    from mlx_lm import load
    
    trust_remote_code = (
        bool(getattr(model_settings, "trust_remote_code", False))
        if model_settings is not None
        else False
    )
    return load(
        model_name,
        tokenizer_config=tokenizer_config,
        trust_remote_code=trust_remote_code,
    )
```

→ **所有 LLM 加载都走这个入口**，包括 monkey-patch、custom quantization 分发等。

---

## 六、oMLX 对 MLX 的关键适配

oMLX 不只是简单地 `import mlx`，它做了 **6 类深度适配** 来让 MLX 在服务器场景下稳定运行：

### 6.1 Thread-Local Stream 隔离（避免 Metal 命令缓冲竞争）

```python
# omlx/engine_core.py
def _init_mlx_thread():
    import mlx.core as mx
    stream = mx.new_thread_local_stream(mx.default_device())
    
    # 替换 mlx_lm.generate 全局变量
    gen_mod = sys.modules.get("mlx_lm.generate")
    if gen_mod is not None:
        gen_mod.generation_stream = stream
    
    sched_mod = sys.modules.get("omlx.scheduler")
    if sched_mod is not None:
        sched_mod.generation_stream = stream
```

**问题**：MLX 的 Metal command buffer 在多线程下会段错误（issue #85）。

**解决**：
- 每个 EngineCore 拥有独立线程 + `mx.new_thread_local_stream`
- 替换 `mlx_lm.generate.generation_stream` 这个模块级全局变量（这是个 hack！）

### 6.2 全局单线程 MLX Executor

```python
# omlx/engine_core.py
_global_mlx_executor: concurrent.futures.ThreadPoolExecutor | None = None

def get_mlx_executor():
    global _global_mlx_executor
    if _global_mlx_executor is None:
        _global_mlx_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-global",
            initializer=_init_mlx_thread,
        )
    return _global_mlx_executor
```

**问题**：MLX Metal 命令缓冲在多线程下竞态。

**解决**：所有 MLX 操作（包括模型加载）都通过这一个线程串行执行。

### 6.3 Compile Cache 生命周期管理（worker 线程崩溃防护）

```python
# omlx/utils/compile_cache.py
_CLEAR_SYMBOL = "_ZN3mlx4core6detail19compile_clear_cacheEv"
# Itanium-mangled mlx::core::detail::compile_clear_cache()

def _resolve_clear_fn():
    lib_dir = os.path.join(os.path.dirname(mx.__file__), "lib")
    libmlx = os.path.join(lib_dir, "libmlx.dylib")
    lib = ctypes.PyDLL(libmlx)
    fn = getattr(lib, _CLEAR_SYMBOL)
```

**问题**：MLX 的 `@mx.compile` 缓存是 C++ `thread_local CompilerCache`。worker 线程销毁时 `~CompilerCache` 在不持有 GIL 的情况下释放 Python 对象 → 进程崩溃（`Fatal Python error: PyThreadState_Get: no current thread`）。

**解决**：通过 ctypes 直接调 `libmlx.dylib` 的 `compile_clear_cache()` 符号，在 worker 线程销毁前清空缓存，使 `~CompilerCache` 成为 no-op。

```python
_immortal_mlx_executors: list = []  # 当 symbol 不可解析时保留 executor 永不复用
_immortal_mlx_streams: list = []
```

### 6.4 Sampling 重写（mx.compile RNG bug）

```python
# omlx/utils/sampling.py 头注释
"""omlx sampling utilities — mx.compile-free re-implementation of mlx-lm samplers.

mlx-lm 0.31.x decorates ``categorical_sampling`` and the apply_* helpers with
``@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)``. In
the omlx server environment the decorator stops advancing the RNG state after
the first call: all subsequent samples reuse the same state, so identical
prompts produce character-identical output even at temperature > 1.
"""
```

**问题**：`mlx_lm.sample_utils` 用了 `@mx.compile(inputs=mx.random.state, outputs=mx.random.state)`，但 server 环境下 RNG state 不前进。

**解决**：oMLX 镜像了 `mlx_lm.sample_utils.make_sampler` 的实现但去掉 `@mx.compile` 装饰器，行为完全一致。

### 6.5 Lazy State 物化（避免跨线程 stream 错误）

```python
# omlx/utils/model_loading.py
def materialize_lazy_state(model: Any) -> None:
    """Force-evaluate every mx.array in the model tree on the loader thread."""
    arrays = [v for _, v in tree_flatten(model) if isinstance(v, mx.array)]
    if arrays:
        mx.eval(arrays)
```

**问题**：`mlx-vlm.load()` 跑 `mx.eval(model.language_model.parameters())` 后，留下了 frozen buffers（RoPE freqs 等）和 vision_tower / audio_tower 子树作为 lazy 数组，绑定到 loader 线程的 default stream。当 EngineCore 线程 #1304 后跑 forward 时，碰到「no Stream(gpu, X) in current thread」。

**解决**：在加载线程就把整个 model tree 物化（`mx.eval(arrays)`），让每个 leaf array 在任何线程都能安全访问。

### 6.6 Model Ownership Registry（防止 BatchGenerator KV 状态冲突）

```python
# omlx/model_registry.py 头注释
"""The problem: mlx-lm's BatchGenerator maintains internal KV cache state
tied to the model. When multiple EngineCore instances use the same model,
the cache objects become incompatible, causing NoneType errors.

Solution: Track model ownership and ensure only one engine's BatchGenerator
is active for each model at a time."""
```

**问题**：BatchGenerator 内部维护 KV cache state 绑定到模型对象。同一个模型被多个 EngineCore 同时持有会冲突。

**解决**：全局 `ModelRegistry` 单例 + weakref，记录「哪个 engine_id 持有哪个 model_id」，强制独占。

---

## 七、为什么 oMLX 不直接用 MLX 而是要包一层

### 7.1 MLX 本身的能力 vs 服务器需求

| 能力 | MLX 是否提供 | oMLX 是否补齐 |
|------|------------|--------------|
| 张量计算 + Metal kernel | ✅ | 直接用 |
| 单模型 forward / backward | ✅ | 直接用 |
| 单请求推理循环 | ✅ (`mlx_lm.generate`) | 直接用 |
| **多请求并发** | ❌ | ✅ EnginePool + AsyncEngineCore |
| **连续批处理** | 部分（BatchGenerator） | ✅ Scheduler 包装 |
| **PagedAttention** | ❌ | ✅ PagedCacheManager |
| **KV 缓存分层** | ❌ | ✅ Hot + SSD 双层 |
| **Prefix sharing** | ❌ | ✅ BlockAwarePrefixCache |
| **Copy-on-Write** | ❌ | ✅ CoW block 复用 |
| **多模型同驻** | ❌ | ✅ LRU + Pin + TTL |
| **HTTP API** | ❌ | ✅ FastAPI |
| **OpenAI/Anthropic 协议** | ❌ | ✅ Adapter 层 |
| **Admin UI** | ❌ | ✅ Alpine.js dashboard |
| **macOS 菜单栏** | ❌ | ✅ SwiftUI |
| **系统级内存守护** | ❌ | ✅ ProcessMemoryEnforcer |
| **Claude Code 集成** | ❌ | ✅ Context scaling |
| **oQ 量化** | ❌ | ✅ omlx/oq.py |

**结论**：MLX 是「能跑模型」，oMLX 是「让模型跑成生产服务」。

### 7.2 oMLX 与 vLLM 的类比

```
vLLM  =  PyTorch  +  CUDA kernel  +  PagedAttention  +  Scheduler  +  HTTP server
oMLX  =  MLX      +  Metal kernel +  PagedCache     +  Scheduler  +  HTTP server
                    (Apple GPU)   (新增, oMLX 自创)  (包装 mlx-lm) (FastAPI)
```

**vLLM 的 PagedAttention 是 vLLM 自创的**，不是 PyTorch 的功能。oMLX 的 PagedCacheManager 同理——是 oMLX 自创，MLX 完全没有。

### 7.3 oMLX 与 mlx-lm 的关系

- **mlx-lm** 提供：`load()`、`BatchGenerator`、`KVCache`、`tokenizer_utils`、各种 `models/*.py` 模型架构
- **oMLX** 提供：把 `mlx_lm.BatchGenerator` 包装进一个能服务多请求、多模型、带缓存分层、内存守护的服务器

oMLX 实际上把 `BatchGenerator` 当成「单个模型的推理执行器」用，然后在它外面套了一层调度、缓存、生命周期管理。

---

## 八、与 MLX 的边界：oMLX 不做这些事

| 不做的 | 说明 |
|--------|------|
| ❌ 重写模型架构 | Llama / Qwen / DeepSeek / GLM 等模型架构完全用 `mlx_lm.models.*` |
| ❌ 重写 attention kernel | 直接用 `mx.fast.scaled_dot_product_attention` |
| ❌ 实现量化算法 | 用 `mx.quantize` / `mx.quantized_matmul` |
| ❌ Metal shader | 完全交给 `libmlx.dylib` |
| ❌ 训练/反向传播 | MLX 支持但 oMLX 不做（仅推理） |

oMLX 的边界非常清晰：**底层 GPU 计算全部交给 MLX，oMLX 只做"服务器化"的系统软件**。

---

## 九、总结

### MLX 与 oMLX 的关系是一句话：

> **MLX 是"GPU 内核 + 张量库"，oMLX 是"跑在 MLX 上的 LLM 推理服务器"，中间用 mlx-lm / mlx-vlm 作为"模型 + 推理循环"的高层封装。**

### oMLX 调用 MLX 的模式：

```
FastAPI route
  → EnginePool (LRU)
    → AsyncEngineCore (per-model lifecycle)
      → Scheduler (FCFS + chunked prefill + decode burst)
        → mlx_lm.BatchGenerator (continuous batching)
          → mlx.nn.Module.__call__() (model forward)
            → mlx.core ops (mx.matmul, mx.softmax, mx.fast.scaled_dot_product_attention)
              → libmlx.dylib (Metal C++ kernel)
                → Apple GPU
```

### oMLX 对 MLX 的关键贡献：

1. **Thread-local Stream + 全局单线程 Executor** —— 让 MLX Metal 能在多模型并发下稳定运行
2. **PagedCache + SSD 分层** —— MLX 完全没有的能力
3. **Continuous Batching 调度** —— 在 `BatchGenerator` 上层包装 vLLM 式调度
4. **多模型生命周期管理** —— EnginePool 的 LRU/Pin/TTL
5. **MLX 兼容性修补** —— 修复 RNG state 不前进、compile cache 崩溃、lazy state 跨线程错误等 MLX 自身 bug
6. **生产级服务器外围** —— HTTP API / Admin UI / macOS App / 内存守护

### 一句话总结 oMLX 的定位：

> **oMLX 是"Apple Silicon 上的 vLLM"——把 MLX 的 GPU 性能包装成生产可用的 LLM 服务。**

---

## 附录：参考链接

- [MLX 官方仓库](https://github.com/ml-explore/mlx) - Apple 官方张量库
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - LLM 模型 + BatchGenerator
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - VLM 模型
- [vLLM](https://github.com/vllm-project/vllm) - oMLX 的设计灵感来源（CUDA 版）
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx) - oMLX 的直接 fork 起点
- [venvstacks](https://venvstacks.lmstudio.ai) - macOS App 打包工具

---

*文档生成时间：基于 omlx 仓库当前 HEAD 分析*