# 100 · Apple MLX 框架原理深度剖析

> **文档编号**：`analysis/100-mlx-framework-principles.md`
> **主题**：Apple MLX 框架的设计原理、核心机制、与 PyTorch/JAX 的对比
> **范围**：统一内存、Lazy Evaluation、JIT 编译、Metal 后端、SDPA、量化、Stream 模型
> **前置阅读**：[002-mlx-omlx-relationship.md](./002-mlx-omlx-relationship.md)

> 📁 本文档属于 [analysis/](./README.md) 目录。

---

## 目录

- [一、MLX 是什么？](#一mlx-是什么)
- [二、MLX 的设计哲学](#二mlx-的设计哲学)
- [三、统一内存：Apple Silicon 的杀手锏](#三统一内存apple-silicon-的杀手锏)
- [四、Lazy Evaluation：延迟求值模型](#四lazy-evaluation延迟求值模型)
- [五、JIT 编译：@mx.compile 与 CompilerCache](#五jit-编译mxcompile-与-compilercache)
- [六、Metal 后端与 Kernel 调度](#六metal-后端与-kernel-调度)
- [七、SDPA / Flash Attention 实现](#七sdpa--flash-attention-实现)
- [八、量化原语](#八量化原语)
- [九、Stream 模型：MLX 的线程安全](#九stream-模型mlx-的线程安全)
- [十、内存管理：active / cache / peak / wired](#十内存管理active--cache--peak--wired)
- [十一、MLX vs PyTorch vs JAX 对比](#十一mlx-vs-pytorch-vs-jax-对比)
- [十二、MLX 的优势与限制](#十二mlx-的优势与限制)
- [十三、MLX 在 oMLX 中的关键使用模式](#十三mlx-在-omlx-中的关键使用模式)
- [十四、总结](#十四总结)

---

## 一、MLX 是什么？

**MLX** 是 Apple 在 2023 年 12 月开源的**机器学习数组框架**，专门为 Apple Silicon（M1/M2/M3/M4）优化。

```python
# MLX 官方仓库: https://github.com/ml-explore/mlx
# License: MIT
# 首次发布: 2023-12 (Apple 官方)
# 当前版本 (2026-06): 0.31.x
```

**核心特性**：
- **NumPy 风格的 API**：用户友好，类似 NumPy / PyTorch
- **可组合函数变换**：自动微分、JIT 编译、自动求导
- **多设备支持**：CPU、GPU（Metal 后端）
- **统一内存**：CPU/GPU 共享内存，零拷贝
- **Lazy Evaluation**：构建计算图，延迟到需要时执行
- **动态图**：与 PyTorch 类似，不是 JAX 的静态 traced graph

**官方口号**：
> *"MLX is an array framework for machine learning on Apple silicon, brought to you by Apple machine learning research."*

---

## 二、MLX 的设计哲学

### 2.1 三条核心原则（来自 MLX 论文/文档）

1. **Familiar API**：像 NumPy 一样直观
2. **Composable function transforms**：自动微分、JIT、vmap 等作为函数变换
3. **Lazy computation**：构建图，按需执行

### 2.2 与三大主流框架的对比定位

```
PyTorch  → 动态图 + Eager Execution + CUDA-first
JAX      → 静态图 + JIT + XLA 跨平台
TensorFlow → 静态图 + Session → 现 tf.function dynamic
MLX      → 动态图 + Lazy + Metal-only (Apple Silicon)
```

**MLX 的独特定位**：

| 维度 | PyTorch | JAX | MLX |
|------|---------|-----|-----|
| 执行模型 | Eager (立即执行) | Lazy + JIT | **Lazy (延迟执行)** |
| 自动微分 | autograd | grad/grad变换 | `mx.grad`/`value_and_grad` |
| JIT | torch.compile (图优化) | jax.jit (必需) | `@mx.compile` (可选) |
| 设备 | CUDA > MPS > CPU | TPU > GPU > CPU | **GPU = CPU (unified)** |
| API 风格 | Pythonic | 函数式 | NumPy + Pythonic |
| 跨平台 | 全平台 | 全平台 | **macOS only** |

### 2.3 MLX 不是 PyTorch 移植

很多人误以为 MLX 是 PyTorch 的 Apple Silicon 移植——**不是**。

MLX 是**全新设计**的框架，借鉴了多个框架的最佳实践：

```
MLX 的灵感来源:
├── NumPy: 数组 API 风格
├── PyTorch: 动态图 + Module 抽象
├── JAX: 函数变换 (grad, vmap, jit)
├── TensorFlow: Graph optimization
└── Swift for TensorFlow: Apple 在 ML 编译器的经验
```

---

## 三、统一内存：Apple Silicon 的杀手锏

### 3.1 什么是统一内存架构 (UMA)？

传统架构（NVIDIA GPU）：

```
┌─────────────┐         PCIe bus        ┌─────────────┐
│   CPU RAM   │  ◄──────────────────►  │   GPU VRAM   │
│   (DDR5)    │    ~32 GB/s 双向带宽    │   (HBM)     │
│             │    但有拷贝开销          │   16-80 GB  │
└─────────────┘                          └─────────────┘
       ↓                                        ↓
   LLM 权重 / activations              LLM 权重 / KV cache
```

**问题**：
- 权重需要 CPU↔GPU 双向拷贝（host-to-device, device-to-host）
- 大模型无法装入单卡 VRAM，必须拆分
- KV cache 受限于 VRAM 大小
- PCIe 带宽是瓶颈

Apple Silicon UMA：

```
┌────────────────────────────────────┐
│      Unified Memory                 │
│      (M3 Max: 128 GB)              │
│                                     │
│   CPU  ─┐         ┌─ GPU            │
│   Core  │ 共享同一物理内存           │  Metal Shader  │
│         │ (无 PCIe)                 │                │
│   ANE ──┘                          └─ Neural Engine │
│                                     │                │
└────────────────────────────────────┘
       ↓                               ↓
   LLM 权重 / activations / KV cache 全部在同一内存
```

**好处**：
- **零拷贝**：CPU 和 GPU 访问同一份内存
- **大内存**：M3 Max 128 GB 远超 H100 80 GB
- **统一地址空间**：指针直接共享，无需 cudaMemcpy
- **Metal 访问 ANE**：Apple Neural Engine 也可参与

### 3.2 oMLX 直接利用 UMA 实现的特性

```python
# omlx/cache/paged_ssd_cache.py
# 利用 UMA 让 KV cache 直接被 Metal 访问，无 host-device 拷贝

def save_block(block_hash, kv_tensors):
    # kv_tensors 是 mlx.core.array (在 unified memory 中)
    # 序列化为 safetensors 到 SSD
    safetensors.save_file({k: np.asarray(v) for k, v in kv_tensors.items()}, path)

def load_block(block_hash):
    # 从 SSD 读取，反序列化为 mlx.core.array
    # 直接在 unified memory 中，GPU 可立即访问
    return {k: mx.array(v) for k, v in tensors.items()}
```

**NVIDIA 等价场景需要**：
```python
# PyTorch + CUDA
kv_gpu = kv_gpu.cuda()       # GPU 上
kv_cpu = kv_gpu.cpu()         # 拷回 CPU
torch.save(kv_cpu, path)      # 存盘
# 反向：
kv_cpu = torch.load(path)
kv_gpu = kv_cpu.cuda()        # 再拷到 GPU（PCIe 带宽限制）
```

**UMA 优势**：
- 无 host↔device 拷贝开销
- 大模型无需 tensor parallel 拆分
- KV cache 可以"溢出"到 host RAM（同地址空间）

---

## 四、Lazy Evaluation：延迟求值模型

### 4.1 立即执行 vs 延迟执行

**PyTorch Eager**（立即执行）：
```python
import torch
a = torch.tensor([1, 2, 3])      # 立刻分配 GPU 内存 + 计算
b = torch.tensor([4, 5, 6])      # 立刻分配
c = a + b                        # 立刻计算，分配结果内存
# 每一步都是真实的 GPU 操作
```

**MLX Lazy**（延迟执行）：
```python
import mlx.core as mx
a = mx.array([1, 2, 3])         # 仅记录到计算图，**不立即执行**
b = mx.array([4, 5, 6])         # 同上
c = a + b                        # 仅记录 "a + b" 节点，**不立即计算**
print(c)                         # 触发 mx.eval(c) → 实际 GPU 计算
```

### 4.2 Lazy 求值的实现机制

MLX 内部用 **C++ 类图结构** 记录所有操作：

```cpp
// MLX 核心数据结构（简化）
struct array {
    primitive* prim;          // 操作类型（如 "Add", "Matmul"）
    std::vector<array> inputs; // 输入数组
    Shape shape;
    Dtype dtype;
    
    // Lazy 评估：构造时不执行，仅记录
    array operator+(const array& other) {
        return array{&add_primitive, {*this, other}};
    }
};

// 触发评估：
void eval(const array& arr) {
    // 拓扑排序找到依赖
    auto topo = topological_sort(arr);
    // 提交到 Metal command buffer
    for (auto& node : topo) {
        MetalKernel::launch(node);
    }
    arr.eval_flag = true;
}
```

### 4.3 Lazy 的优势

**性能优化**：
- **算子融合**：连续的 elementwise ops 可以合并为一个 kernel
- **减少内存分配**：避免中间结果占用内存
- **并行化机会**：编译器可以看到完整图做调度

```python
# PyTorch: 3 次 kernel launch + 2 个中间 buffer
y = torch.relu(x @ W + b)
# kernel 1: x @ W         → alloc tmp1
# kernel 2: tmp1 + b      → alloc tmp2
# kernel 3: relu(tmp2)    → alloc y

# MLX: 1 次 fused kernel（如果用 @mx.compile）
@mx.compile
def fused(x, W, b):
    return mx.relu(x @ W + b)
y = fused(x, W, b)  # 单次 Metal kernel launch
```

**灵活编程**：
```python
# 动态控制流不会破坏 lazy graph
@mx.compile
def step(x):
    if x.sum() > 0:        # 控制流也是图的一部分
        return mx.sin(x)
    else:
        return mx.cos(x)
```

### 4.4 oMLX 中的 Lazy 风险与对策

```python
# omlx/utils/model_loading.py
def materialize_lazy_state(model: Any) -> None:
    """Force-evaluate every mx.array in the model tree on the loader thread."""
    arrays = [v for _, v in tree_flatten(model) if isinstance(v, mx.array)]
    if arrays:
        mx.eval(arrays)
```

**问题**：`mlx-vlm.load()` 跑 `mx.eval(model.language_model.parameters())` 后，**剩余**的 lazy arrays（RoPE freqs、vision_tower、audio_tower）**绑定到 loader 线程的 default stream**。当 EngineCore worker 线程（独立 stream）跑 forward 时，会报 `"no Stream(gpu, X) in current thread"`。

**解决**：在加载线程物化整个 model tree。

---

## 五、JIT 编译：@mx.compile 与 CompilerCache

### 5.1 @mx.compile 工作原理

```python
import mlx.core as mx

@mx.compile
def attention(q, k, v):
    scale = q.shape[-1] ** -0.5
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    weights = mx.softmax(scores, axis=-1)
    return weights @ v

# 第一次调用：编译 + 执行
y = attention(q, k, v)  # 慢（编译）

# 后续调用：复用编译产物
y = attention(q, k, v)  # 快（直接执行编译后的 kernel）
```

**编译过程**：

```
Python 函数
  ↓ MLX graph capture
MLX 计算图 (多个 primitive 节点)
  ↓ 图优化
优化后计算图 (算子融合、死代码消除)
  ↓ Metal Shading Language 代码生成
MSL 源码 (类似 CUDA kernel)
  ↓ Metal compiler
.metallib (Metal kernel 库)
  ↓ 缓存到 CompilerCache
线程本地缓存 (key = 输入 shape + dtype)
```

### 5.2 CompilerCache 的生命周期与陷阱

```python
# mlx/core/compile.cpp (MLX 源码，简化)
class CompilerCache {
    // C++ thread_local (PR #3280 之后)
    thread_local static std::unordered_map<Hash, CompiledKernel> cache;
    
    ~CompilerCache() {
        // 析构时释放所有缓存的 Python 对象
        // ⚠️ 此析构在不持有 GIL 的线程上下文执行
        for (auto& [k, v] : cache) {
            v.py_obj.dec_ref();  // ← 这里会 crash
        }
    }
};
```

**问题**：worker 线程销毁时 `~CompilerCache` 在 thread-exit handler 中执行，**不持有 GIL**，但要 dec_ref Python 对象 → 触发 `Fatal Python error: PyThreadState_Get: no current thread`。

### 5.3 oMLX 的应对：ctypes 直接调用 `compile_clear_cache()`

```python
# omlx/utils/compile_cache.py (核心)
_CLEAR_SYMBOL = "_ZN3mlx4core6detail19compile_clear_cacheEv"
# Itanium-mangled: mlx::core::detail::compile_clear_cache()

def _resolve_clear_fn():
    lib_dir = os.path.join(os.path.dirname(mx.__file__), "lib")
    libmlx = os.path.join(lib_dir, "libmlx.dylib")
    lib = ctypes.PyDLL(libmlx)  # PyDLL 保持 GIL
    fn = getattr(lib, _CLEAR_SYMBOL)
    return fn

def clear_thread_compile_cache() -> None:
    """Clear the CALLING thread's MLX thread-local compile cache.
    MUST run ON the thread whose cache should be cleared."""
    fn = _resolve_clear_fn()
    if fn is not None:
        fn()  # 直接调 libmlx.dylib 的 C++ 符号
```

**调用时机**：
```python
# omlx/engine_core.py
def close(self):
    if compile_cache_clear_available():
        # 在 worker 线程销毁前，先 clear
        clear_thread_compile_cache()
        self._mlx_executor.shutdown(wait=True)
    else:
        # Symbol 不可解析（fallback）：保留 executor 永不复用
        _immortal_mlx_executors.append(self._mlx_executor)
```

**这是 oMLX 对 MLX 最深层的适配**——直接调 C++ 符号修复 MLX 自身的线程安全问题。

### 5.4 MLX 的 RNG State 与 @mx.compile 冲突

```python
# omlx/utils/sampling.py 头注释
"""mlx-lm 0.31.x decorates ``categorical_sampling`` and the apply_* helpers with
``@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)``. In
the omlx server environment the decorator stops advancing the RNG state after
the first call: all subsequent samples reuse the same state, so identical
prompts produce character-identical output even at temperature > 1."""
```

**问题**：mlx-lm 的 `categorical_sampling` 装饰了 `@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)`，声明 RNG state 是输入也是输出。**但在 server 环境下**，这个声明似乎没有真正生效——RNG state 不前进。

**解决**：oMLX 重写 sampling：

```python
# omlx/utils/sampling.py
def make_sampler(sampling_params):
    """Mirror of mlx_lm.sample_utils.make_sampler but without @mx.compile."""
    def sampler(logits):
        # 不用 @mx.compile，直接调底层 primitives
        if sampling_params.temp == 0:
            return mx.argmax(logits, axis=-1)
        # ... 完整 mirror mlx_lm 逻辑
    return sampler
```

**结果**：RNG state 正确前进，每个请求采样不同。

---

## 六、Metal 后端与 Kernel 调度

### 6.1 Metal 是什么？

**Metal** 是 Apple 的 GPU 编程框架：

```
Metal API (C/Objective-C/Swift)
  ↓
Metal Shading Language (MSL) (类似 CUDA C)
  ↓
.metallib (编译后的 kernel)
  ↓
Apple GPU 执行
```

### 6.2 MLX 的 Metal 后端架构

```cpp
// MLX 后端抽象（简化）
class Backend {
    virtual void eval(const array& arr) = 0;
};

class MetalBackend : public Backend {
    MTLDevice* device;
    MTLCommandQueue* queue;
    std::unordered_map<Primitive*, MetalKernel*> kernel_cache;
    
    void eval(const array& arr) override {
        // 1. 拓扑排序计算图
        auto order = topo_sort(arr);
        
        // 2. 创建 command buffer
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        
        // 3. 为每个 primitive 调度 kernel
        for (auto& node : order) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            auto kernel = get_or_compile_kernel(node.prim);
            [enc setComputePipelineState:kernel.pipeline];
            // 设置 buffer / threadgroup
            [enc dispatchThreadgroups:... threadsPerThreadgroup:...];
            [enc endEncoding];
        }
        
        // 4. 提交
        [cb commit];
        [cb waitUntilCompleted];  // 同步
    }
};
```

### 6.3 Metal Command Buffer 模型

```
CPU (Python)                 GPU (Metal)
    │                            │
    │── MX array operation ─────►│
    │                            │ → kernel 1 launch
    │                            │ → kernel 2 launch (depends on 1)
    │                            │ → kernel 3 launch (depends on 2)
    │                            │
    │── mx.eval() ──────────────►│ → 提交 command buffer
    │                            │ → GPU 顺序执行所有 kernel
    │                            │ → 完成
    │◄── 返回结果 ──────────────│
```

**关键特性**：
- 单次 command buffer 可包含多个 kernel
- kernel 之间通过 Metal buffer 共享数据（unified memory）
- 显式 `waitUntilCompleted` 同步

### 6.4 mx.fast 子模块：手写高性能 kernel

```python
# mlx/core/fast.cpp (MLX 源码)
namespace fast {

// 手写 RMSNorm kernel（针对 Apple GPU 优化）
array rms_norm(const array& x, const array& weight, float eps) {
    // 用 Metal Performance Shaders + 手写 MSL
    // 比通用 primitive 快 2-3 倍
}

// 手写 RoPE
array rope(const array& x, const array& freqs, bool traditional, ...) {
    // ...
}

// Flash Attention (SDPA)
array scaled_dot_product_attention(
    const array& q, const array& k, const array& v,
    float scale, bool mask  // ...
) {
    // 调用 Metal Performance Shaders 的 MPSSDPA
    // 或 fallback 到手写 attention kernel
}

} // namespace fast
```

**oMLX 用到的 mx.fast**：
```bash
# grep -rn "mx\.fast\." omlx/omlx/ --include="*.py"
mx.fast.rms_norm          # 9 次  (RMSNorm 层)
mx.fast.rope              # 1 次  (RoPE 位置编码)
mx.fast.scaled_dot_product_attention  # 4 次  (Flash Attention)
mx.fast.metal_kernel      # 1 次  (自定义 kernel)
```

---

## 七、SDPA / Flash Attention 实现

### 7.1 MLX 的 SDPA

```python
import mlx.core as mx

q = mx.random.normal((1, 32, 2048, 64))  # (B, H, S, D)
k = mx.random.normal((1, 32, 2048, 64))
v = mx.random.normal((1, 32, 2048, 64))

# 标准 attention (会 OOM 长序列)
scores = q @ k.transpose(0, 1, 3, 2)  # (B, H, S, S) → 2048^2 * 32 floats = 512 MB
weights = mx.softmax(scores, axis=-1)
out = weights @ v

# Flash Attention (内存 O(S)，不分块)
out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0/(64**0.5))
```

### 7.2 MLX SDPA 内部实现路径

```python
# omlx/memory_monitor.py 中的镜像
_SDPA_VECTOR_QUERY_TOKEN_THRESHOLD = 8
_SDPA_FULL_SUPPORTED_HEAD_DIMS = frozenset({64, 80, 128})
_SDPA_VECTOR_SUPPORTED_HEAD_DIMS = frozenset({64, 96, 128, 256})

# 注释：
# Mirrors MLX Metal ScaledDotProductAttention::use_fallback for the
# generation/inference path. Full prefill and short vector kernels support
# different head dimensions; unsupported cases fall back to an unfused
# score-matrix allocation.
```

**MLX 内部决策树**：

```
input: (B, H, S_q, D), (B, H, S_kv, D), (B, H, S_kv, D)
  │
  ├─ head_dim ∈ {64, 80, 128} 且非 short-vector case
  │   → 用 MPSSDPA (Metal Performance Shaders) full kernel
  │
  ├─ head_dim ∈ {64, 96, 128, 256} 且 S_q ≤ 8 (decode 阶段)
  │   → 用 MPSSDPA vector kernel
  │
  └─ 否则（unsupported head_dim 或大 S_q）
      → Fallback：完整计算 Q @ K^T score matrix
        → 内存 O(S_q × S_kv)
        → 慢但通用
```

### 7.3 oMLX 利用 SDPA 的细节

```python
# omlx/memory_monitor.py - 内存估算考虑 SDPA fallback
def estimate_prefill_memory(num_tokens, num_heads, head_dim, dtype_size, model):
    # 标准 KV 内存
    kv_memory = num_tokens * num_layers * num_kv_heads * head_dim * dtype_size * 2
    
    # SDPA fallback score matrix 内存（如果用 fallback）
    if needs_sdpa_fallback(model):
        score_memory = num_tokens * num_tokens * num_heads * dtype_size
        # 矩阵 S x S 在 prefill 时是平方级！
        # S=8192 时: 8192*8192*32 = 2 GB (fp16)
    else:
        score_memory = 0
    
    return kv_memory + score_memory
```

---

## 八、量化原语

### 8.1 MLX 的量化 API

```python
import mlx.core as mx

# 1. 简单量化 (per-tensor 或 per-group)
w = mx.random.normal((1024, 1024))
w_q, scales, biases, group_size = mx.quantize(w, bits=4, group_size=64)
# w_q: (1024, 1024) int4 (packed)
# scales: (1024, 16) fp16 (每 group 64 元素一个 scale)
# biases: (1024, 16) fp16
# group_size: 64

# 2. 反量化
w_deq = mx.dequantize(w_q, scales, biases, group_size=64, bits=4)
# w_deq ≈ w （有损）

# 3. 量化矩阵乘法
y = mx.quantized_matmul(
    x,            # 输入 (fp16)
    w_q, scales, biases,  # 量化权重
    group_size=64,
    bits=4,
    transpose=True,
)
```

### 8.2 量化格式

| 格式 | bits | group_size | 适用 |
|------|------|-----------|------|
| INT4 | 4 | 32 / 64 / 128 | **最常用** (4-bit GPTQ/AWQ) |
| INT8 | 8 | 32 / 64 / 128 | 高质量 |
| FP16 | 16 | - | 无损 |
| BF16 | 16 | - | 无损 |

### 8.3 MLX 的 GPTQ/AWQ 集成

```python
# mlx-examples/quantization/gptq.py (MLX 官方示例)
def gptq_quantize(model, calibration_data, bits=4, group_size=64):
    for layer in model.layers:
        # 1. 收集 Hessian
        H = compute_hessian(layer, calibration_data)
        
        # 2. 计算量化顺序
        order = compute_gptq_order(H, layer.weight.shape)
        
        # 3. 按顺序量化
        for i in order:
            w_q, scales, biases = quantize_block(layer.weight[..., i], bits, group_size)
            layer.weight[..., i] = w_q
            # 更新 Hessian 误差
            H = update_hessian(H, error)
```

**oMLX 的 oQ 量化**（`omlx/oq.py`）：

```python
# omlx/oq.py 头注释
"""Mixed-precision quantization combining GGUF K-quant layer position strategy,
[..] GPTQ algorithm by Frantar et al. The batched expert processing and MoE-aware
Hessian sharing are oQ-specific optimizations. Sensitivity-driven budget allocation
was inspired by approaches in llm-compressor and GGUF K-quants."""

# 自定义混合精度策略：
def universal_quant_predicate(path, oq_level):
    """
    - oq_level 1: 极限 (~2 BPW)
    - oq_level 5: 高质量 (~7 BPW)
    
    关键模块保持高精度：
    - vision_tower, image_newline → 跳过
    - lm_head → 低精度 (高敏感)
    - 早期层 (layers.0-2) → 高精度
    - MoE 路由 → 低 bits
    """
```

---

## 九、Stream 模型：MLX 的线程安全

### 9.1 默认 Stream 模型

MLX 默认是 **process-wide Metal command queue**，所有线程共享：

```cpp
// MLX 默认 stream（简化）
class DefaultStream {
    static MTLCommandQueue* queue;  // 进程级单例
    // 任何线程的 eval() 都用这个 queue
};
```

**问题**：

```
线程 A: eval(arr_A) → 提交 kernel 1, kernel 2 到 command queue
线程 B: eval(arr_B) → 提交 kernel 3, kernel 4 到 command queue (交错)
线程 A: item(arr_A) → 等待 → 但 kernel 4 (线程 B) 还在执行 → 等很久
```

实际表现：**随机 segfault 或数据竞争**（issue #85）。

### 9.2 MLX Stream API

```python
import mlx.core as mx

# 1. 默认 stream (进程级)
mx.synchronize()  # 等所有 default stream 完成

# 2. 显式 stream
stream = mx.new_stream(mx.default_device())
with mx.stream(stream):
    y = mx.matmul(x, W)  # 此操作绑定到 stream
# 离开 with 块后操作仍属于 stream

# 3. 线程本地 stream (per-thread 隔离)
stream = mx.new_thread_local_stream(mx.default_device())
# 每个线程第一次访问时，自动创建一个 thread_local stream
```

### 9.3 oMLX 的 Stream 策略

```python
# omlx/engine_core.py
def _init_mlx_thread():
    """Replace generation_stream with a thread-local stream on the executor thread."""
    stream = mx.new_thread_local_stream(mx.default_device())
    
    # mlx-lm 的 generation_stream 是模块级全局
    gen_mod = sys.modules.get("mlx_lm.generate")
    if gen_mod is not None:
        gen_mod.generation_stream = stream  # 替换！
    
    sched_mod = sys.modules.get("omlx.scheduler")
    if sched_mod is not None:
        sched_mod.generation_stream = stream  # 也替换
```

**为什么必须替换 `mlx_lm.generate.generation_stream`？**

```python
# mlx_lm/generate.py (MLX 官方代码)
generation_stream = mx.new_stream(mx.default_device())  # 模块级全局

class BatchGenerator:
    def step(self, batch):
        with mx.stream(generation_stream):  # 用全局 stream
            # ... forward 计算
            ...
```

**问题**：这个 `generation_stream` 在哪个线程首次 `import mlx_lm.generate` 就属于哪个线程（main thread）。

**如果其他线程用它**：崩溃（"no Stream(gpu, X) in current thread"）。

**oMLX 的解决**：在 worker 线程的 initializer 里替换它。

### 9.4 多 EngineCore 并行

```python
# omlx/engine_pool.py
class EnginePool:
    def __init__(self):
        self._entries = {}  # model_id → EngineEntry
    
    async def load_model(self, model_id):
        # 每个 EngineCore 独立线程 + 独立 stream
        engine = AsyncEngineCore(
            model=...,
            config=...,
        )
        # engine._mlx_stream = mx.new_thread_local_stream(...)
        # engine._mlx_executor = ThreadPoolExecutor(max_workers=1, ...)
        
        # 不同模型可以真正并行（不同 stream 不会互相阻塞）
```

**示意图**：

```
EngineCore 1 (model A)     EngineCore 2 (model B)     EngineCore 3 (model C)
   │                          │                          │
   ├─ Thread 1                ├─ Thread 2                ├─ Thread 3
   ├─ Stream 1                ├─ Stream 2                ├─ Stream 3
   │                          │                          │
   ├─ Kernel A1               ├─ Kernel B1               ├─ Kernel C1
   ├─ Kernel A2               ├─ Kernel B2               ├─ Kernel C2
   │                          │                          │
   └─ 完全独立，无 Metal 竞争   └─ 完全独立               └─ 完全独立
```

**这与 vLLM 的多 GPU worker 思路相同**，但在一个 M-series GPU 上通过 stream 隔离实现"逻辑多 GPU"。

---

## 十、内存管理：active / cache / peak / wired

### 10.1 MLX 内存模型

MLX 区分四类内存：

| API | 含义 | 单位 |
|-----|------|------|
| `mx.get_active_memory()` | **活跃**内存（仍在引用的数组） | bytes |
| `mx.get_cache_memory()` | **缓存池**（已释放但保留以便复用） | bytes |
| `mx.get_peak_memory()` | **峰值**（自上次 reset 以来最高值） | bytes |
| `mx.reset_peak_memory()` | 重置峰值 | - |

**三者关系**：

```
total memory = active + cache
peak memory = max(active) since last reset
```

### 10.2 mx.clear_cache 的工作原理

```python
import mlx.core as mx

a = mx.array([1, 2, 3])  # active: 24 bytes
b = a + 1                # active: 48 bytes (a + b)
del a                    # active: 24 bytes (only b)
mx.clear_cache()         # cache: 0 bytes (释放 b 的旧版本等)
```

**类比**：
- `active` = Python `gc.get_objects()` 中仍引用的对象
- `cache` = 已 del 但 Metal runtime 还没释放的 buffer（保留以便重用）

### 10.3 oMLX 的内存估算

```python
# omlx/memory_monitor.py
def get_memory_info() -> MemoryInfo:
    return MemoryInfo(
        total_bytes=get_max_working_set_bytes(),
        used_bytes=mx.get_active_memory(),
        available_bytes=get_max_working_set_bytes() - mx.get_active_memory(),
        utilization=mx.get_active_memory() / get_max_working_set_bytes(),
    )
```

### 10.4 mx.set_wired_limit：kernel 级 GPU 内存上限

```python
# Apple iogpu.wired_limit_mb
# 这是 macOS 内核参数：限制 GPU 可"wire"的物理内存量

mx.set_wired_limit(48 * 1024**3)  # 48 GB
# 设置后，Metal 最多 wire 48 GB 到 GPU page table
# 超出部分会触发 swap（很慢）
```

**为什么需要它？**

默认 `iogpu.wired_limit_mb` 由 macOS 动态决定，但经常设置过高（如 80%），导致：
- 系统整体 OOM（kernel pressure）
- 其他 app 被 swap

oMLX 在 `ProcessMemoryEnforcer` 中主动设置：

```python
# omlx/process_memory_enforcer.py
def _apply_metal_wired_limit(desired_bytes: int) -> tuple[int, int | None]:
    """通过 mx.set_wired_limit 限制 Metal wired 内存"""
    # 读取当前 iogpu.wired_limit_mb (sysctl)
    # 计算 desired (基于 memory_guard_tier)
    # 调用 mx.set_wired_limit
    # 返回 (applied, old_limit)
```

### 10.5 Apple Silicon 的 wired vs active

```
System RAM (128 GB M3 Max)
├── Wired (locked pages for GPU): 48 GB
│   └── MLX active arrays
│   └── KV cache blocks
├── Compressed: ~10 GB
│   └── macOS 内存压缩
└── Free/Swap: ~70 GB
```

**wired** 是 Metal 提交 GPU kernel 时锁定的物理页。一旦 wired，无法 swap 到 SSD。

**oMLX 的策略**：用 `mx.set_wired_limit` 把 wired 限制到不超过 (总 RAM - 8 GB)，给 macOS 系统和其他 app 留余地。

---

## 十一、MLX vs PyTorch vs JAX 对比

### 11.1 三框架架构对比

```
PyTorch                          MLX                          JAX
├── Tensor (eager)               ├── array (lazy)              ├── Array (lazy)
├── nn.Module                    ├── nn.Module                 ├── flax.stax
│   └── __call__                  │   └── __call__             │   └── nn.Module
├── autograd                      ├── mx.grad                    ├── jax.grad
├── torch.compile (optional)      ├── @mx.compile (optional)    ├── @jax.jit (required)
├── CUDA stream                   ├── Metal stream              ├── XLA device
└── DistributedDataParallel       └── Single device (no TP)     └── pjit/pmap (multi-device)
```

### 11.2 API 风格对比

```python
# PyTorch
import torch
x = torch.randn(2, 3, device='cuda', dtype=torch.float16)
y = x @ x.T
loss = y.sum()
loss.backward()

# MLX
import mlx.core as mx
import mlx.nn as nn
x = mx.random.normal((2, 3), dtype=mx.float16)
y = x @ x.T
loss = y.sum()
# 自动微分：
grad_fn = mx.grad(lambda p: (p @ p.T).sum())
dx = grad_fn(x)

# JAX
import jax.numpy as jnp
import jax
x = jnp.ones((2, 3), dtype=jnp.float16)
y = x @ x.T
loss = y.sum()
grad_fn = jax.grad(lambda p: (p @ p.T).sum())
dx = grad_fn(x)
```

### 11.3 三框架在 LLM 推理中的差异

| 维度 | PyTorch | MLX | JAX |
|------|---------|-----|-----|
| **模型加载** | `AutoModelForCausalLM.from_pretrained()` | `mlx_lm.load()` (需先转换) | `flax`/`optax` |
| **前向传播** | Eager，立即执行 | Lazy，eval 时执行 | 必须 jit |
| **KV cache** | 自定义或 HF cache | mlx.nn.Module 集成 | 自实现 |
| **生成循环** | `model.generate()` | `mlx_lm.generate()` | 自实现 |
| **批处理** | vLLM / TGI 包装 | `BatchGenerator` | 自实现 |
| **生产服务器** | vLLM / TGI / sglang | **oMLX** | 少 |
| **Apple Silicon** | MPS (实验) | **MLX (原生)** | 不支持 |

### 11.4 MLX 的关键优势（相对 PyTorch）

1. **真正的 Metal 后端**：PyTorch MPS 后端是"半成品"，Flash Attention 等 kernel 不完整
2. **Lazy 评估**：可做 kernel fusion，比 PyTorch eager 快
3. **统一内存**：零拷贝，PyTorch MPS 仍需走 CPU↔GPU 桥接
4. **更轻量**：无需 TorchScript、C++ extension

### 11.5 PyTorch 的关键优势（相对 MLX）

1. **生态**：HuggingFace、vLLM、TGI、DeepSpeed 全 PyTorch
2. **CUDA 优化**：Flash Attention v2/v3, FlashInfer 等多年优化
3. **分布式**：tensor parallel, pipeline parallel, expert parallel
4. **多后端**：CUDA, ROCm, XPU, MPS, CPU
5. **生产工具链**：TorchScript, ONNX export, TensorRT

---

## 十二、MLX 的优势与限制

### 12.1 MLX 的优势

✅ **统一内存**：M-series 128GB 远超消费级 GPU 显存

✅ **Zero-copy**：CPU↔GPU 共享地址空间

✅ **Lazy + 编译融合**：kernel 融合减少 launch overhead

✅ **NumPy API**：开发者学习成本低

✅ **Flash Attention**：mx.fast.scaled_dot_product_attention

✅ **量化原语**：mx.quantize / mx.dequantize / mx.quantized_matmul

✅ **开源 + Apple 官方维护**：质量保证 + 长期支持

### 12.2 MLX 的限制

❌ **仅 macOS**：不能跑 Linux/Windows（虽然能编译）

❌ **无 tensor parallel**：M-series 最多 192GB unified memory，但单 GPU

❌ **生态薄**：相比 PyTorch 差 5 年

❌ **多线程不稳**：需要 thread-local stream workaround

❌ **Compile cache 崩溃**：worker 线程销毁时的 C++ bug

❌ **Lazy 陷阱**：跨线程 stream 错误、RNG state bug

❌ **vLLM 级服务缺失**：需要 oMLX 来补齐

---

## 十三、MLX 在 oMLX 中的关键使用模式

### 13.1 加载模式：模型进入 MLX Array

```python
# omlx/utils/model_loading.py
def load_text_model(model_name, tokenizer_config=None, model_settings=None):
    from mlx_lm import load
    model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)
    # model 内部所有权重都是 mx.array
    # tokenizer 是 HF tokenizer (CPU)
    return model, tokenizer
```

### 13.2 推理模式：BatchGenerator + Lazy eval

```python
# omlx/scheduler.py
class Scheduler:
    def __init__(self, model, tokenizer, config, stream):
        # mlx_lm.BatchGenerator 包装
        self.generator = BatchGenerator(
            model=model,
            max_tokens=config.max_tokens,
            stream=stream,  # thread-local stream
        )
    
    def step(self):
        # 内部 BatchGenerator.step() 调用
        # 每次构建 lazy 计算图，eval 时才真正跑 Metal kernel
        ...
```

### 13.3 内存监控模式：mx.get_active_memory

```python
# omlx/process_memory_enforcer.py
def _current_usage_bytes() -> int:
    # 主机 RSS
    rss = get_process_rss()
    
    # MLX active (Metal wired)
    mlx_active = mx.get_active_memory()
    
    # Hot cache bytes
    hot_cache = ...
    
    return rss + mlx_active + hot_cache
```

### 13.4 量化模式：mx.quantize + mx.quantized_matmul

```python
# omlx/oq.py
def quantize_model(model, plan):
    for path, tensor in tree_flatten(model):
        if should_quantize(path, plan):
            # 调用 mx.quantize
            w_q, scales, biases, group_size = mx.quantize(
                tensor, bits=plan.bits, group_size=plan.group_size
            )
            # 替换原权重
            ...
```

### 13.5 持久化模式：mx.save_safetensors

```python
# omlx/oq.py
def save_quantized(model, output_dir):
    # MLX 原生 safetensors 保存
    weights = dict(tree_flatten(model))
    mx.save_safetensors(
        str(output_dir / "model.safetensors"),
        weights,
        {"format": "mlx"}
    )
```

### 13.6 Compile 模式：极少使用

```python
# oMLX 几乎不用 @mx.compile
# 原因 1: mlx_lm.BatchGenerator 内部已有 fused kernels
# 原因 2: server 环境下 compile cache 崩溃风险

# 仅在 patch 的特殊模型中使用：
# omlx/patches/deepseek_v4/deepseek_v4_model.py
@mx.compile  # 模型作者加的
def fused_mlp(x, w1, w2, w3):
    ...
```

---

## 十四、总结

### 14.1 MLX 是什么一句话总结？

> **MLX 是 Apple 为 Apple Silicon 设计的"PyTorch-like 张量库"，利用统一内存架构和 Metal GPU，提供 NumPy 风格的 API 和 lazy evaluation + JIT 编译。**

### 14.2 核心架构特征

```
MLX = NumPy-like API
    + Lazy evaluation (computational graph)
    + JIT compile (@mx.compile + CompilerCache)
    + Unified memory (zero CPU↔GPU copy)
    + Metal backend (Apple GPU kernels)
    + Quantization primitives (mx.quantize / dequantize / quantized_matmul)
    + SDPA / Flash Attention (mx.fast.scaled_dot_product_attention)
```

### 14.3 oMLX 与 MLX 的关系

```
MLX     →  GPU 计算原语
mlx-lm  →  LLM 模型 + 推理循环 (BatchGenerator)
oMLX    →  LLM 服务器 (FastAPI + 多模型管理 + Tiered KV cache)
```

oMLX **重度依赖** MLX 的 7 类 API：
1. 张量操作 (`mx.array`, `mx.matmul`, `mx.softmax`, etc.)
2. Lazy eval (`mx.eval`, `mx.synchronize`)
3. 内存管理 (`mx.get_active_memory`, `mx.clear_cache`)
4. Stream (`mx.new_thread_local_stream`, `mx.stream`)
5. Compile (`@mx.compile` 仅在 patch 模型用)
6. 量化 (`mx.quantize`, `mx.quantized_matmul`)
7. SDPA (`mx.fast.scaled_dot_product_attention`)

### 14.4 oMLX 对 MLX 的 6 类深度适配

| 适配 | 问题 | 解决 |
|------|------|------|
| Thread-local Stream | Metal 多线程崩溃 | `mx.new_thread_local_stream` + 替换 `mlx_lm.generation_stream` |
| 全局单线程 Executor | 命令缓冲竞争 | `ThreadPoolExecutor(max_workers=1)` |
| Compile Cache 崩溃 | C++ thread_local 析构 | ctypes 调 `compile_clear_cache()` |
| Sampling RNG state | mx.compile 不前进 RNG | 重写 sampling，去掉 @mx.compile |
| Lazy State 跨线程 | Stream 错误 | `materialize_lazy_state()` 在加载线程物化 |
| Model Ownership | BatchGenerator KV 冲突 | 全局 `ModelRegistry` 单例 |

### 14.5 MLX 在 LLM 推理生态中的位置

```
最底层 (GPU/硬件):
    Apple Metal → M-series GPU
底层 (张量库):
    MLX (Apple 官方) ← 在此层
中层 (模型库):
    mlx-lm / mlx-vlm / mlx-embeddings / mlx-audio
    dflash-mlx / paroquant
生产层 (服务器):
    oMLX ← 在此层
应用层:
    Claude Code / Cursor / Hermes Agent / OpenCode
```

---

## 附录：MLX 内部源码链接

- [MLX 主仓库](https://github.com/ml-explore/mlx)
- [MLX C++ 后端](https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal)
- [MLX JIT compile](https://github.com/ml-explore/mlx/blob/main/mlx/core/compile.cpp)
- [MLX Fast ops](https://github.com/ml-explore/mlx/tree/main/mlx/core/fast)
- [mlx-lm](https://github.com/ml-explore/mlx-lm)
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
- [mlx-examples](https://github.com/ml-explore/mlx-examples)

---

*文档生成时间：基于 omlx 仓库当前 HEAD（mlx 0.31.x）分析*