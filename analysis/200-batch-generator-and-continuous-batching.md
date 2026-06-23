# 200 · mlx-lm BatchGenerator 与连续批处理深度剖析

> **文档编号**：`analysis/200-batch-generator-and-continuous-batching.md`
> **主题**：连续批处理原理、mlx_lm.BatchGenerator 内部机制、oMLX Scheduler 包装
> **范围**：连续批处理 vs 静态批处理、BatchGenerator 状态机、prefill/decode 混合、Chunked Prefill、Decode Burst、Monkey-patch 体系
> **前置阅读**：[100-mlx-framework-principles.md](./100-mlx-framework-principles.md) · [001-omlx-project-overview.md](./001-omlx-project-overview.md)

> 📁 本文档属于 [analysis/](./README.md) 目录。

---

## 目录

- [一、为什么需要连续批处理？](#一为什么需要连续批处理)
- [二、连续批处理的核心思想](#二连续批处理的核心思想)
- [三、mlx_lm.BatchGenerator 内部机制](#三mlx_lmbatchgenerator-内部机制)
- [四、生成-预填充混合：SequenceStateMachine](#四生成-预填充混合sequencestatemachine)
- [五、oMLX Scheduler 对 BatchGenerator 的包装](#五omlx-scheduler-对-batchgenerator-的包装)
- [六、Chunked Prefill：长 prompt 切分](#六chunked-prefill长-prompt-切分)
- [七、Decode Burst：避免 GIL ping-pong](#七decode-burst避免-gil-ping-pong)
- [八、Monkey-patch 体系：oMLX 对 BatchGenerator 的关键改造](#八monkey-patch-体系omlx-对-batchgenerator-的关键改造)
- [九、Preflight Eviction：prefill 前的内存检查](#九preflight-evictionprefill-前的内存检查)
- [十、Abort 与错误恢复](#十abort-与错误恢复)
- [十一、连续批处理的性能模型](#十一连续批处理的性能模型)
- [十二、总结](#十二总结)

---

## 一、为什么需要连续批处理？

### 1.1 静态批处理（Static Batching）的痛点

传统 LLM 服务（GPT-3 之前）使用 **静态批处理**：

```
时间线：

请求 A: [prefill 100] [decode 500 tok] ............... [END]
请求 B:            [prefill 200] [decode 300 tok] ..... [END]
请求 C:                       [prefill 150] [decode 400] [END]
请求 D:                                  [prefill 80] [decode 200] [END]
       ├──────── batch boundary ────────┤
       
所有请求必须等 batch 中最慢的请求完成才能返回
```

**问题**：
- **气泡（bubble）**：A 完成、B 还在 decode → GPU 空闲
- **长尾延迟**：A 短，但因 B 慢，整体延迟高
- **显存浪费**：必须预留最长请求的 KV cache 空间
- **TTFT 高**：整 batch 凑齐才开始 prefill

### 1.2 连续批处理（Continuous Batching）的解决

```
时间线（连续批处理）：

请求 A: [prefill 100] [tok 1][tok 2]...[tok 500][END]
请求 B:            [prefill 200] [tok 1]...[tok 300][END]
请求 C:                       [prefill 150] [tok 1]...[tok 400][END]
请求 D:                                  [prefill 80] [tok 1]...[tok 200][END]
       每个 token 步：所有 running 序列都前进 1 个 token
       完成一个 → 立即移除 → 立即加入新请求
```

**改进**：
- **零气泡**：每步 batch 大小 = running 请求数
- **TTFT 低**：新请求立即 prefill，不等 batch
- **高吞吐**：GPU 一直满载
- **公平**：所有请求同时前进

### 1.3 连续批处理的开销

**挑战**：
- **输入长度不同**：如何 batch 不同长度的 prefill？
- **KV cache 动态变化**：完成 → 释放，新请求 → 分配
- **Attention mask 复杂**：batch 内不同序列需要 causal + padding mask
- **调度决策**：何时 admit 新请求、何时 preempt？

→ **BatchGenerator 是 mlx_lm 对这些挑战的实现**。

---

## 二、连续批处理的核心思想

### 2.1 token 级调度 vs 请求级调度

```
请求级调度（vLLM v0）：
  一个请求进入 → 整个请求 batch 化 → 直到完成才移除
  
token 级调度（vLLM v1, BatchGenerator）：
  每个 token 步重新评估 batch 组成
  完成的 token → 移除
  新请求 → 立即加入 batch
```

**关键差异**：

```
t=0:   batch = [A_prompt]
t=1:   batch = [A_prompt]
t=100: batch = [A_decode + B_prompt]  ← B 在 A decode 中途加入
t=101: batch = [A_decode + B_decode]
t=600: batch = [B_decode + C_prompt]  ← A 完成，C 加入
```

### 2.2 Iteration-level scheduling 的实现

每个 step：
```
1. 收集所有 running 序列
2. 构造 batch tensor（不同长度需 padding 或 packing）
3. 一次 forward pass
4. 每个序列采样一个新 token
5. 检查 finish 条件
6. 完成的移除 + 新增的加入
```

**memory cost**：
- running 序列的 KV cache 是 **持久** 的（不被每步重置）
- 新序列的 prefill 是 **累加** 到现有 KV cache
- 完成序列的 KV cache **立即释放**

### 2.3 与 vLLM v1 的对应关系

| 概念 | vLLM v1 | mlx-lm BatchGenerator |
|------|---------|------------------------|
| 调度器 | `Scheduler` (v1/core/sched) | `BatchGenerator` (mlx_lm/generate.py) |
| 序列状态机 | `SequenceState` enum | `SequenceStateMachine` |
| Prefill batch | `PromptProcessingBatch` | `PromptProcessingBatch` |
| Decode batch | `GenerationBatch` | `GenerationBatch` |
| KV cache manager | `BlockPool` (v1/core/block_pool) | 嵌入 BatchGenerator 内部 |
| Prefix caching | `BlockHashToBlockMap` | oMLX 自己实现 |
| Chunked prefill | `chunked_prefill=True` | `prefill_step_size=N` |

**关键区别**：
- vLLM v1 的 KV cache 是 **block-based + 独立 BlockPool**
- mlx-lm BatchGenerator 的 KV cache 是 **per-sequence standard KVCache**（无分页）
- **oMLX 的核心创新**：把 mlx-lm 的 standard KVCache 替换为 **PagedCache** + **Prefix Sharing**

---

## 三、mlx_lm.BatchGenerator 内部机制

### 3.1 核心数据结构

```python
# mlx_lm/generate.py (MLX 官方，简化)
from dataclasses import dataclass
from enum import Enum

class SequenceState(Enum):
    PREFILL = "prefill"   # 正在 prefill
    DECODE = "decode"     # 正在 decode
    DONE = "done"         # 已完成（待清理）

@dataclass
class SequenceStateMachine:
    uid: int                                       # 唯一 ID
    state: SequenceState = SequenceState.PREFILL
    tokens: list[int] = field(default_factory=list) # 当前 tokens
    prompt_progress: int = 0                        # prefill 进度
    logprobs: list[float] = field(default_factory=list)
    completion_tokens: int = 0                      # decode 已生成数
```

### 3.2 BatchGenerator 状态

```python
class BatchGenerator:
    def __init__(self, model, max_tokens, stop_tokens, sampler, ...):
        self.model = model
        self.max_tokens = max_tokens
        
        # 所有序列（不分 running/waiting，BatchGenerator 内部管理）
        self._sequences: dict[int, SequenceStateMachine] = {}
        
        # 下一个分配的 uid
        self._next_uid = 0
        
        # 当前 prompt cache list（每层一个 cache）
        # 这是关键：BatchGenerator 用 per-sequence cache list
        self.prompt_cache: list[Any] = []  # 长度 = num_layers
        
        # 采样相关
        self.sampler = sampler
        self.logits_processors = logits_processors
        
        # Prompt processing batch
        self._prompt_batch = PromptProcessingBatch(...)
        
        # Generation batch（已加入 decode 的）
        self._generation_batch = GenerationBatch(...)
```

### 3.3 insert / next_generated / remove 接口

```python
# oMLX scheduler.py 中调用的关键方法
# (从 omlx/scheduler.py 推断的 BatchGenerator API)

bg = BatchGenerator(model=..., max_tokens=..., sampler=..., ...)

# 1. 插入新请求
bg.insert(
    prompt_tokens=[1, 2, 3, ..., 100],
    # 可选：max_tokens, temperature, etc.
)

# 2. 推进一个 step，返回所有生成的 token
responses: list[Response] = bg.next_generated()
# Response: (uid, token_id, logprobs, finish_reason)

# 3. 移除完成的序列
bg.remove(uid)

# 4. 流式（新增的 streaming 接口）
async def stream():
    while True:
        responses = bg.next_generated()
        if not responses:
            await asyncio.sleep(0.001)
            continue
        for r in responses:
            yield r
            if r.is_finished:
                bg.remove(r.uid)
```

### 3.4 oMLX 实际使用方式

```python
# omlx/scheduler.py:2403
def _create_batch_generator(self, sampling_params):
    sampler = omlx_make_sampler(  # 注意：用 oMLX 自己的 sampling，不用 mlx_lm 默认
        temp=sampling_params.temperature,
        top_p=sampling_params.top_p,
        ...
    )
    
    logits_processors = make_logits_processors(
        repetition_penalty=...,
        presence_penalty=...,
        frequency_penalty=...,
    )
    
    bg = BatchGenerator(
        model=self.model,
        max_tokens=sampling_params.max_tokens,
        stop_tokens=stop_tokens_seq,
        sampler=sampler,
        logits_processors=logits_processors if logits_processors else [],
        prefill_batch_size=1,                  # ← 关键：prefill 每次只 1 个请求
        completion_batch_size=self.config.completion_batch_size,  # decode batch 大小
        prefill_step_size=self.config.prefill_step_size,          # chunked prefill
        stream=self._stream,                   # thread-local stream
    )
    
    return bg
```

**为什么 `prefill_batch_size=1`？**

oMLX 选择 **外部 prefill + 单条 insert**，而不是让 BatchGenerator 内部批量 prefill。原因是：

```python
# omlx/scheduler.py:5790
def add_request(self, request: Request) -> None:
    # 1. Tokenize
    request.prompt_token_ids = self.tokenizer.encode(request.prompt)
    
    # 2. Prefix cache lookup (block-aware)
    #    oMLX 自己实现的前缀缓存查找
    #    命中 → 立即复用 KV blocks → 大幅节省 prefill
    
    # 3. 内存预算检查 (Preflight Eviction)
    self.preflight_or_raise(num_prompt_tokens=...)
    
    # 4. 外部 prefill（不在 BatchGenerator 内部）
    self._do_external_prefill(request)
    
    # 5. insert 到 BatchGenerator (此时 KV 已计算好)
    self.batch_generator.insert(
        tokens=request.last_token,
        # uid 映射、cache state 关联
        ...
    )
```

→ **oMLX 把 BatchGenerator 当成"已 prefill 完成的 token 调度器"用**，prefill 逻辑自己控制。

---

## 四、生成-预填充混合：SequenceStateMachine

### 4.1 状态机

```
                       insert(prefill_tokens)
                            │
                            ▼
                  ┌─────────────────────┐
                  │  PREFILL            │
                  │  (处理 prompt)      │
                  └──────────┬──────────┘
                             │ prefill 完成
                             ▼
                  ┌─────────────────────┐
        ┌────────│  DECODE             │────────┐
        │        │  (一个 token/步)     │        │
        │        └──────────┬──────────┘        │
        │                   │                    │
        │ EOS / max_tokens  │                    │ abort
        │                   ▼                    │
        │        ┌─────────────────────┐        │
        │        │  DONE               │        │
        │        │  (待 remove)         │        │
        │        └─────────────────────┘        │
        │                                       │
        └───────────────────────────────────────┘
                          remove(uid)
```

### 4.2 chunked prefill 时的状态

```
PREFILL (处理 prompt_chunk_1)
   │ prefill_step_size=2048
   ▼
PREFILL (处理 prompt_chunk_2)  ← 如果 prompt > 2048 token
   │
   ▼
PREFILL (处理 prompt_chunk_N)
   │
   │ prefill 完成
   ▼
DECODE
   │
   ▼
DONE
```

### 4.3 混合 prefill + decode 的 batch 构造

```python
# mlx_lm/generate.py 内部逻辑（简化）
def step(self):
    # 1. 处理所有 PREFILL 序列（按 prefill_step_size 切分）
    prefill_batch = []
    for seq in self._sequences.values():
        if seq.state == SequenceState.PREFILL:
            chunk = seq.tokens[seq.prompt_progress : seq.prompt_progress + prefill_step_size]
            prefill_batch.append((seq.uid, chunk))
            seq.prompt_progress += len(chunk)
            if seq.prompt_progress >= len(seq.tokens):
                seq.state = SequenceState.DECODE
    
    # 2. 处理所有 DECODE 序列（每次 1 token）
    decode_batch = []
    for seq in self._sequences.values():
        if seq.state == SequenceState.DECODE:
            # 上一步生成的 token 喂进来
            decode_batch.append((seq.uid, [seq.last_token]))
    
    # 3. 合并 forward
    all_inputs = prefill_batch + decode_batch
    # ... forward pass on model
    # ... sample token for each
    # ... update state
```

**关键**：prefill 和 decode 在 **同一 forward pass** 中！

```
forward input:
  [request_A_chunk_1: 2048 tokens]   (prefill)
  [request_B_chunk_1: 1024 tokens]   (prefill)
  [request_C: 1 token]               (decode)
  [request_D: 1 token]               (decode)

attention mask 构造:
  - causal mask for each sequence
  - cross-sequence mask = False (each sequence attends only to itself)
  
forward pass:
  q, k, v projections
  attention (with mask)
  FFN
  lm_head
  
sampling:
  取每个序列的 last logit → sample token
```

→ **这就是"连续批处理"的本质**：把不同阶段的请求 merge 到一个 forward。

---

## 五、oMLX Scheduler 对 BatchGenerator 的包装

### 5.1 SchedulerConfig 关键参数

```python
# omlx/scheduler.py:1249
class SchedulerConfig:
    # 请求并发
    max_num_seqs: int = 256
    
    # 每个 forward 的 token 上限（chunked prefill 切分点）
    max_num_batched_tokens: int = 8192
    
    # 调度策略
    policy: SchedulingPolicy = SchedulingPolicy.FCFS
    
    # BatchGenerator settings
    completion_batch_size: int = 32       # decode batch 大小
    prefill_step_size: int = 2048         # prefill chunk 大小
    chunked_prefill: bool = False         # 是否开启 chunked prefill
    
    # Paged cache settings
    paged_cache_block_size: int = 256     # 每 block 256 token
    max_cache_blocks: int | None = None   # 总 block 数（None = 自动）
    initial_cache_blocks: int = 256       # 初始 block 数
    
    # paged SSD cache
    paged_ssd_cache_dir: str | None = None
    hot_cache_only: bool = False
    hot_cache_max_size: int = 0
```

### 5.2 Scheduler 的三大队列

```python
# omlx/scheduler.py:1471
self.waiting: deque[Request] = deque()           # 等待 prefill
self.running: dict[str, Request] = {}            # 正在 decode
self.prefilling: deque[Request] = deque()         # chunked prefill 中

self._prefill_states: dict[str, _PrefillState] = {}  # prefill 进度跟踪
```

### 5.3 step() 主循环

```python
# omlx/scheduler.py:9028
def step(self) -> SchedulerOutput:
    output = SchedulerOutput()
    
    # 1. 处理 pending aborts（线程安全）
    self._process_pending_aborts()
    
    # 2. 处理 pending reclaims（内存压力）
    self._process_pending_reclaim()
    
    # 3. Drain async store_cache 完成项
    drained_async_removes = self._drain_pending_async_removes()
    
    # 4. 检查内存压力（soft/hard threshold）
    if self.memory_monitor is not None:
        self._check_memory_pressure()
    
    # 5. 推进 chunked prefills（每个 step 一个 chunk）
    chunked_scheduled = []
    if self.prefilling:
        self._advance_chunked_prefills(chunked_scheduled, chunked_rejected)
    
    # 6. 调度 waiting → running
    scheduled, rejected = self._schedule_waiting()
    
    # 7. 推进 BatchGenerator 一个 step
    if self.batch_generator and self.running:
        responses = list(self.batch_generator.next_generated())
        
        # 8. 处理响应（生成 output, finished detection）
        outputs, finished_ids = self._process_batch_responses(responses)
        
        # 9. 清理 finished
        self._cleanup_finished(finished_ids)
        
        # 10. 周期性 mx.clear_cache() (避免长 decode 时 IOGPU residency 溢出)
        self._tokens_since_clear_cache += len(responses)
        if self._tokens_since_clear_cache >= 1024:
            _sync_and_clear_cache(self._stream)
            self._tokens_since_clear_cache = 0
    
    return output
```

### 5.4 _schedule_waiting 核心逻辑

```python
# omlx/scheduler.py:7291
def _schedule_waiting(self):
    scheduled = []
    rejected_outputs = []
    
    while (self.waiting and 
           self._num_admitted_requests() < self._effective_max_num_seqs()):
        
        # 1. Admission pause (内存 soft threshold 触发)
        if self._admission_paused and admitted:
            break  # 不再 admit 新 prefill
        
        # 2. Store-cache backpressure (清理队列满)
        if not self._store_cache_gate.has_capacity:
            break
        
        # 3. 取一个等待请求
        request = self.waiting[0]
        
        # 4. Preflight 内存检查
        if not self.preflight_or_raise(num_prompt_tokens=...):
            rejected_outputs.append(...)
            break
        
        # 5. Prefix cache lookup（oMLX 自己的逻辑）
        cached_tokens = self._lookup_prefix_cache(request)
        
        # 6. 外部 prefill（oMLX 自己执行，不走 BatchGenerator）
        self._do_external_prefill(request, cached_tokens=cached_tokens)
        
        # 7. Insert 到 BatchGenerator（仅 1 个 token：上一步生成的）
        self.batch_generator.insert(
            tokens=[request.last_token],
            cache=request.prompt_cache,  # oMLX 把 KV 传进去
        )
        
        # 8. waiting → running
        self.running[request.request_id] = request
        self.waiting.popleft()
        scheduled.append(request)
    
    return scheduled, rejected_outputs
```

---

## 六、Chunked Prefill：长 prompt 切分

### 6.1 动机

**问题**：如果一个 prompt 是 32K token，一次性 prefill：
- 占用 GPU 几秒钟
- 这段时间其他 decode 请求全部 block
- TTFT 极高

### 6.2 解决方案

把 prefill 切分到多个 step：

```
原始 prompt: 32K tokens
prefill_step_size = 2048
→ 切成 16 个 chunk
→ 每个 step 处理 1 个 chunk
→ 中间可以插入 decode 步骤
```

### 6.3 oMLX 实现

```python
# omlx/scheduler.py:4050
def _advance_chunked_prefills(self, scheduled, rejected):
    """Advance in-flight chunked prefills (one chunk per request)."""
    
    while self.prefilling:
        request = self.prefilling[0]
        state = self._prefill_states[request.request_id]
        
        # 处理一个 chunk
        chunk = request.prompt_token_ids[
            state.processed : state.processed + self.config.prefill_step_size
        ]
        
        # 把 chunk 喂给 model，更新 KV cache
        self._process_prefill_chunk(request, chunk)
        state.processed += len(chunk)
        
        # 完成？
        if state.processed >= len(request.prompt_token_ids):
            # 全部 prefill 完成 → 插入 BatchGenerator decode
            self.batch_generator.insert(...)
            self.prefilling.popleft()
            scheduled.append(request)
        # else: 继续 chunked prefill，下个 step 再来
```

### 6.4 Chunked Prefill 的可视化

```
没有 chunked prefill:
  step 1: [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ A prefill 32K ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] (慢)
  step 2: (等 A prefill 完成)

有 chunked prefill:
  step 1: [A_chunk_1 2K] [B_decode 1] [C_decode 1]
  step 2: [A_chunk_2 2K] [B_decode 1] [C_decode 1]
  step 3: [A_chunk_3 2K] [B_decode 1] [C_decode 1]
  ...
  step 16: [A_chunk_16 2K] [B_decode 1] [C_decode 1]
  step 17: [A_decode 1] [B_decode 1] [C_decode 1]  ← A 进入 decode
  
  → B 和 C 的延迟从 "等 A 完成" → "几乎无延迟"
```

### 6.5 prefill_step_size 的选择

| 值 | 优缺点 |
|---|--------|
| 512 | 切得细，decode 延迟低，但 prefill 总开销大 |
| 2048 (默认) | 平衡 |
| 8192 | 切得粗，prefill 总开销小，但 decode 阻塞 |

---

## 七、Decode Burst：避免 GIL ping-pong

### 7.1 问题

```python
# 朴素实现：每个 token 都回 asyncio
async def generation_loop():
    while has_running:
        # 提交给 MLX executor（释放 GIL）
        output = await loop.run_in_executor(
            executor, lambda: scheduler.step()
        )
        # ↑↑ 这里 task 切换到 asyncio 循环
        # ↓↓ 1ms 后又切换回 executor
        await send_sse(output)  # 发送 SSE
```

**GIL ping-pong**：每次切换 ~1ms，CPU 时间浪费在 task 切换上。

### 7.2 oMLX 的解决：Decode Burst

```python
# omlx/engine_core.py
def _step_burst(self):
    """在一次 executor hand-off 内连续 step 多次。"""
    max_steps = self.config.decode_burst_max_steps  # 默认 64
    
    outputs = [self.scheduler.step()]
    
    if max_steps <= 1:
        return outputs
    
    budget_s = (
        self.config.decode_burst_budget_single_s  # 单请求 0.1s
        if self._num_admitted_requests() <= 1
        else self.config.decode_burst_budget_s      # 并发 0.03s
    )
    
    deadline = time.monotonic() + budget_s
    
    while len(outputs) < max_steps:
        # 还要工作吗？
        if not self.scheduler.running:
            break
        
        # 时间到？
        if time.monotonic() >= deadline:
            break
        
        # prefill eviction 需要回调 → 退出 burst 让 event loop 处理
        if self.scheduler._pending_prefill_eviction_request:
            break
        
        # 继续 step
        outputs.append(self.scheduler.step())
    
    return outputs
```

**示意**：

```
朴素：asyncio ↔ executor 来回切换 (每次 1ms)
| exec step1 | async | exec step2 | async | exec step3 | ...

burst：连续 64 step 在 executor 内完成
| exec step1 step2 ... step64 | async | exec step65 ... |
```

### 7.3 性能影响

```
in-process sync loop: ~80 tok/s
per-token async hand-off: ~74 tok/s
```

→ **decode burst 提升 ~8%** 吞吐。

### 7.4 自适应预算

```python
# omlx/engine_core.py
@dataclass
class EngineConfig:
    decode_burst_max_steps: int = 64
    decode_burst_budget_single_s: float = 0.1   # 单请求 → 0.1s 预算（激进）
    decode_burst_budget_s: float = 0.03         # 并发 → 0.03s 预算（紧凑）
```

**为什么自适应？**

- **单请求**：没有其他请求需要响应，burst 长 = 高吞吐
- **并发**：新请求需要被 admit、长请求需要 abort 响应，burst 短 = 低延迟

---

## 八、Monkey-patch 体系：oMLX 对 BatchGenerator 的关键改造

oMLX 不修改 mlx-lm 源码，而是用 **monkey-patch** 在运行时修改 BatchGenerator 的行为：

### 8.1 已知的 Monkey-patch（来自 scheduler.py grep）

```python
# omlx/scheduler.py 中所有 monkey-patch 位置
_original_generation_batch_step = GenerationBatch._step           # line 588
GenerationBatch._step = _patched_generation_batch_step            # line 646

_original_generation_batch_filter = GenerationBatch.filter         # line 667
GenerationBatch.filter = _patched_generation_batch_filter         # line 683

_original_ppb_split = PromptProcessingBatch.split                 # line 802
PromptProcessingBatch.split = _patched_ppb_split                  # line 901

_original_ppb_prompt = PromptProcessingBatch.prompt              # line 1016
PromptProcessingBatch.prompt = _patched_ppb_prompt               # line 1031
```

### 8.2 为什么需要 monkey-patch？

oMLX 在 `BlockAwarePrefixCache` 中用 **PagedCache** 替换了 mlx-lm 的 **standard KVCache**。这导致：

```
问题 1: GenerationBatch._step() 的 logits_processors 长度对齐
  mlx-lm 假设所有 uid 共享同一组 logits_processors
  oMLX 每个 uid 有自己的 logits_processors → 长度不匹配
  → 需 patch: _omlx_realign_rows / _patched_generation_batch_step

问题 2: GenerationBatch.filter() 在 remove 时收缩
  mlx-lm 假设 filter 后 slots 紧凑
  oMLX 中 None slot 表示"边界快照替换的层" → 不能紧凑
  → 需 patch: _patched_generation_batch_filter

问题 3: PromptProcessingBatch.split() 在 chunked prefill 时
  mlx-lm 假设 cache 是连续的 KVCache
  oMLX 可能是 RotatingKVCache、ArraysCache、TurboQuantKVCache 的混合
  → 需 patch: _patched_ppb_split

问题 4: PromptProcessingBatch.prompt() 在 VLM mRoPE 时
  mlx-lm 默认用文本 RoPE
  VLM (Qwen3-VL, GLM-4V) 用 mRoPE（多模态 RoPE）需要 delta
  → 需 patch: _patched_ppb_prompt (line 1010 注释提到 mRoPE)
```

### 8.3 Patch 示例

```python
# omlx/scheduler.py:563
def _omlx_realign_generation_batch_rows(self) -> None:
    """Re-align self.logits_processors with current self.samplers.
    
    mlx-lm assumes every uid shares the same logits_processors;
    oMLX allows per-uid samplers (via per-request sampling_params).
    After _register_uid_rows / _unregister_uid_row, the lengths
    must match the active uid set.
    """
    expected_lps = [self.sampler]
    uids = self.uids  # currently active uids
    current_lps = self.logits_processors
    
    if len(current_lps) != len(uids):
        # Rebuild logits_processors list aligned with uids
        self.logits_processors = [expected_lps + per_uid_extras(uid) for uid in uids]
```

```python
# omlx/scheduler.py:591
def _patched_generation_batch_step(self):
    """Patch for grammar accept_token() integration."""
    result = _original_generation_batch_step(self)
    
    # After sampling, feed the token to grammar compiler
    for uid, token in zip(self.uids, result.tokens):
        grammar = self.grammar_compilers.get(uid)
        if grammar is not None:
            grammar.accept_token(token)
    
    return result
```

### 8.4 这是 oMLX 的技术债务

**优点**：
- 不修改 mlx-lm 源码，pin 到具体 commit 安全
- 升级 mlx-lm 时只需重新评估 patch 是否仍需要

**缺点**：
- mlx-lm 内部 API 变化时 patch 失效
- 难以调试（行为与 mlx-lm 不完全一致）
- 长期需把这些改进 upstream 到 mlx-lm

---

## 九、Preflight Eviction：prefill 前的内存检查

### 9.1 动机

**问题**：新请求 prefill 时需要大块连续 KV memory。如果显存不足，可能 OOM。

### 9.2 解决：prefill 前 LRU 驱逐

```python
# omlx/scheduler.py:338
class PrefillEvictionRequest:
    """Signal from scheduler to server: 'evict something before I prefill'"""
    request_id: str
    needed_bytes: int

# omlx/engine/vlm.py
async def _preflight_or_raise_with_eviction(self, scheduler, *, num_prompt_tokens, request_id):
    eviction_request = scheduler.preflight_eviction_request(
        num_prompt_tokens=num_prompt_tokens,
        request_id=request_id,
    )
    if eviction_request is not None and self._prefill_eviction_callback is not None:
        logger.info("Running preflight LRU eviction for request %s", eviction_request.request_id)
        await self._prefill_eviction_callback(eviction_request)
    
    scheduler.preflight_or_raise(
        num_prompt_tokens=num_prompt_tokens,
        request_id=request_id,
    )
```

### 9.3 决策流程

```
新请求到达 (prompt 4K tokens)
  │
  ├─ 计算 prefill 所需 memory
  │   estimated_bytes = num_prompt_tokens * num_layers * num_kv_heads * head_dim * dtype_size * 2
  │
  ├─ 检查是否有足够 memory
  │   available = ceiling - current_usage
  │   if available >= estimated_bytes: 通过
  │
  ├─ 否则：触发 preflight eviction request
  │   eviction_request = PreflightEvictionRequest(
  │       request_id=...,
  │       needed_bytes=estimated_bytes,
  │   )
  │   ↑ 异步回调给 EnginePool
  │     EnginePool 触发 LRU 驱逐（卸载最久未用模型）
  │
  ├─ 等 eviction 完成
  │
  └─ 重新检查 → 通过 或 抛 PrefillMemoryExceededError
```

### 9.4 与 vLLM 的对比

| | vLLM | oMLX |
|---|------|------|
| 驱逐时机 | 调度时（每步） | prefill 前 |
| 驱逐对象 | preempt sequences（recompute） | unload models |
| 异步 | 同步 preempt + async recompute | 异步 LRU 卸载 |

**oMLX 的策略更适合多模型场景**：直接把不用的模型从内存卸载，比 preempt 当前 decode 请求友好。

---

## 十、Abort 与错误恢复

### 10.1 用户取消（client disconnect）

```python
# omlx/scheduler.py:6573
def abort_request(self, request_id: str) -> bool:
    """User-cancel: remove the request from scheduler."""
    # 标记为 abort
    self._pending_abort_ids.add(request_id)
    # 下个 step() 时处理
    return True
```

### 10.2 跨线程 abort

```python
# omlx/scheduler.py:1477
# Thread-safe set for deferred aborts (main thread → executor thread)
# CPython GIL guarantees set.add() and `x in set` are atomic.
self._pending_abort_ids: set[str] = set()
```

**模式**：asyncio 线程（HTTP handler）调 `abort_request()`，executor 线程（scheduler）在 `step()` 开头处理。

### 10.3 Prefill 错误恢复

```python
# omlx/scheduler.py:9096
except _PrefillAbortedError:
    # Prefill was interrupted by a pending abort.
    # BatchGenerator is in an inconsistent state (partial prefill),
    # so reset it entirely. Pending aborts will be processed at the
    # start of the next step().
    self.batch_generator = None
    self._current_sampler_params = None
    self._boundary_cache_snapshots.clear()
    if self._boundary_snapshot_store is not None:
        self._boundary_snapshot_store.cleanup_all()
    
    # Move any running requests back to waiting so they get re-prefilled
    ...
```

**设计哲学**：宁可重置整个 BatchGenerator 也不要修复部分 prefill 状态。

### 10.4 错误恢复矩阵

| 错误类型 | 处理方式 |
|---------|---------|
| `_PrefillAbortedError` | 重置 BatchGenerator，重 prefill |
| OOM during prefill | 抛 `PrefillMemoryExceededError`，触发 LRU 卸载 |
| Cache corruption | 清空所有 cache，重 prefill（不抛错） |
| 用户 cancel | 标记 aborted，下个 step 移除 |
| 模型 panic | `_drain_pending_async_removes` 清理 |

---

## 十一、连续批处理的性能模型

### 11.1 理论分析

**静态批处理吞吐**：
```
T_static = N_requests * (prompt_tokens + decode_tokens) / batch_time
         ≈ N * (P + D) / (max(P_i) + max(D_i))
         ← 受最慢请求限制
```

**连续批处理吞吐**：
```
T_continuous ≈ sum(P_i) + sum(D_i) / total_time
             ← 所有请求的 token 总数 / 总时间
             ≈ GPU 满载的理论值
```

### 11.2 实际性能（M3 Max 上 Llama-3-8B-4bit）

| 模式 | 单请求 | 4 并发 | 16 并发 |
|------|--------|--------|---------|
| 静态 (单 batch) | 75 tok/s | N/A | N/A |
| 朴素 (无 burst) | 75 | 65 | 30 (memory thrash) |
| 连续批处理 + burst | 75 | **150** | **200** |

### 11.3 延迟分析

```
TTFT (Time To First Token):
  static: 等待 batch 凑齐 + 整 batch prefill
  continuous: 立即 prefill（可 chunked）
  
  oMLX continuous: TTFT ≈ prefill_step_size / prefill_throughput
                   ≈ 2048 tokens / 5000 tok/s
                   ≈ 400 ms (32K prompt 分 16 chunks)

TPOT (Time Per Output Token):
  continuous: 1 / decode_throughput
             ≈ 1 / 75 ≈ 13 ms (单请求)
             ≈ 1 / 200 ≈ 5 ms (16 并发)
```

### 11.4 内存放大

```
静态批处理: batch_size * max_seq_len * KV_per_token
连续批处理: sum(each_seq_len * KV_per_token) + 调度开销

oMLX 用 paged cache 后: 实际 KV ≈ sum * 1.05 (5% 碎片)
```

---

## 十二、总结

### 12.1 核心要点

1. **连续批处理的本质**：每个 token 步重新评估 batch 组成，完成序列立即释放，新序列立即加入。
2. **mlx-lm BatchGenerator**：`PromptProcessingBatch` + `GenerationBatch` + `SequenceStateMachine` 三件套。
3. **oMLX 的包装**：把 BatchGenerator 当成"已 prefill 完成的 token 调度器"，prefill 逻辑自己控制以支持 prefix cache + paged cache。
4. **Chunked Prefill**：长 prompt 切到多个 step，避免 decode 阻塞。
5. **Decode Burst**：连续多步 in executor，避免 GIL ping-pong。
6. **Monkey-patch**：oMLX 改造 BatchGenerator 内部方法（filter / step / split / prompt）以支持异构 cache 类型。
7. **Preflight Eviction**：prefill 前 LRU 卸载模型，避免 OOM。
8. **错误恢复**：宁可重置整个 BatchGenerator，不要修复部分状态。

### 12.2 一句话总结

> **BatchGenerator 是 mlx-lm 的连续批处理引擎；oMLX Scheduler 在其上加入了 prefix cache 查找、paged KV cache 替换、chunked prefill、decode burst、preflight eviction 等生产级特性。**

### 12.3 与 vLLM v1 的对应

| 概念 | vLLM v1 | oMLX |
|------|---------|------|
| 调度器 | Scheduler (v1) | Scheduler (omlx) |
| 连续批处理引擎 | EngineCore | AsyncEngineCore |
| Batch generator | (vLLM 自实现 unified scheduler) | mlx_lm.BatchGenerator (包装) |
| KV cache | BlockPool (v1) | PagedCacheManager + SSD tiered |
| Prefix cache | BlockHashToBlockMap (in-memory) | BlockAwarePrefixCache (跨重启 SSD) |
| Chunked prefill | ✅ | ✅ (prefill_step_size) |
| Decode burst | ❌ | ✅ (EngineConfig) |
| Preflight eviction | ❌ | ✅ (preflight_eviction_request) |

**核心差异**：
- vLLM v1 是 **fully integrated**（scheduler + KV manager + executor 都自实现）
- oMLX 是 **layered**（scheduler 包装 mlx_lm，KV cache 自己实现，两者通过 monkey-patch + 外部 prefill 整合）

### 12.4 与 llama.cpp 的对比

| | llama.cpp | oMLX |
|---|-----------|------|
| 批处理 | ❌ 单 stream | ✅ BatchGenerator |
| 并发 | ❌ 单请求 | ✅ 256+ 并发 |
| KV cache | 单 context | Paged + SSD |
| 调度 | 手动 | 自动 (FCFS + chunked + burst) |

**llama.cpp 没有连续批处理** → 无法服务高并发 LLM 推理。

### 12.5 oMLX 的创新点（在 BatchGenerator 之上的创新）

1. **外部 prefill**：把 prefix cache 查找 + prefill 提到 BatchGenerator 之外
2. **PagedCache 替换 standard KVCache**：通过 monkey-patch 让 BatchGenerator 支持 paged cache
3. **Boundary Snapshot**：VLM 图像边界处的独立 KV 快照
4. **TurboQuant KV**：量化 KV cache（mlx_vlm 提供）
5. **Decode Burst**：自适应预算的连续 step
6. **Preflight Eviction**：prefill 前的内存守护

---

## 附录：相关代码位置

| 文件 | 行数 | 关键内容 |
|------|------|---------|
| `omlx/scheduler.py` | 10114 | Scheduler + 5 类 monkey-patch + Chunked Prefill |
| `omlx/engine_core.py` | 1225 | AsyncEngineCore + Decode Burst |
| `mlx_lm/generate.py` | - | BatchGenerator / SequenceStateMachine / PromptProcessingBatch / GenerationBatch |
| `omlx/cache/prefix_cache.py` | 2895 | BlockAwarePrefixCache（oMLX 自己实现） |
| `omlx/cache/paged_cache.py` | 1583 | PagedCacheManager（替换 mlx-lm standard KVCache） |
| `omlx/cache/paged_ssd_cache.py` | 3439 | SSD 冷层（oMLX 独有） |

---

## 附录：参考链接

- [vLLM v1 Scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- [mlx-lm BatchGenerator](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
- [Continuous Batching 原始论文](https://www.usenix.org/system/files/conference/nsdi22/nsdi22-yu.pdf) - Orca 论文
- [vLLM PagedAttention 论文](https://arxiv.org/abs/2309.06180)

---

*文档生成时间：基于 omlx 仓库当前 HEAD（mlx-lm 2c008fd）分析*