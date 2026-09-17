---
title: DeepEP ACE Mode Programming Guide
description: DeepEP ACE 模式适用场景、zero-copy buffer、dispatch/combine 推荐流程及问题排查
tags: [MUSA, DeepEP, MCCL, ACE, MoE]
---

# DeepEP ACE Mode Programming Guide

本文介绍 DeepEP ACE 模式的适用场景、zero-copy buffer 约束、推荐的
`dispatch()` / `combine()` 调用方式，以及性能测试和问题排查方法。本文描述的是当前版本的
intranode ACE 路径；其他通信模式应以对应版本的发布说明为准。

## 1. 什么时候使用 ACE 模式

ACE 模式通过设备的 copy engine 承担主要数据搬运，目标是减少 EP 通信与专家计算对计算
MP 资源的竞争。它更适合以下场景：

- 单次 dispatch/combine 的 payload 较大，通信时间足以覆盖 ACE 的固定启动和调度开销；
- 希望把 EP 通信与 attention、expert/GEMM 等计算重叠；
- prefill、训练等吞吐优先的工作负载；
- 能使用两个 micro-batch 组成流水，并为两个在途 batch 创建两个独立 `deep_ep.Buffer` 的场景。

对于小 payload 或严格追求单 batch 最低时延的场景，普通 DeepEP 路径可能更快。ACE 的主要
收益来自大 payload 搬运，以及通信和计算重叠后的端到端吞吐提升，而不是保证每一种 shape
的单次调用时延都更低。

> ACE 的数据搬运主要由 copy engine 完成，但 routing、permute、unpermute 和必要的控制逻辑
> 仍可能执行 GPU kernel。因此更准确的描述是“降低通信对计算 MP 的占用”，不应把完整的
> dispatch/combine 路径理解为绝对不使用 MP。

## 2. Zero-copy 的含义和 buffer 约束

ACE 通信访问的是在 `Buffer` 初始化阶段分配并向 MCCL 注册的内存窗口。dispatch 和 combine
对 registered buffer 的处理方式不同：

```text
Dispatch:
  model tensor -> DeepEP permute/stage -> registered dispatch buffer -> ACE -> recv_x

Combine:
  expert output -> registered combine buffer -> ACE -> unpermute/reduce -> model tensor
                  ^
                  必须由 get_ace_combine_buffer() 获取
```

- `dispatch()` 会在内部把输入 permute 到 registered dispatch buffer。调用方可以继续传入普通
  MUSA tensor。
- `combine()` 不会把任意输入自动转换为 registered buffer。传给 `combine()` 的 `x` 必须是
  `get_ace_combine_buffer()` 返回的 tensor。
- 如果 `combine()` 同时传入 `topk_weights`，它也必须来自同一次
  `get_ace_combine_buffer(..., with_topk=True)` 返回的 registered buffer。
- 最优方式是让 expert/grouped-GEMM 直接把结果写入 registered combine buffer。若上游算子
  不支持指定输出 buffer，则先用 `copy_()` 把结果复制进去，再调用 `combine()`。后一种方式
  满足 ACE 通信的内存要求，但会多一次显式 device-to-device copy，因而不是端到端零拷贝。

`get_ace_combine_buffer(num_out_tokens, hidden, num_topk, ...)` 中的
`num_out_tokens` 表示 **combine 输入的 routed-row 数量**，通常应传
`recv_x.size(0)`。它不是 dispatch 前的本地 batch token 数，也不是 combine 后的输出 token 数。

## 3. 初始化和容量规划

创建 `Buffer` 时启用 ACE，并为实际工作负载显式设置容量：

```python
buffer = deep_ep.Buffer(
    group,
    num_nvl_bytes=num_nvl_bytes,
    num_rdma_bytes=0,
    use_ace=True,
    token_num=capacity_tokens,
    hidden_size=hidden,
    num_topk=num_topk,
)
```

容量应覆盖所有实际输入和路由偏斜：

- dispatch 输入行数必须不大于 `token_num`；
- combine registered buffer 可容纳的 routed rows 为 `token_num * num_topk`；
- `hidden_size` 和 `num_topk` 必须分别不小于运行时的 hidden size 和 top-k；
- 如果已知本 rank 的最大 combine 输入行数为 `max_combine_rows`，至少应满足：

  ```text
  capacity_tokens >= max(max_dispatch_tokens,
                         ceil(max_combine_rows / num_topk))
  ```

`max_combine_rows` 应包含专家路由不均衡、padding 和 `expert_alignment` 的影响。生产环境不要只按
平均路由量规划容量。

> **Deprecated:** 不推荐通过 `num_ace_buffers=2` 配合 `ace_buffer_idx` / `buffer_index`
> 在一个 `deep_ep.Buffer` 内划分多个 slot。该接口仅用于兼容已有集成。新的 two-batch overlap
> 集成应创建两个独立 `deep_ep.Buffer`，每个实例使用默认的单 buffer 配置。

## 4. 推荐的单 batch 调用流程

推荐按以下顺序调用：

1. 用 `get_dispatch_layout()` 计算 routing layout；
2. 调用 `dispatch()`，保留本次返回的 opaque `handle`；
3. 异步调用时，先让计算 stream 等待 dispatch event，再消费 `recv_x`；
4. 根据 `recv_x.size(0)` 获取 combine registered buffer；
5. 让 expert 算子直接写入该 buffer，或者把 expert 输出 `copy_()` 到该 buffer；
6. 调用 `combine()` 时传入 registered tensor 和同一 batch 的 `handle`；
7. 在消费 combine 输出或将同一个 `Buffer` 用于下一批数据前等待 combine event。

下面的示例展示兼容性最好的显式 `copy_()` 方式：

```python
import deep_ep


def ace_moe_forward(buffer, x, topk_idx, topk_weights,
                    num_experts, expert_forward):
    num_topk = topk_idx.size(1)

    # 1. Layout
    (num_tokens_per_rank, _, num_tokens_per_expert, _, layout_done) = \
        buffer.get_dispatch_layout(
            topk_idx,
            num_experts,
            async_finish=True,
        )

    # 2. Dispatch。ACE 路径内部会使用 registered dispatch buffer。
    (recv_x, recv_topk_idx, recv_topk_weights,
     num_recv_tokens_per_expert, handle, dispatch_done) = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=layout_done,
        async_finish=True,
        allocate_on_comm_stream=True,
    )

    # 3. expert_forward 在当前计算 stream 上消费 dispatch 输出。
    dispatch_done.current_stream_wait()
    expert_out = expert_forward(
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        num_recv_tokens_per_expert,
    )

    # 4. combine 输入必须位于 ACE registered buffer。
    ace_combine_x, _ = buffer.get_ace_combine_buffer(
        recv_x.size(0),
        recv_x.size(1),
        num_topk,
        with_topk=False,
    )
    ace_combine_x.copy_(expert_out)

    # 不要误传 expert_out；它不是 ACE registered buffer。
    combined_x, _, combine_done = buffer.combine(
        ace_combine_x,
        handle,
        async_finish=True,
    )
    combine_done.current_stream_wait()
    return combined_x
```

如果 expert 算子支持指定输出 tensor，应优先直接写入：

```python
ace_combine_x, _ = buffer.get_ace_combine_buffer(
    recv_x.size(0), recv_x.size(1), num_topk,
    with_topk=False,
)
expert_forward(..., out=ace_combine_x)
combined_x, _, combine_done = buffer.combine(
    ace_combine_x, handle, async_finish=True,
)
```

如果需要通过 combine 一起处理 `topk_weights`，应成对获取和写入 registered buffer：

```python
ace_combine_x, ace_combine_topk_weights = buffer.get_ace_combine_buffer(
    recv_x.size(0), recv_x.size(1), num_topk,
    with_topk=True,
)
ace_combine_x.copy_(expert_out)
ace_combine_topk_weights.copy_(weights_to_combine)

combined_x, combined_topk_weights, combine_done = buffer.combine(
    ace_combine_x,
    handle,
    topk_weights=ace_combine_topk_weights,
    async_finish=True,
)
```

## 5. 推荐的 two-batch overlap

单 batch 串行执行只能体现 ACE 通信本身的差异，不能充分体现低 MP 占用的价值。推荐为两个
micro-batch 创建两个独立 `deep_ep.Buffer`，把 batch 0 的 expert 计算与 batch 1 的 dispatch
重叠，再把 batch 1 的 expert 计算与 batch 0 的 combine 重叠。不要通过一个 `Buffer` 的两个
内部 slot 实现新集成。

两个 `Buffer` 必须在所有 rank 上以相同顺序完成 collective 初始化，并使用相同的 group、容量
和 shape 配置：

```python
import deep_ep


def create_ace_buffer(group, num_nvl_bytes, capacity_tokens,
                      hidden, num_topk):
    return deep_ep.Buffer(
        group,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=0,
        use_ace=True,
        token_num=capacity_tokens,
        hidden_size=hidden,
        num_topk=num_topk,
    )


# 每个实例使用默认 num_ace_buffers=1。
buffer0 = create_ace_buffer(
    group, num_nvl_bytes, capacity_tokens, hidden, num_topk,
)
buffer1 = create_ace_buffer(
    group, num_nvl_bytes, capacity_tokens, hidden, num_topk,
)
```

推荐流水如下：

```text
time -------------------------------------------------------------->

Buffer 0 comm:  dispatch[0] |              combine[0]
Buffer 1 comm:       dispatch[1] |                  combine[1]
compute stream:           expert[0] | expert[1]
                           ^ overlap  ^ overlap
```

推荐调度顺序如下：

1. 使用 `buffer0` 异步发起 batch 0 dispatch；
2. 使用 `buffer1` 异步发起 batch 1 dispatch；
3. 计算 stream 等待 batch 0 dispatch event，执行 batch 0 expert，并写入 `buffer0` 的 combine buffer；
4. 使用 `buffer0` 异步发起 batch 0 combine；
5. 计算 stream 等待 batch 1 dispatch event，执行 batch 1 expert，并写入 `buffer1` 的 combine buffer；
6. 使用 `buffer1` 异步发起 batch 1 combine；
7. 在消费各 batch 输出或将对应 `Buffer` 用于下一批数据前，等待各自的 combine event。

框架侧可以把上一节的逻辑拆成两个非阻塞函数，形成下面的稳态调度。这里的 `batch` 保存
`x/topk_idx/topk_weights`，`state` 保存 dispatch 的全部返回值；两个辅助函数内部的 API 调用
与上一节一致。

```python
# launch_dispatch 只排队 layout + dispatch，不等待整卡同步。
state0 = launch_dispatch(buffer0, batch0)
state1 = launch_dispatch(buffer1, batch1)

# 内部只等待各自 dispatch_done，然后执行 expert；expert 输出写入对应
# get_ace_combine_buffer()，最后异步排队 combine。
out0, combine_done0 = expert_then_launch_combine(
    buffer0, state0,
)
out1, combine_done1 = expert_then_launch_combine(
    buffer1, state1,
)

# 只在输出的真实消费者之前建立依赖。
combine_done0.current_stream_wait()
consume_batch0(out0)
combine_done1.current_stream_wait()
consume_batch1(out1)
```

这种顺序先把两个 dispatch 排到两个 `Buffer` 各自的 ACE/communication stream，再在 compute stream 上消费 batch 0，
从而让 `expert[0]` 与 `dispatch[1]`、`expert[1]` 与 `combine[0]` 获得重叠机会。关键要求是：

- 两个 batch 使用两个独立 `deep_ep.Buffer`，不要设置 `num_ace_buffers=2`；
- `handle`、event、`recv_x` 和 combine buffer 都按 batch、按所属 `Buffer` 保存，不可交叉使用；
- `buffer0.dispatch()` 返回的 `handle` 只能交给 `buffer0.combine()`；`buffer1` 同理；
- 同一个 `Buffer` 的上一轮 combine 未完成前，不得用于下一批数据；
- 不要在每个阶段调用全设备 `synchronize()`，否则会破坏 overlap；只在真正的数据依赖处等待
  `EventOverlap`；
- 所有 rank 必须以相同的 collective 顺序推进 batch。

实际流水线的最佳切分取决于 attention、dispatch、expert 和 combine 的耗时。若某一阶段明显更长，
应结合 trace 调整 micro-batch 大小，而不是假设四个阶段天然等长。

## 6. 常见错误与排查

### 6.1 Dispatch 成功，但 combine 报 MCCL `Invalid usage`

最常见原因是把普通 tensor 直接传给了 ACE combine：

```python
# 错误：expert_out 是普通 PyTorch/MUSA tensor
buffer.combine(expert_out, handle)

# 正确
ace_x, _ = buffer.get_ace_combine_buffer(
    expert_out.size(0), expert_out.size(1), num_topk,
    with_topk=False,
)
ace_x.copy_(expert_out)
buffer.combine(ace_x, handle)
```

按以下顺序检查：

1. 最终传给 `combine()` 的对象是否就是 `get_ace_combine_buffer()` 返回的 tensor；不要在
   `copy_()` 后仍误传原始 `expert_out`；
2. 如果传了 `topk_weights`，它是否也来自 registered combine buffer；
3. `num_out_tokens` 是否等于本次 dispatch 的 `recv_x.size(0)`；
4. `hidden`、dtype、contiguous layout 和 top-k 是否符合约束；当前 combine data buffer 为
   BF16 `[num_routed_rows, hidden]`；
5. 当前 `Buffer` 是否仍被上一批异步通信使用；
6. 所有 rank 是否以相同顺序调用了 dispatch/combine，且 `handle` 来自同一 batch、同一 `Buffer`；
7. DeepEP、MCCL、MUSA runtime 是否来自兼容的发布组合。

调试时可以保留明确的对象检查，防止传错 tensor：

```python
combine_input = ace_x
assert combine_input.data_ptr() == ace_x.data_ptr()
```

### 6.2 Buffer capacity assertion 或越界提示

- 增大构造 `Buffer` 时的 `token_num`、`hidden_size` 或 `num_topk`；
- 使用 routed rows，而不是原始 batch tokens，评估 combine 容量；
- 收集各 rank 的最大 `recv_x.size(0)`，不要只观察 rank 0 或平均值；
- 检查动态 batch、路由偏斜和 `expert_alignment` 是否超过初始化假设。

### 6.3 异步模式偶发错误或结果不稳定

- 消费 `recv_x` 前等待 dispatch event；
- 发起 combine 前，保证写入 registered combine buffer 的 expert/copy 操作已经成为通信 stream
  的依赖；默认当前 stream 依赖或显式 `previous_event` 二选一；
- 将同一个 `Buffer` 用于下一批数据前，等待该 `Buffer` 的 combine event；
- 把 dispatch 返回的 `handle` 当作 per-batch opaque object，不要修改内容或跨 batch 复用。

### 6.4 ACE 没有比普通模式快

这不一定是异常。先确认测量的是通信单次延迟，还是 two-batch overlap 后的端到端吞吐。小
payload 下 ACE 固定开销占比更高；如果又使用 `copy_()` staging，显式 copy 也会进入端到端成本。

