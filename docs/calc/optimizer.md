# 优化器与梯度更新

## 概述

RecStore 采用分离式梯度更新架构：嵌入模块记录梯度，外部优化器统一处理，避免频繁同步与锁竞争。

## 架构

| 阶段 | 模块 | 文件路径 | 说明 |
|------|------|---------|------|
| 1 | 前向传播 | `src/python/pytorch/torchrec/EmbeddingBag.py` | 接收特征，调用 forward() |
| 2 | EmbeddingBag | `RecStoreEmbeddingBagCollection.forward()` | 执行 _RecStoreEBCFunction.forward |
| 3 | 梯度计算 | `loss.backward()` | PyTorch 自动求导，触发 backward |
| 4 | 梯度追踪 | `_DistEmbFunction.backward()` | 收集 (ID, grad) 对到 `_trace` |
| 5 | 梯度应用 | `SparseOptimizer.step()` | 遍历 _trace，调用 KVClient.update |
| 6 | PS 更新 | `KVClientOp.EmbUpdate()` | 参数服务器应用 SGD 更新 |

## 梯度追踪机制

### RecStoreEmbeddingBagCollection 梯度追踪

反向传播时，_RecStoreEBCFunction.backward 将梯度追踪到模块的 _trace：

**追踪流程**

| 步骤 | 代码位置 | 操作 | 输出 |
|------|---------|------|------|
| 1 | `src/python/pytorch/torchrec/EmbeddingBag.py:_RecStoreEBCFunction.backward` | `for i, key in enumerate(feature_keys)` | 遍历每个特征 |
| 2 | 同上 | `config_name = module._config_names[key]` | 获取嵌入表名 |
| 3 | 同上 | `ids_to_update = values_cpu[start:end]` | 提取该特征的 ID |
| 4 | 同上 | `grad_for_bag = grad_output_reshaped[sample_idx, i]` | 提取该特征的梯度 |
| 5 | 同上 | `module._trace.append((config_name, ids, grads))` | 追踪到 _trace |

**追踪内容**

| 字段 | 类型 | 说明 |
|------|------|------|
| config_name | str | 嵌入表名称 |
| ids_to_update | Tensor | int64, [N] 的 ID |
| grads_to_trace | Tensor | float32, [N, D] 的梯度 |

### DistEmbedding 梯度追踪

_DistEmbFunction.backward 对重复 ID 的梯度进行合并：

**梯度合并流程**

| 步骤 | 代码位置 | 代码 | 说明 |
|------|---------|------|------|
| 1 | `src/python/pytorch/recstore/DistEmb.py:_DistEmbFunction.backward` | `unique_ids, inverse = torch.unique(ids_cpu, return_inverse=True)` | 对 ID 去重，保留逆映射 |
| 2 | 同上 | `grad_sum = torch.zeros((unique_ids.size(0), grad_cpu.size(1)))` | 创建去重后的梯度容器 |
| 3 | 同上 | `grad_sum.index_add_(0, inverse, grad_cpu)` | 相同 ID 的梯度累加 |
| 4 | 同上 | `module_instance._trace.append((unique_ids, grad_sum))` | 追踪合并后的梯度 |

## SparseOptimizer

处理稀疏梯度的优化器

### 接口

```python
class SparseOptimizer:
    def step(self, modules: List[torch.nn.Module]):
        """处理所有模块的梯度追踪"""
        for module in modules:
            if hasattr(module, '_trace'):
                self._apply_gradients(module)
    
    def _apply_gradients(self, module):
        """应用模块内记录的梯度"""
```

### 工作流程

| 步骤 | 代码位置 | 代码 | 说明 |
|------|---------|------|------|
| 1 | 用户代码 | `optimizer = SparseOptimizer(kv_client, lr=0.01)` | 初始化优化器 |
| 2 | 模型 | `output = emb_module(features)` | 前向传播，执行嵌入查询 |
| 3 | 用户代码 | `loss.backward()` | 反向传播，收集梯度到 _trace |
| 4 | `src/python/pytorch/recstore/optimizer.py` | `optimizer.step([emb_module])` | 遍历 _trace，应用梯度 |
| 5 | 同上 | `emb_module.reset_trace()` | 清理追踪，为下一个 batch 做准备 |

### 梯度更新策略

#### SGD (默认)

```python
emb -= lr * grad
```

| 参数 | 说明 |
|------|------|
| lr | 学习率 |

#### Momentum (可选)

```python
v = momentum * v + grad
emb -= lr * v
```

| 参数 | 说明 |
|------|------|
| momentum | 动量系数 (0.9) |

#### Adam (可选)

```python
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
emb -= lr * m / (sqrt(v) + eps)
```

| 参数 | 说明 |
|------|------|
| beta1 | 一阶矩估计系数 (0.9) |
| beta2 | 二阶矩估计系数 (0.999) |
| eps | 数值稳定性 (1e-8) |

## C++ 侧梯度应用

### CommonOp.EmbUpdate

```cpp
virtual void EmbUpdate(
    const RecTensor& keys, 
    const RecTensor& grads
) = 0;
```

| 参数 | 说明 |
|------|------|
| keys | uint64 张量 [N] |
| grads | float32 张量 [N, D] |

### KVClientOp 实现

**梯度应用流程**

| 步骤 | 代码位置 | 代码 | 说明 |
|------|---------|------|------|
| 1 | `src/framework/op.h:KVClientOp::EmbUpdate` | `auto embeddings = ps_client_->Get(keys);` | 从 PS 读取当前嵌入 |
| 2 | 同上 | `embeddings[i][j] -= lr * grads[i][j];` | 应用 SGD: emb -= lr * grad |
| 3 | 同上 | `ps_client_->Put(keys, embeddings);` | 将更新后的嵌入写回 PS |

## 学习率调度

### 全局学习率

```python
optimizer.set_learning_rate(0.001)
```

影响所有嵌入表

### 按表学习率

```python
optimizer.set_learning_rate(table_name="user_emb", lr=0.001)
optimizer.set_learning_rate(table_name="item_emb", lr=0.01)
```

### 学习率衰减

```python
class ScheduledOptimizer:
    def step(self, epoch):
        # 指数衰减: lr = initial_lr * 0.95^epoch
        new_lr = self.initial_lr * (0.95 ** epoch)
        self.optimizer.set_learning_rate(new_lr)
```

## 梯度聚合与同步

### 内存中梯度聚合

梯度追踪在 Python 内存中进行，不涉及 PS 同步

```python
_trace = [
    (table_name, ids, grads),
    (table_name, ids, grads),
    ...
]
```

### 批量应用

SparseOptimizer 逐条处理 _trace，每个 (ids, grads) 对应一次 KVClientOp.EmbUpdate 调用

### 异步更新选项

使用异步写入加速梯度应用：

```python
async def apply_gradients_async(self, module):
    for table_name, ids, grads in module._trace:
        write_id = self.kv_client.emb_write_async(ids, updated_embs)
        self.pending_writes.append(write_id)
    
    # 等待所有写入完成
    for write_id in self.pending_writes:
        self.kv_client.wait_for_write(write_id)
```

## 分布式优化器考虑

对于多机多卡训练：

### All-Reduce 梯度聚合

```python
# 所有 worker 的梯度首先在本地聚合
local_grads = aggregate_trace(emb_module._trace)

# 跨 worker all-reduce
aggregated_grads = all_reduce(local_grads)

# 应用聚合后的梯度
optimizer.apply(aggregated_grads)
```

### 模型并行

```python
# 不同 worker 保存不同的嵌入表
worker_0: user_emb, item_emb
worker_1: category_emb, brand_emb

# 前向：拉取所需的嵌入
for table_name in required_tables:
    embs[table_name] = remote_pull(table_name, ids)

# 反向：梯度按表发送到对应 worker
for table_name, ids, grads in _trace:
    send_to_worker(table_name, ids, grads)
```
