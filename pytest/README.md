# pytest 批量测试

本目录提供基于 CSV 配置的批量测试脚本，通过 `run_from_csv.py` 读取 `test_cmd.csv` 中的用例并依次执行（或仅打印命令）。

`run_test.py` 中 `low_latency` 的压测参数语义如下：

- `--pressure-test`：开启压测，并使用 `tests/test_low_latency.py` 的默认轮数
- `--pressure-test 40`：开启压测，并向下游传递 `--pressure-test --pressure-test-rounds 40`

## 文件说明

| 文件              | 说明                                                    |
| ----------------- | ------------------------------------------------------- |
| `run_from_csv.py` | 主入口：从 CSV 构建命令并执行（或 dry-run 仅打印）      |
| `run_test.py`     | 底层测试执行脚本，支持单机/多机、intranode/internode 等 |
| `test_cmd.csv`    | 测试用例配置表，定义每条用例的参数与 mark               |

## 快速开始

以下命令均可直接复制执行（请将 `cd` 路径和 IP 按需替换）。

### 一键复制：所有常用命令

```bash
# ---------- 1. 进入脚本目录 ----------
cd /path/to/DeepEP/pytest

# ---------- 2. 取消所有过滤环境变量（恢复跑全部用例） ----------
unset TEST_TYPE_T MARK_T WORLD_SIZE_T NUM_TOKENS_T HIDDEN_T NUM_TOPK_T NUM_EXPERTS_T NIC_HANDLER_T LINK_MODEL_T NOISE_MODE_T TOPK_MODE_T

# ---------- 3. 单机：仅查看将要执行的命令（不执行） ----------
python run_from_csv.py --dry-run

# ---------- 4. 多机：先 dry-run 查看命令，再实际执行（1 server + 2 client） ----------
python run_from_csv.py --dry-run --server-ip=0.0.0.0 --client-ip=1.1.1.1,2.2.2.2
python run_from_csv.py --server-ip=0.0.0.0 --client-ip=1.1.1.1,2.2.2.2

# ---------- 5. 使用自定义 CSV ----------
python run_from_csv.py --csv=/path/to/my_tests.csv --server-ip=0.0.0.0 --client-ip=1.1.1.1

# ---------- 6. 只跑 test_type 包含 smoke 的用例 ----------
TEST_TYPE_T=smoke python run_from_csv.py --dry-run --server-ip=0.0.0.0 --client-ip=1.1.1.1
TEST_TYPE_T=smoke python run_from_csv.py --server-ip=0.0.0.0 --client-ip=1.1.1.1

# ---------- 7. 直接运行 low_latency 压测 ----------
python run_test.py -m low_latency --server-ip=0.0.0.0 --client-ip=1.1.1.1 --pressure-test
python run_test.py -m low_latency --server-ip=0.0.0.0 --client-ip=1.1.1.1 --pressure-test 40
```

## 命令行参数

| 参数          | 说明                      | 默认值                |
| ------------- | ------------------------- | --------------------- |
| `--csv`       | CSV 配置文件路径          | `pytest/test_cmd.csv` |
| `--dry-run`   | 只打印命令，不执行        | 无（加上即启用）      |
| `--server-ip` | 服务端 IP（仅支持单个）   | 无                    |
| `--client-ip` | 客户端 IP，多个用逗号分隔 | 无                    |

说明：

- **多机测试**：必须同时指定 `--server-ip` 和 `--client-ip`；仅指定 `--client-ip` 会跳过所有命令。
- **world_size**：由 CSV 的 `world_size` 列决定；若 `world_size > 1` 且未给 IP，或 client 数量不足，该用例会被跳过。

## CSV 配置说明（test_cmd.csv）

列含义简要说明：

| 列名 | 说明 |
| --- | --- |
| `name` | 用例名称，也作为日志文件名 `{name}.log` |
| `mark` | 测试类型：`intranode_normal` / `internode_normal` / `low_latency` |
| `world_size` | 进程数（1=单机，2=1 server+1 client，以此类推） |
| `num_tokens` / `hidden` | 测试参数，会转为 `run_test.py` 的 `--num-tokens`、`--hidden` 等 |
| `num_topk` / `num_experts` | 测试参数，会转为 `run_test.py` 的对应参数 |
| `link_model` | `pure_rdma` / `rdma_link`，rdma_link 会加 `--disable-nvlink` |
| `nic_handler` | `cpu` 或 `gpu`，cpu 时会加 `NVSHMEM_IBGDA_NIC_HANDLER=cpu` 等 |
| `noise_mode` / `topk_mode` | 对应 `--noise-mode` / `--topk-mode`（值为 none 则不传） |
| `env_export` | 执行前导出的环境变量（可选） |
| `num_topk` / `num_experts` | 测试参数，会转为 `run_test.py` 的对应参数 |
| `other_args` | 其它参数，会以 `-args ...` 形式传给 `run_test.py` |
| `test_type` | 用例标签，用于环境变量过滤（daily、smoke-daily；满足 noise_mode=none 且 topk_mode=random 且 hidden=7168 的用例为 smoke-daily） |

修改 CSV 后可直接再次运行 `run_from_csv.py`（建议先用 `--dry-run` 确认命令）。

### 测试用例组成方式（笛卡尔覆盖规则）

test_cmd.csv 中的用例按以下规则生成，覆盖各维度取值的笛卡尔积（world_size、link_model 按 mark 约束匹配，不独立笛卡尔）。

**维度与取值约束：**

- **mark**：三种 intranode_normal、internode_normal、low_latency
- **nic_handler**：两种 cpu、gpu
- **num_tokens**：两种 64、128
- **hidden**：两种 4096、7168
- **num_topk**：固定 8
- **num_experts**：固定 256
- **other_args**：固定空
- **test_type**：一般为 daily；满足 noise_mode=none 且 topk_mode=random 且 hidden=7168 的用例为 smoke-daily
- **noise_mode**：intranode_normal、internode_normal 仅 none；low_latency 四种 none、gemm、sleep、both
- **topk_mode**：intranode_normal、internode_normal 仅 random；low_latency 三种 random、uniform、no-routing
- **world_size**：intranode_normal 仅 1；internode_normal 为 2、3、4；low_latency 为 1、2、3、4
- **link_model**：intranode_normal 仅 pure_rdma；internode_normal 仅 rdma_link；low_latency 为 pure_rdma、rdma_link

在上述约束下，对每个 mark 内：除 world_size、link_model 按上述匹配外，其余维度做笛卡尔积生成用例。

**列顺序：**  
name → mark → world_size → num_tokens → hidden → num_topk → num_experts → link_model → nic_handler → noise_mode → topk_mode → env_export → other_args → test_type。

**行排序优先级（从左到右依次为键）：**  
mark → world_size → num_tokens → hidden → num_topk → num_experts → link_model → nic_handler → noise_mode（none 排最前）→ topk_mode。name、env_export、other_args、test_type 不参与排序。

## 环境变量过滤（可选）

通过环境变量可只运行 CSV 中部分用例（精确匹配对应列）：

| 环境变量        | 对应 CSV 列                                              |
| --------------- | -------------------------------------------------------- |
| `TEST_TYPE_T`   | `test_type`（包含匹配：CSV 的 test_type 包含该值即保留） |
| `MARK_T`        | `mark`                                                   |
| `WORLD_SIZE_T`  | `world_size`                                             |
| `NUM_TOKENS_T`  | `num_tokens`                                             |
| `HIDDEN_T`      | `hidden`                                                 |
| `NUM_TOPK_T`    | `num_topk`                                               |
| `NUM_EXPERTS_T` | `num_experts`                                            |
| `NIC_HANDLER_T` | `nic_handler`                                            |
| `LINK_MODEL_T`  | `link_model`                                             |
| `NOISE_MODE_T`  | `noise_mode`                                             |
| `TOPK_MODE_T`   | `topk_mode`                                              |

使用示例（含 `TEST_TYPE_T=smoke`、`unset` 等）见上方「快速开始」。
