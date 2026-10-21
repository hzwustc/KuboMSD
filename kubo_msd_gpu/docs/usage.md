# 运行说明

以下命令在项目根目录、已安装 `kubo-msd-gpu` 的环境中执行。正式入口为 `kubomsd-gpu`，也可使用 `python -m kubo_msd`。每个 `--out` 都应是新目录。

## 环境检查

```bash
kubomsd-gpu capabilities --device 0
kubomsd-gpu run --help
kubomsd-gpu prepare --help
```

`capabilities` 检查当前设备和 CuPy/CUDA 栈。首次运行需要可用的 NVRTC、CUDA 头文件及可写的编译缓存；缓存位置可以通过 `CUPY_CACHE_DIR` 指定。记录驱动、runtime、NVRTC、CuPy 和 GPU 型号，便于复现实验。

公开入口固定使用 GPU 和串行进程。帮助中保留了一些参考后端的参数名称；传入 CPU、hybrid 或 MPI 设置会被拒绝。设备检查失败后不会切换到 CPU。

## 从 LCAO 输入构造算符

仓库内的 `raw_lcao` 示例含有合成 H 和数值轨道，能独立运行以下流程：

```bash
kubomsd-gpu prepare --input examples/raw_lcao/input.json \
  --out runs/raw-audit --mode audit

kubomsd-gpu prepare --input examples/raw_lcao/input.json \
  --out runs/raw-pilot --mode pilot --device-memory-limit-gb 2

kubomsd-gpu prepare --input examples/raw_lcao/input.json \
  --out runs/operators --mode build --table-cache runs/tables \
  --integral-tol 1e-5 --device-memory-limit-gb 2
```

`audit` 检查输入和构造计划，`pilot` 运行局部试算。只有 `build` 生成可以用于输运的完整算符缓存。成功缓存包含 `model.json`、`operators.h5`、校验信息和 `CACHE_READY` 标记。

`--table-cache` 只缓存积分数值表；它与完整算符缓存不同。`--integral-tol` 决定积分表精度，真实材料应做容差收敛检查。重叠和偶极矩的邻居支持由轨道截断半径确定，不以 H 的非零块范围代替。构造时始终保留 xyz 三个方向。

也可以直接把原始 `input.json` 传给 `run`。需要保存该次构造的算符时，加 `--cache-out runs/operators-direct`。

## 全谱精确解

```bash
kubomsd-gpu run --input runs/operators/model.json \
  --out runs/exact --solver exact --axes x,y,z \
  --emin -1 --emax 1 --n-energy 9 \
  --tmax-fs 0.04 --dt-fs 0.02 --sigma-ev 0.1 \
  --memory-limit-gb 2 --device-memory-limit-gb 2 --npz
```

默认能量核为 Gaussian，宽度由 `--sigma-ev` 指定。与 KPM 逐项对照时，应改用相同的 `--kernel jackson --n-moments ...`，并统一谱界、网格和方向。

`--initial-state-selection window --buffer-ev ...` 只限制外层初态求和。完整对角化、全部中间态和完整 DOS 分母仍然保留；窗口截断误差需要单独检查。

## 随机迹：逐时传播

```bash
kubomsd-gpu run --input runs/operators/model.json \
  --out runs/stepwise --solver stochastic-kpm \
  --stochastic-method stepwise --kernel jackson --n-moments 16 \
  --spectral-min -10 --spectral-max 10 \
  --nsamples 8 --sample-batch 2 --statistics-block 2 --seed 1729 \
  --axes x,y,z --emin -1 --emax 1 --n-energy 9 \
  --tmax-fs 0.04 --dt-fs 0.02 --segment-fs 0.02 --no-vac \
  --memory-limit-gb 2 --device-memory-limit-gb 2 --npz
```

`stepwise` 逐段推进随机向量，直接计算位移 MSD。默认还计算 VAC；这里只用 `--no-vac` 关闭额外的 VAC 路线。

## 随机迹：谱展开与 VAC 积分

```bash
kubomsd-gpu run --input runs/operators/model.json \
  --out runs/spectral --solver stochastic-kpm \
  --stochastic-method spectral --kernel jackson --n-moments 16 \
  --spectral-min -10 --spectral-max 10 \
  --nsamples 8 --sample-batch 2 --statistics-block 2 --seed 1729 \
  --axes x,y,z --emin -1 --emax 1 --n-energy 9 \
  --tmax-fs 0.04 --dt-fs 0.02 --integration-dt-fs 0.005 \
  --memory-limit-gb 2 --device-memory-limit-gb 2 --npz
```

`spectral` 从谱展开得到 VAC，再积分得到 MSD 和 D。`--integration-dt-fs` 控制内部积分网格，独立于输出的 `--dt-fs`，应逐步减小以检查积分误差。

随机迹只支持 Jackson 核。`--stochastic-method auto` 在计算开始前固定方法并写入元数据；没有已验证的性能标定时选择 `stepwise`。

上述谱界、矩数、时间和样本数只适用于合成示例。真实材料可省略两个谱界参数，由 GPU Ritz 估计；手动设置谱界也必须通过同一极值检查，不能绕过未收敛问题。

## 参数与资源

| 参数 | 含义 |
| --- | --- |
| `--emin`、`--emax` | 绝对能量，单位 eV；不会自动减去费米能 |
| `--efermi-ev`、`--efermi-source` | 记录费米能及来源，便于后处理 |
| `--energy-grid`、`--time-grid` | 从 JSON 数组读取自定义能量、时间节点 |
| `--axes x,y,z` | 三维笛卡尔方向；薄膜按实际取向指定，如 `x,y` |
| `--metric-rtol`、`--metric-atol` | 非正交度量方程的真残差容差 |
| `--metric-maxiter` | 每次度量求解的迭代上限 |
| `--nsamples` | 总随机样本数 |
| `--sample-batch` | 同时在设备上处理的样本数 |
| `--statistics-block` | 每个统计块的样本数，与设备批宽不同 |
| `--memory-limit-gb` | 主机工作内存预算，实际单位 GiB |
| `--device-memory-limit-gb` | GPU 工作内存预算，实际单位 GiB |

有效显存预算还受实际空闲显存约束。随机迹可以减小设备批宽，但最小工作集也放不下时会报 `RESOURCE_BLOCKED`。精确解的稠密本征系统工作区随轨道数平方增长。

发生失败时保留输出中的 `metadata.json` 和 `results.partial.h5`。不要把部分结果作为完整结果读取；修正配置后使用新输出目录。

## Slurm

`scripts/server/` 提供两个提交模板，均请求一个进程和一张 GPU。账户、分区、时限和主机内存应按集群设置补充。模板本身不安装软件。

```bash
export KUBOMSD_ROOT="$PWD"
export KUBOMSD_ENV="$PWD/.venv"
export INPUT="$PWD/runs/operators/model.json"
export OUT="$PWD/runs/slurm-exact"
export MEMORY_GB=2
export EMIN=-1 EMAX=1
export AXES=x,y,z
export N_ENERGY=9 TMAX_FS=0.04 DT_FS=0.02
sbatch scripts/server/submit_single_gpu_exact.slurm
```

随机迹使用 `submit_single_gpu_stochastic.slurm` 并指定新的 `OUT`。可用环境变量见 `run_single_gpu.sh`，包括 `METHOD`、`N_MOMENTS`、`NSAMPLES`、`SAMPLE_BATCH`、`STATISTICS_BLOCK`、`SEED` 和 `SPECTRAL_MIN/MAX`。这些脚本的默认网格用于短程检查，材料计算需要自行设置。
