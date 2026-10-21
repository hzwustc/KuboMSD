# 开发与测试

## 环境

建议使用独立的 Python 3.11 环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-tested.txt -e '.[test]'
python -m pip install build wheel 'setuptools>=64'
```

生产入口为 `kubomsd-gpu` 和 `python -m kubo_msd`。`kubo_msd.cli`、CPU 算符和部分并行接口来自参考实现，用于测试，不能据此推断 GPU 版支持 CPU 回退或 MPI 生产计算。

## 回归测试

不需要 GPU 的参考测试：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/unit tests/integration \
  --ignore=tests/integration/test_gpu_cli_pipeline.py \
  --deselect=tests/integration/test_cli.py::test_cli_end_to_end_and_existing_directory
```

完整测试需要真实 GPU：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q --require-gpu
```

完整目录包含 GPU 构造与命令行测试；上面的 CPU 命令明确排除这些测试。部分 GPU 测试直接创建设备上下文。因此不能依靠默认跳过行为，在没有 GPU 的机器上运行完整目录。GitHub Actions 使用上面的 CPU 测试选择，不代表 GPU 验证。

数值修改应给出能区分旧行为和新行为的最小回归，并检查独立参考、尺度、单位和非正交度量。改动影响材料算例时，另记录真实输入对照，不能用小模型结果代替。

## 构建与安装验证

```bash
python -m build
```

构建产物位于 `dist/`。wheel 含 CUDA `.cu` 文件，在目标设备首次运行时 JIT 编译；不打包本机 `.so` 或编译缓存。

在源码目录之外创建一个干净环境，安装 wheel 后运行安装检查：

```bash
PROJECT_ROOT="$PWD"
python3.11 -m venv /tmp/kubomsd-wheel-check
/tmp/kubomsd-wheel-check/bin/python -m pip install \
  -c "$PROJECT_ROOT/constraints-tested.txt" \
  "$PROJECT_ROOT/dist/kubo_msd_gpu-1.0.4-py3-none-any.whl"
cd /tmp
/tmp/kubomsd-wheel-check/bin/python "$PROJECT_ROOT/scripts/validate_installed.py" \
  --project "$PROJECT_ROOT" --out /tmp/kubomsd-installed-results
```

临时环境与输出目录若已存在，应换一个新名称。此检查需要 GPU，会验证导入路径、四个 CUDA 源文件、空 JIT 缓存的首次构造，以及 EXACT、stepwise、spectral 三条路线。

## 目录约定

- `src/kubo_msd/`：计算代码。
- `examples/`：可运行的合成模型。
- `tests/`：回归和冻结参考数据。
- `benchmarks/`：小模型计时与积分容差检查。
- `scripts/server/`：单 GPU Slurm 模板。
- `docs/`：输入、数值定义、运行和验证说明。

计算输出放在 `runs/`，材料数据放在 `local_data/`，二者已加入忽略规则。虚拟环境、JIT 缓存、构建目录和大型验证目录也不提交。示例与测试所需的小型 HDF5/NPZ 是仓库内容，不应按后缀全部忽略。

问题报告请附软件版本、设备与 CUDA 信息、完整命令、最小输入、报错和必要的残差记录。涉及材料私有数据时，优先提供可以复现问题的缩小模型。

CI 使用 [actions/checkout](https://github.com/actions/checkout) 和 [actions/setup-python](https://github.com/actions/setup-python) 设置源码与 Python 环境；只需要仓库内容的读取权限。
