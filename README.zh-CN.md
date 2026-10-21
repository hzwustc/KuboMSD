# KuboMSD

[English](README.md)

KuboMSD 根据固定的单粒子哈密顿量计算能量分辨的电子均方位移（MSD）、速度自关联函数和扩散系数，支持单个 Γ 点下的正交与非正交基组。

当前版本为 **1.1.0.dev4**，仍属开发版本。源码构建环境为 Linux、Python 3.10 及以上、支持 OpenMP 的 C++17 编译器；本次发布检查使用 Python 3.11。

## 功能

- 求解完整广义本征问题 `H C = S C ε`，支持 CPU、CPU/GPU 混合和单 GPU。
- 使用 Jackson 核的随机迹 KPM，提供逐步传播和谱重组两种算法。
- 使用 PCG 或可复用 Cholesky 分解求解非正交度量方程；PCG 可选对角、原子分块、空间分块和 IC(0) 预条件器。
- 根据 POSCAR、数值原子轨道和 DeePTB 格式的哈密顿量构造重叠与偶极矩阵；支持 s/p/d/f 轨道、多径向函数和周期像。
- 使用 OpenMP 加速算符构造和随机迹内核；可选 MPI 分配原子对构造任务或随机样本。

程序不计算自洽电子结构、总能量、原子力或结构弛豫，也不包含迁移率后处理、多 k 点积分、分布式完整对角化和电子—声子散射。

## 安装

在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[test]'
kubomsd --version
kubomsd capabilities
```

安装时会编译两个 C++ 扩展，需要可用的 C++17/OpenMP 工具链。CUDA、MPI 和编译要求见[安装说明](docs/INSTALL.md)。

## 运行示例

仓库内的示例均为合成模型，不需要下载真实材料数据。安装后，在仓库根目录执行：

```bash
kubomsd run \
  --input examples/two_level/model.json \
  --out runs/two_level_exact \
  --solver exact --backend cpu \
  --emin -1.5 --emax 1.5 --n-energy 61 \
  --sigma-ev 0.1 --axes x \
  --tmax-fs 1 --dt-fs 0.05 --npz
```

输出目录必须尚不存在。计算完成后生成 `results.h5`、`metadata.json`；加上 `--npz` 时还会生成 `results.npz`。

能量参数使用与哈密顿量一致的绝对能量，单位 eV。`--efermi-ev` 只记录费米能参考，不会自动平移能量网格。扩散系数单位为 Å²/fs，乘以 0.1 可转换为 cm²/s。

## 使用文档

- [运行方法](docs/USAGE.md)：完整对角化、随机迹、原始 LCAO 输入及 MPI。
- [输入格式](docs/INPUT_FORMAT.md)：矩阵定义、HDF5 块和单位。
- [输出格式](docs/OUTPUTS.md)：数据维度、估计器、有效性标记。
- [数值方法](docs/NUMERICS.md)：归一化、谱界、误差和收敛。
- [度量求解器](docs/STOCHASTIC_METRIC.md)：PCG、Cholesky 和预条件器。
- [基底约定](docs/HARMONIC_CONVENTIONS.md)：轨道顺序和相位。
- [测试说明](docs/VALIDATION.md)：可复现测试及本次发布检查。
- [参考资料](SOURCES.md)：方法文献和依赖来源。

软件测试通过只说明所测输入和数值路径满足检查条件。用于材料分析前，还需检查轨道基组、展宽、随机样本数、传播误差、模拟时长及有限尺寸效应。

## 许可证

本仓库暂未指定开源许可证。各项依赖沿用其自身许可证，详情见[参考资料与来源](SOURCES.md)。
