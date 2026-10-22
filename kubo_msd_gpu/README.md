# KuboMSD GPU

本目录是单 GPU 版本。克隆整个仓库后，先执行 `cd kubo_msd_gpu`，再运行以下安装和计算命令。此版本与仓库根目录的 KuboMSD 共用 Python 包名 `kubo_msd`，请分别安装在独立的虚拟环境中。

KuboMSD GPU 用于计算 Γ 点哈密顿量的能量分辨均方位移（MSD）、速度自关联函数（VAC）和扩散系数。支持非正交 LCAO 基组，提供全谱精确对角化和随机迹 KPM 两种求解方案。

当前版本为 **1.0.4**。程序采用单进程、单 NVIDIA GPU；算符构造、矩阵求解和主要输运运算在 GPU 上执行，CPU 负责输入解析、任务索引、小型系数和结果写出。运行时不依赖 RSATB、MPI 或 CPU C++ 扩展。

## 安装

已测试环境为 Linux、Python 3.11、CuPy 13.6.0 和 CUDA 12.x。GPU 需要支持所用的 float64、complex128 和稀疏矩阵运算；实际可计算规模取决于可用显存。

在项目根目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-tested.txt .
kubomsd-gpu capabilities
```

CUDA Toolkit 位于默认位置 `/usr/local/cuda` 时，可先执行 `source env-gpu.sh`；其他位置用 `KUBOMSD_CUDA_ROOT` 指定，集群环境也可以加载管理员提供的 CUDA 模块。首次运行会通过 NVRTC 编译包内的 CUDA 源文件。

## 快速运行

仓库提供一个两轨道模型，便于检查安装和输出：

```bash
kubomsd-gpu run \
  --input examples/nonorthogonal/model.json \
  --out runs/example-exact \
  --solver exact --axes x,y \
  --emin -0.2 --emax 0.3 --n-energy 11 \
  --tmax-fs 0.1 --dt-fs 0.02 \
  --sigma-ev 0.1 \
  --memory-limit-gb 2 --device-memory-limit-gb 2 --npz
```

输出目录必须尚不存在。成功后得到 `results.h5`、`metadata.json` 和可选的 `results.npz`。这个合成模型只用于功能检查，不能作为材料输运参数的依据。

真实 LCAO 输入包含 POSCAR、ABACUS 数值轨道文件和 DeePTB 格式的 `hamiltonians.h5`。程序构造重叠矩阵和偶极矩，再由同一组算符进行精确解或随机迹计算。**dftio 转换后的 H 应使用 `deeptb-real` 约定**；输入约定、缓存和完整命令见下列文档。

| 文档 | 内容 |
| --- | --- |
| [运行说明](docs/usage.md) | 算符构造、精确解、两种随机迹、资源参数、Slurm 脚本 |
| [输入格式](docs/inputs.md) | HDF5 分块、轨道顺序与符号、单位、缓存 |
| [数值方法与输出](docs/numerics.md) | 谱界、非正交度量、MSD/VAC/D、误差与结果读取 |
| [验证记录](docs/validation.md) | 合成模型测试、In₂O₃ 对照结果及限制 |
| [开发与测试](CONTRIBUTING.md) | 回归测试、安装验证、构建方法 |
| [版本记录](CHANGELOG.md) | 1.0 系列修正 |
| [代码与测试来源](docs/provenance.md) | 参考实现和测试数据说明 |

## 使用范围

本版处理无自旋的单 Γ 点模型，不提供多 k 点输运、MPI、多 GPU、电子–声子散射、陷阱模型或迁移率后处理。薄膜可以指定面内方向，但程序不会从目录名判断方向，也不按真空层体积自动归一化。

精确解始终需要完整本征系统，限制初态能窗不会消除其显存需求。随机迹需要收敛的谱界；Ritz 检查失败时程序会停止。当前 160 原子 In₂O₃ 算例已完成两类求解器的数值对照，800 原子块体仍有谱界未收敛和本机全谱显存不足的限制，详见验证记录。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。第三方依赖及材料沿用其自身许可证，来源说明见[代码与测试来源](docs/provenance.md)。
