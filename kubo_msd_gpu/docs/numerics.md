# 数值方法与结果

## 非正交基组

精确解求完整广义本征问题：

$$
H C = S C\varepsilon,\qquad C^\dagger S C=I.
$$

传播和算符作用中的度量求解写成 `S Y = B`。PCG 按原方程检查每列真残差：

$$
\|B-SY\| \leq \max(\mathrm{atol},\mathrm{rtol}\,\|B\|).
$$

内部迭代使用较严格的停止目标，最终验收仍使用配置中的容差。求解失败会中止计算，不会通过放宽容差继续。

## 精确解与能量核

`exact` 使用完整本征系统计算态对求和。默认能量核是 Gaussian，也支持 Jackson。选择初态能窗只裁减初态索引 n；中间态 m 和 DOS 分母仍使用完整谱。

输出能量网格和用于 Chebyshev 缩放的谱区间是两个独立概念。不能把关注的费米能附近能窗当作完整谱界。

## 随机迹和谱界

`stochastic-kpm` 支持 `stepwise` 和 `spectral`，均采用 Jackson 核。随机向量在非正交度量下归一化，样本编号与设备批宽分离，因此调整批宽不会更换随机样本。

GPU 的广义 Ritz 检查在 S 内积下建立子空间，通过两遍修正 Gram–Schmidt 保持 `Q† S Q = I`，投影原始 H，并检查 `H v - λ S v` 的广义残差。两个独立起点使用种子 811、812；大矩阵每 16 个向量检查一次，当前子空间上限为 512。

自动谱区间包含宽度 1% 或残差对应的更大余量。这属于数值估计，不是完整谱包围的数学证明。手动给定谱界也必须严格包含已检查的极值及残差余量；不能通过输入更宽区间绕过 Ritz 未收敛。

矩数、样本数、传播容差和积分时间步分别控制不同误差，需要独立检查。固定参数下与参考实现一致，并不等于材料计算已经收敛。

## MSD、VAC 和扩散系数

对方向 a，程序保留以下关系所定义的不同估计器：

$$
D_a(E,t)=\int_0^t C_{vv,a}(E,u)\,du,
$$

$$
\operatorname{MSD}_a(E,t)=2\int_0^t(t-u)C_{vv,a}(E,u)\,du,
$$

$$
D_{a,\mathrm{gradient}}(E,t)=\tfrac12\frac{d\operatorname{MSD}_a(E,t)}{dt}.
$$

`vac_integral` 和 `msd_gradient` 分开保存。梯度直接使用输出时间节点，不平滑，也不删除负值或振荡。三维扩散系数取 `(Dx + Dy + Dz) / 3`；面内扩散取所声明的两个正交方向的平均。

单位为 eV、fs、Å² 和 Å²/fs。扩散系数从 Å²/fs 转为 cm²/s 时乘以 0.1。DOS 单位为 states/eV/system，不默认乘自旋简并度，也不按体积归一化。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `results.h5` | 完整的数值结果和元数据属性 |
| `metadata.json` | 实际配置、输入来源、求解诊断、运行状态 |
| `results.npz` | 可选导出，通过 `--npz` 启用 |
| `results.partial.h5` | 未完成的结果；不作为完整输出读取 |

主要数组按 `(axis, time, energy)` 排列；统计分块按 `(block, axis, time, energy)` 排列。方向别名数组为 `(time, energy)`。不同方法生成的估计器不同，应从元数据读取 `primary_estimator` 和 `primary_diffusion_estimator`，不要写死数组名称。

```python
import h5py
import json
import numpy as np

with open("runs/exact/metadata.json", encoding="utf-8") as stream:
    meta = json.load(stream)
assert meta["run_status"] == "complete"
estimator = meta["primary_estimator"]
with h5py.File("runs/exact/results.h5", "r") as f:
    energy = f["energy_eV"][:]
    time = f["time_fs"][:]
    msd = f[f"msd_A2/{estimator}"][:]
    valid = f[f"valid/{estimator}"][:]
    reason = f[f"reason_code/{estimator}"][:]
    print(energy.shape, time.shape, msd.shape)
    print("valid:", np.count_nonzero(valid), "/", valid.size)
```

`run_status=complete` 表示输出已完整提交。还应检查有效掩码、原因码、度量残差、范数漂移以及统计和积分收敛情况。

## 有效性与统计

原因码是可组合的位标志：

| 值 | 名称 | 含义 |
| --- | --- | --- |
| 1 | `LOW_DOS` | DOS 低于阈值 |
| 2 | `UNCERTAIN_TRACE` | 随机迹归一化的不确定性过大 |
| 4 | `NO_SELECTED_STATES` | 初态窗口内没有可用态 |
| 8 | `TRUNCATION_UNVERIFIED` | 初态截断误差尚未验证 |
| 16 | `INTEGRATION_UNVERIFIED` | VAC 积分误差尚未验证 |
| 32 | `INSUFFICIENT_BLOCKS` | 统计块不足 |

原因码不全是数值失效标记，须与对应 `valid` 数组一起读取。原始分子、DOS、分块和误差数据均保留；统计块不足时不能把缺失的误差条解释成零误差。当前比值统计采用配对 bootstrap，重采样 256 次，种子 21491。

## 显存和性能记录

全谱精确解的资源检查为稠密本征系统及库工作区预留约 `10 × 16 × N²` 字节，另加当前常驻算符等开销。这是当前实现的预算策略，不是仅保存本征向量的理论最小值。

阶段计时可能嵌套，不能直接相加。显存记录为阶段边界或外部采样，不代表瞬时硬件峰值。比较性能时应使用相同输入、误差标准和输出配置，并区分首次 JIT、冷表、热表和完整算符缓存。
