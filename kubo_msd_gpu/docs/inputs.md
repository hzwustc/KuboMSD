# 输入格式

程序接受两类输入：原始 LCAO 模型，以及已经组装好的 `kubomsd-model-v1` 算符模型。文件路径都相对于 JSON 所在目录解析。

## 原始 LCAO 模型

以下展示 In/O 模型的字段结构。文件名需要对应实际文件；可直接运行的版本在 [raw_lcao 示例](../examples/raw_lcao/input.json)。

```json
{
  "version": 1,
  "structure": {"file": "POSCAR"},
  "model": {
    "type": "lcao",
    "spin": "none",
    "hamiltonian": {
      "path": "model/hamiltonians.h5",
      "format": "deeptb",
      "energy_unit": "eV",
      "index_base": 0,
      "orbital_convention": "deeptb-real"
    },
    "orbitals": {
      "dir": "orbitals",
      "names": ["In.orb", "O.orb"],
      "convention": "abacus-real"
    }
  }
}
```

POSCAR 中的原子顺序必须与 HDF5 的原子索引一致。轨道文件采用 ABACUS 数值轨道格式，元素名从文件内容读取；支持 s、p、d、f 轨道及多重径向函数。轨道不会被自动重新归一化。

HDF5 的顶层组为 `0`。每个数据集命名为 `i_j_Rx_Ry_Rz`，表示原子 i、j 和晶格平移 R 对应的 H 块；块的行列数应与两原子的轨道数一致。支持二维实数/复数数组，以及末维长度为 2 的实部、虚部表示。程序检查块形状、有限性及伴随一致性，并在 Γ 点求和。

`energy_unit` 支持 `eV`、`hartree`、`rydberg`；`index_base` 支持 0 或 1。默认分别为 eV 和 0。POSCAR 的晶格缩放按文件解析，内部长度为 Å。

## dftio、DeePTB 与 ABACUS 约定

H、S 和偶极矩必须处于相同的轨道顺序与相位约定。文件后缀 `.h5` 本身不能说明这些信息。

| 字段 | 所描述的对象 | 本版处理 |
| --- | --- | --- |
| `orbitals.convention` | ABACUS 数值轨道与积分基底 | 仅接受 `abacus-real` |
| `hamiltonian.orbital_convention` | HDF5 内 H 的轨道基底 | 接受 `deeptb-real` 或 `abacus-real` |

dftio 通常已把 ABACUS H 转为 DeePTB 的实球谐顺序和符号。此时设置 `hamiltonian.orbital_convention="deeptb-real"`，程序会把构造出的 S 和偶极矩变换到该基底，保留输入 H。`format="deeptb"` 时这个字段的默认值也是 `deeptb-real`；实际项目中建议显式写出。

只有确认 HDF5 仍然保存 ABACUS 原始轨道约定时，才设置 `"abacus-real"`。这是配置声明，不是从 H 的数值自动识别；自定义转换器产生的顺序或符号需要先核对。

## 完整缓存

`prepare --mode build` 将原始模型转换为完整缓存，供精确解和随机迹共用。缓存记录输入来源、积分精度、基底身份和校验信息。重复使用时，应保留整个缓存目录及其内部相对路径。

带有 `operator-basis-v2` 身份的缓存支持当前基底约定。1.0.0/1.0.1 生成的原始 LCAO GPU 缓存缺少该身份，不能继续使用，需要从原始输入重建。收紧积分容差时，也应重新构造满足新要求的缓存。

## 已组装的算符模型

[非正交示例](../examples/nonorthogonal/model.json) 给出了完整 JSON。核心字段为：

- `format: "kubomsd-model-v1"`，`kpoint: [0, 0, 0]`。
- `n_orbitals`、`orbital_order` 与矩阵维数、排列一致。
- `basis` 为 `orthogonal` 或 `nonorthogonal`；后者需要 `overlap`。
- H、S 和位置/速度矩阵使用协变表示，即普通矩阵元 `〈φ_i|A|φ_j〉`。
- `velocity_source` 使用单一来源：位置、显式速度或复合算符。不能混合不同来源。

对于显式输入的矩阵，提供者负责保证所有矩阵的基底和单位一致；原始 LCAO 路线的基底变换不会再次套用到这些矩阵上。不要把未经相同变换的 ABACUS S 与 DeePTB H 直接组合。

仓库中的 `.npz` 和 `.h5` 均为小型合成示例或测试数据，未包含实际材料的哈密顿矩阵、赝势或生产轨道文件。
