# MLIPX: PaiNN-style 机器学习原子势工程项目

MLIPX 是一个可训练、可评估、可推理、可扩展的原子势项目，面向 ASE/extxyz 数据，支持能量+力联合训练，默认采用 **PaiNN-style message passing**（标量+向量通道，支持从能量自动微分得到力）。

## 1. 项目结构

```text
.
├── configs/
│   ├── minimal.yaml
│   └── standard.yaml
├── data/
├── scripts/
│   ├── make_demo_dataset.py
│   ├── train.sh
│   ├── evaluate.sh
│   └── infer.sh
├── src/mlipx/
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── datamodule.py
│   │   ├── graph.py
│   │   └── types.py
│   ├── models/
│   │   ├── layers.py
│   │   ├── painn.py
│   │   └── model.py
│   ├── training/engine.py
│   ├── ase_ext/calculator.py
│   └── utils/
└── tests/
```

## 2. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## 3. 数据格式（ASE/extxyz）

每个结构需要至少包含：
- 原子序数（ASE 自动包含）
- positions
- `info['energy']`（总能）
- `arrays['forces']`
- cell / pbc（若有）

可先生成可跑通数据：

```bash
python scripts/make_demo_dataset.py --out data/demo.extxyz --n-samples 80
```

## 4. 训练

最小实验：

```bash
python -m mlipx.train --config configs/minimal.yaml
# 或 bash scripts/train.sh configs/minimal.yaml
```

正式实验建议：

```bash
python -m mlipx.train --config configs/standard.yaml
```

可命令行覆盖超参数：

```bash
python -m mlipx.train --config configs/minimal.yaml --epochs 10 --batch-size 8 --lr 3e-4
```

训练输出：
- `artifacts/.../best.pt`
- `artifacts/.../last.pt`
- `artifacts/.../history.json`

## 5. 评估

```bash
python -m mlipx.evaluate \
  --config configs/minimal.yaml \
  --checkpoint artifacts/minimal/best.pt \
  --split test \
  --out artifacts/test_predictions.csv
```

输出指标：Energy/Force 的 MAE 与 RMSE。

## 6. 推理

```bash
python -m mlipx.infer \
  --config configs/minimal.yaml \
  --checkpoint artifacts/minimal/best.pt \
  --input data/demo.extxyz \
  --output artifacts/infer.txt
```

## 7. ASE Calculator 接入

```python
from mlipx.config import load_config
from mlipx.ase_ext.calculator import build_calculator
from ase.io import read

cfg = load_config("configs/minimal.yaml")
calc = build_calculator(cfg, "artifacts/minimal/best.pt")
atoms = read("data/demo.extxyz", index=0)
atoms.calc = calc
print(atoms.get_potential_energy())
print(atoms.get_forces())
```

## 8. 当前实现说明

- 使用 cutoff 邻域图，处理不同原子数 batch。
- 使用 energy normalization（可选 per-atom）。
- 支持 AdamW、CosineLR、gradient clipping、AMP（CUDA时）、early stopping、checkpoint resume。
- stress/virial 目前未启用（可在后续版本加入基于应变导数的实现）。
