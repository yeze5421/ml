# 机器学习原子间势能（MLIP）完整项目示例

这是一个**可直接运行**的教学项目，目标是演示如何从零搭建一个简化版 MLIP 流程：

1. 用 Lennard-Jones 势自动生成训练标签（能量 + 力）
2. 构造原子局域描述符（径向高斯 + 截断函数）
3. 训练神经网络预测总能量
4. 通过自动微分得到原子受力
5. 保存训练 checkpoint，方便后续推理/继续训练

> 说明：该项目是教学级 toy implementation，不是工业级势函数。

---

## 项目结构

```text
.
├── mlip.py                      # 兼容入口
├── mlip_project/
│   ├── __init__.py
│   ├── cli.py                   # 命令行入口
│   ├── config.py                # 超参数配置
│   ├── data.py                  # 数据集构造
│   ├── descriptors.py           # 描述符
│   ├── model.py                 # 原子网络 + MLIP模型
│   ├── physics.py               # 物理标签（LJ）
│   └── train.py                 # 训练与checkpoint
├── tests/
│   ├── test_dataset.py
│   └── test_model.py
└── pyproject.toml
```

---

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

---

## 训练

```bash
python -m mlip_project.cli --epochs 30 --n-samples 800 --n-atoms 5 --device auto
```

也可运行兼容入口：

```bash
python mlip.py --epochs 30
```

训练结束后会在 `artifacts/` 下生成 checkpoint（默认 `tiny_mlip.pt`）。

---


## 打包下载（ZIP）

在项目根目录执行：

```bash
bash scripts/package.sh mlip_project.zip
```

执行后会在当前目录生成 `mlip_project.zip`，可直接下载/分发。

---

## 你可以如何扩展

- 将描述符替换为 SOAP/ACE/图网络消息传递
- 增加多元素类型（embedding）
- 在 loss 中显式加入 force loss（多任务训练）
- 接入 ASE/LAMMPS 的真实数据
- 加入周期边界条件（PBC）和邻居列表加速

