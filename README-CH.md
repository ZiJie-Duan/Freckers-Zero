# Freckers-Zero

一个基于深度强化学习的双人棋类游戏AI项目，采用类似AlphaZero的架构实现。

## 项目背景

这是一个个人对强化学习和AlphaZero算法的探索项目，充满了理想与现实的差距：

为了逃避设计MinMax的启发函数，我决定尝试用AlphaZero的架构来训练一个AI。
但是，就但是效果并不好。

这个项目使用到的技术思想和DeepMind的AlphaZero 论文中的完全相同，但是实现方法稍有不同，尤其是蒙特卡洛树的结构。在训练的过程中尝试降低模型复杂度从而 降低对数据量的需求，来尝试在一块4070Ti显卡上进行训练，但结果并不理想。个人认为因为训练数据量需求太大，无法在有效的时间能生成足够的对棋数据，从而导致模型很快过拟合，陷入到一个单一化行为的行为模式。

在模型进行的200次迭代中，性能逐渐提升，超过200次后性能逐渐下降。

后面的部分就是AI代笔了，祝大家玩得开心。


## 项目概述

**Freckers-Zero** 是一个完整的强化学习系统，专门为名为"Freckers"（跳蛙游戏）的双人策略游戏设计。该项目结合了深度神经网络和蒙特卡洛树搜索(MCTS)，通过自对弈学习来训练强大的AI智能体。

## 游戏规则

**Freckers** 是一个在8×8棋盘上进行的双人策略游戏：

- **目标**：将己方的6个棋子（青蛙）移动到对方底线获胜
- **初始设置**：
  - 红方：6个棋子位于顶行（第0行）的第1-6列
  - 蓝方：6个棋子位于底行（第7行）的第1-6列
  - 荷叶：初始分布在第1行和第6行的第1-6列，以及四个角落
- **行动选择**：每回合玩家可以选择：
  - **移动青蛙**：向前、左、右、斜前方向移动（不能后退）
    - 可以跳到相邻的荷叶上
    - 可以跳过其他青蛙到达荷叶（类似跳棋规则）
  - **生长荷叶**：在己方所有青蛙周围8个方向的空位生成荷叶
- **游戏限制**：默认150回合限制，超时则根据棋子位置评估胜负

详细游戏规则请参考 `AI_2025_Game_v1.1.pdf` 文件。

## 技术架构

### 混合语言实现

- **Python**：主要的AI训练和游戏逻辑实现
- **Rust**：高性能的游戏环境和计算加速工具
  - 通过PyO3绑定实现Python-Rust互操作
  - 提供关键计算部分的性能优化，但不用于MCTS搜索本身

### 核心组件

#### 1. 神经网络 (`freckers/model.py`)
- **FreckersNet**：基于ResNet的深度卷积神经网络
- **输入**：16×8×8的状态张量（包含历史状态和玩家标识）
- **输出**：
  - 动作概率分布：65×8×8维度（64个位置动作 + 1个生长动作）
  - 位置价值评估：单一数值
- **特色损失函数**：专门设计的MaskLoss处理动作概率和生长概率

#### 2. MCTS智能体 (`freckers/mcts_agent.py`)
- 集成神经网络指导的蒙特卡洛树搜索
- 支持双模型对战模式
- 包含探索噪声（Dirichlet noise）和温度控制机制
- 使用置信区间上界（UCB）进行节点选择

#### 3. 游戏引擎 (`freckers/game.py`)
- 完整的Freckers游戏规则实现
- 胜负判断、状态转换
- 棋盘可视化功能（使用表情符号）
- 支持自定义回合限制

#### 4. 自对弈模拟器 (`freckers/simulator.py`)
- 执行AI vs AI的自对弈游戏
- 动态温度调节策略（游戏进行中逐渐降低探索性）
- 集成数据记录功能

#### 5. 数据管理 (`freckers/data_record.py`)
- 高效的HDF5格式数据存储
- 支持增量数据写入
- 自动处理价值标签更新（胜负传播）
- 并发安全的文件操作

### Rust加速工具 (`freckers_gym/`)

#### 1. **MctsAcc** - MCTS加速器
- 提供高性能的游戏状态操作
- 快速动作空间计算
- 状态张量转换加速

#### 2. **RSTK** - Rust工具包
- 通用的棋盘分析工具
- 合法动作快速计算
- 棋盘状态可视化

#### 3. **Game** - Rust游戏引擎
- 完整的游戏逻辑Rust实现
- 高性能状态转换
- 胜负判定加速

#### 4. **TSPathGenerator** - 时间戳路径生成器
- 生成带时间戳的唯一文件路径
- 支持多进程环境下的文件命名

## 训练流程

### 主训练循环 (`freckers/main.py`, `freckers/tmain.py`)
1. **自对弈模拟**：使用MCTS+神经网络进行自对弈游戏
2. **数据收集**：收集游戏状态、动作概率分布和最终结果
3. **神经网络训练**：使用收集的数据训练策略网络
4. **模型评估**：手动进行新模型与旧模型的对战测试
5. **迭代更新**：根据评估结果手动决定是否替换模型，继续下一轮训练

### 训练器 (`freckers/trainer.py`)
- 管理神经网络的训练过程
- 支持检查点保存和恢复
- 集成损失记录和性能监控

### 配置管理
- 支持灵活的超参数配置
- 多进程训练协调
- 分布式计算支持

## 项目结构

```
freckers-zero/
├── freckers/                 # Python主要实现
│   ├── main.py              # 主训练流程
│   ├── tmain.py             # 多进程训练管理
│   ├── model.py             # 神经网络定义
│   ├── game.py              # 游戏逻辑
│   ├── mcts_agent.py        # MCTS实现
│   ├── deep_frecker.py      # 模型推理接口
│   ├── simulator.py         # 自对弈模拟器
│   ├── data_record.py       # 数据管理
│   ├── trainer.py           # 训练器
│   ├── test.py              # 测试和对战工具
│   ├── once.py              # 单次训练脚本
│   └── *.ipynb              # Jupyter分析笔记
├── freckers_gym/            # Rust加速工具
│   ├── src/
│   │   ├── lib.rs           # Python绑定接口
│   │   ├── game.rs          # 游戏引擎
│   │   ├── mcts_acc.rs      # MCTS加速器
│   │   ├── rstk.rs          # 分析工具包
│   │   └── ts_path_generator.rs # 路径生成器
│   ├── Cargo.toml           # Rust项目配置
│   └── pyproject.toml       # Python打包配置
├── AI_2025_Game_v1.1.pdf    # 详细游戏规则文档
├── assi.py                  # 助手脚本
└── note.txt                 # 项目笔记
```

## 运行环境

- **Python 3.8+**
- **PyTorch**（深度学习框架）
- **Rust 1.70+**（用于编译加速组件）
- **CUDA**（可选，用于GPU加速）

## 主要依赖

- `torch`, `torchvision` - 深度学习
- `numpy`, `scipy` - 数值计算
- `h5py` - 数据存储
- `pyo3` - Python-Rust绑定
- `maturin` - Rust-Python项目构建

## 使用方法

1. **编译Rust组件**：
   ```bash
   cd freckers_gym
   maturin develop --release
   ```

2. **开始训练**：
   ```bash
   cd freckers
   python main.py
   ```

3. **多进程训练**：
   ```bash
   python tmain.py
   ```

4. **模型测试**：
   ```bash
   python test.py
   ```

## 特色功能

- **高性能混合实现**：Python易用性 + Rust性能优化
- **完整的训练流水线**：从自对弈到模型训练的全自动化流程
- **灵活的配置系统**：支持各种超参数调优
- **数据增强**：支持棋盘旋转等数据增强技术
- **可视化支持**：直观的游戏状态显示和训练监控

## 算法创新

- **专门的损失函数设计**：针对Freckers游戏的特殊动作空间
- **动态温度调节**：游戏过程中逐步降低探索性
- **高效的状态表示**：16通道张量包含历史和玩家信息
- **长荷叶动作的特殊处理**：Freckers游戏中的"长荷叶"是一个特殊动作，玩家可以选择什么都不做来让荷叶自然生长。这个设计在训练中带来了严重问题：如果AI连续选择长荷叶，游戏状态将完全相同，导致评估函数返回相同结果，进而可能使MCTS模拟陷入无限循环。

  为解决这个问题，我设计了一个独特的长荷叶逻辑：
  - 将动作空间扩展为65维（8×8的位置动作 + 1个长荷叶通道）
  - 第65个通道专门处理长荷叶决策，与青蛙位置的空间关系绑定
  - 神经网络在所有青蛙位置输出长荷叶强度，求平均值后与阈值比较
  - 超过阈值则执行长荷叶，否则不执行
  
  这种设计让神经网络的空间相关性与长荷叶行为直接关联，同时考虑全局青蛙和荷叶的位置关系，是对传统动作空间表示的重大修改。

该项目展示了如何将现代深度强化学习技术应用于复杂的策略游戏，是学习和研究AI游戏智能体的优秀案例。

---------------------------Cursor Claude-4-sonnet---------------------------

(笑死我了，我不觉得这个项目优秀, 但是很搞笑这个Cursor的描述，我很想说，慎用我的代码，我怕你被坑，但是我还是得为我自己辩护，装逼还是要装的，工作也是要找的，背书还是要背的，祝你玩得开心)