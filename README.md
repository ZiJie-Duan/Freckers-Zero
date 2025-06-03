# Freckers-Zero

A deep reinforcement learning-based two-player board game AI project, implemented using an AlphaZero-like architecture.

## Project Background

This is a personal exploration project on reinforcement learning and the AlphaZero algorithm, filled with the gap between ideals and reality:

To avoid designing heuristic functions for MinMax, I decided to try using AlphaZero's architecture to train an AI.
But, well, the results weren't great.

This project uses the exact same technical concepts as DeepMind's AlphaZero paper, but with slightly different implementation methods, especially in the Monte Carlo tree structure. During training, I attempted to reduce model complexity to lower data requirements, trying to train on a single RTX 4070Ti GPU, but the results were not ideal. I personally believe that due to the massive training data requirements, it was impossible to generate sufficient game data within a reasonable time frame, leading to rapid overfitting and the model falling into a single behavioral pattern.

The model's performance gradually improved during the first 200 iterations, then gradually declined after 200 iterations.

The following part is AI-generated, hope everyone has fun.

## Project Overview

**Freckers-Zero** is a complete reinforcement learning system specifically designed for a two-player strategy game called "Freckers" (Frog Jumping Game). This project combines deep neural networks with Monte Carlo Tree Search (MCTS), training powerful AI agents through self-play learning.

## Game Rules

**Freckers** is a two-player strategy game played on an 8×8 board:

- **Objective**: Move your 6 pieces (frogs) to the opponent's baseline to win
- **Initial Setup**:
  - Red player: 6 pieces on the top row (row 0) in columns 1-6
  - Blue player: 6 pieces on the bottom row (row 7) in columns 1-6
  - Lily pads: Initially distributed on row 1 and row 6 in columns 1-6, plus four corners
- **Action Selection**: Each turn, players can choose to:
  - **Move Frog**: Move forward, left, right, or diagonally forward (no backward movement)
    - Can jump to adjacent lily pads
    - Can jump over other frogs to reach lily pads (similar to checkers rules)
  - **Grow Lily Pads**: Generate lily pads in empty spaces around all your frogs in 8 directions
- **Game Limits**: Default 150-turn limit; if exceeded, winner is determined by piece positions

For detailed game rules, please refer to the `AI_2025_Game_v1.1.pdf` file.

## Technical Architecture

### Hybrid Language Implementation

- **Python**: Main AI training and game logic implementation
- **Rust**: High-performance game environment and computational acceleration tools
  - Python-Rust interoperability through PyO3 bindings
  - Performance optimization for critical computational parts, but not used for MCTS search itself

### Core Components

#### 1. Neural Network (`freckers/model.py`)
- **FreckersNet**: ResNet-based deep convolutional neural network
- **Input**: 16×8×8 state tensor (including historical states and player identification)
- **Output**:
  - Action probability distribution: 65×8×8 dimensions (64 position actions + 1 grow action)
  - Position value evaluation: Single numerical value
- **Custom Loss Function**: Specially designed MaskLoss for handling action probabilities and grow probabilities

#### 2. MCTS Agent (`freckers/mcts_agent.py`)
- Monte Carlo Tree Search guided by neural network integration
- Supports dual-model battle mode
- Includes exploration noise (Dirichlet noise) and temperature control mechanisms
- Uses Upper Confidence Bound (UCB) for node selection

#### 3. Game Engine (`freckers/game.py`)
- Complete Freckers game rule implementation
- Win/loss determination, state transitions
- Board visualization functionality (using emojis)
- Supports custom turn limits

#### 4. Self-Play Simulator (`freckers/simulator.py`)
- Executes AI vs AI self-play games
- Dynamic temperature adjustment strategy (gradually reducing exploration during gameplay)
- Integrated data recording functionality

#### 5. Data Management (`freckers/data_record.py`)
- Efficient HDF5 format data storage
- Supports incremental data writing
- Automatic value label updates (win/loss propagation)
- Concurrent-safe file operations

### Rust Acceleration Tools (`freckers_gym/`)

#### 1. **MctsAcc** - MCTS Accelerator
- Provides high-performance game state operations
- Fast action space computation
- State tensor conversion acceleration

#### 2. **RSTK** - Rust Toolkit
- General-purpose board analysis tools
- Fast legal action computation
- Board state visualization

#### 3. **Game** - Rust Game Engine
- Complete game logic Rust implementation
- High-performance state transitions
- Win/loss determination acceleration

#### 4. **TSPathGenerator** - Timestamp Path Generator
- Generates unique file paths with timestamps
- Supports file naming in multi-process environments

## Training Process

### Main Training Loop (`freckers/main.py`, `freckers/tmain.py`)
1. **Self-Play Simulation**: Conduct self-play games using MCTS + neural network
2. **Data Collection**: Collect game states, action probability distributions, and final results
3. **Neural Network Training**: Train policy network using collected data
4. **Model Evaluation**: Manually conduct battles between new and old models
5. **Iterative Updates**: Manually decide whether to replace the model based on evaluation results, continue to next training round

### Trainer (`freckers/trainer.py`)
- Manages neural network training process
- Supports checkpoint saving and recovery
- Integrated loss recording and performance monitoring

### Configuration Management
- Supports flexible hyperparameter configuration
- Multi-process training coordination
- Distributed computing support

## Project Structure

```
freckers-zero/
├── freckers/                 # Main Python implementation
│   ├── main.py              # Main training process
│   ├── tmain.py             # Multi-process training management
│   ├── model.py             # Neural network definition
│   ├── game.py              # Game logic
│   ├── mcts_agent.py        # MCTS implementation
│   ├── deep_frecker.py      # Model inference interface
│   ├── simulator.py         # Self-play simulator
│   ├── data_record.py       # Data management
│   ├── trainer.py           # Trainer
│   ├── test.py              # Testing and battle tools
│   ├── once.py              # Single training script
│   └── *.ipynb              # Jupyter analysis notebooks
├── freckers_gym/            # Rust acceleration tools
│   ├── src/
│   │   ├── lib.rs           # Python binding interface
│   │   ├── game.rs          # Game engine
│   │   ├── mcts_acc.rs      # MCTS accelerator
│   │   ├── rstk.rs          # Analysis toolkit
│   │   └── ts_path_generator.rs # Path generator
│   ├── Cargo.toml           # Rust project configuration
│   └── pyproject.toml       # Python packaging configuration
├── AI_2025_Game_v1.1.pdf    # Detailed game rules document
├── assi.py                  # Assistant script
└── note.txt                 # Project notes
```

## Runtime Environment

- **Python 3.8+**
- **PyTorch** (Deep learning framework)
- **Rust 1.70+** (For compiling acceleration components)
- **CUDA** (Optional, for GPU acceleration)

## Main Dependencies

- `torch`, `torchvision` - Deep learning
- `numpy`, `scipy` - Numerical computation
- `h5py` - Data storage
- `pyo3` - Python-Rust bindings
- `maturin` - Rust-Python project building

## Usage

1. **Compile Rust Components**:
   ```bash
   cd freckers_gym
   maturin develop --release
   ```

2. **Start Training**:
   ```bash
   cd freckers
   python main.py
   ```

3. **Multi-process Training**:
   ```bash
   python tmain.py
   ```

4. **Model Testing**:
   ```bash
   python test.py
   ```

## Key Features

- **High-performance Hybrid Implementation**: Python usability + Rust performance optimization
- **Complete Training Pipeline**: Fully automated process from self-play to model training
- **Flexible Configuration System**: Supports various hyperparameter tuning
- **Data Augmentation**: Supports data augmentation techniques like board rotation
- **Visualization Support**: Intuitive game state display and training monitoring

## Algorithm Innovations

- **Specialized Loss Function Design**: Tailored for Freckers game's special action space
- **Dynamic Temperature Adjustment**: Gradually reducing exploration during gameplay
- **Efficient State Representation**: 16-channel tensor containing history and player information
- **Special Handling of Grow Lily Pad Actions**: The "grow lily pad" action in Freckers is special - players can choose to do nothing and let lily pads grow naturally. This design caused serious problems during training: if the AI continuously chooses to grow lily pads, the game state remains identical, causing the evaluation function to return the same results, potentially leading MCTS simulation into infinite loops.

  To solve this problem, I designed a unique grow lily pad logic:
  - Expanded action space to 65 dimensions (8×8 position actions + 1 grow lily pad channel)
  - The 65th channel specifically handles grow lily pad decisions, bound to spatial relationships of frog positions
  - Neural network outputs grow lily pad intensity at all frog positions, compares average with threshold
  - Executes grow lily pad if above threshold, otherwise doesn't execute
  
  This design directly correlates the neural network's spatial relevance with grow lily pad behavior, while considering global relationships between frogs and lily pads - a major modification to traditional action space representation.

This project demonstrates how to apply modern deep reinforcement learning techniques to complex strategy games, serving as an excellent case study for learning and researching AI game agents.

---------------------------Cursor Claude-4-sonnet---------------------------

(Haha, I don't think this project is excellent, but Cursor's description is quite amusing. I really want to say, use my code with caution as I'm afraid you might get screwed over, but I still need to defend myself - showing off is still necessary, jobs still need to be found, and credentials still need to be built. Hope you have fun playing with it!)