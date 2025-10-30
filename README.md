# iChess.io v12.27 — Advanced Chess Engine built on Darknet Foundations

[![Build Darknet & iChess.io (Multi-Arch)](https://github.com/sowson/ichess.io/actions/workflows/ciaction.yml/badge.svg)](https://github.com/sowson/ichess.io/actions/workflows/ciaction.yml)

📘 **Author:** Piotr Sowa — Founder of [iSowa.io](https://iSowa.io), creator of [iChess.io](https://iChess.io)  
🧠 **Blog:** [iBlog.io](https://iBlog.io)

---

<img width="1536" height="1024" alt="ichess io" src="https://github.com/user-attachments/assets/cae8fcc8-f637-4a03-bc54-dea8ccf69e69" />

## 🚀 Overview

**iChess.io v12.27** is the newest generation of the *Darknet-based* chess engine — a fully integrated, GPU-accelerated AI system for classical and Fischer 960 chess.  
It merges a **policy CNN network** with a handcrafted, dynamically tuned **Piece-Square Table (PST)** evaluation system, delivering explainable, deterministic strength without relying on NNUE weights.

---

## 🧩 Core Highlights

- 🧠 **Hybrid policy CNN + context-aware PST evaluation**
- 🎯 **Dual-network system** — *policy (p)* for move guidance and *value (v)* for outcome estimation
- ♟️ **Adaptive PST model** — phase-weighted, mobility-adjusted, and material-sensitive heuristics
- 🔁 **Monte Carlo Tree Search (MCTS)** with top-N pruning and temperature scheduling
- ⚙️ **Self-play learning loop** — automatic dataset generation (FEN + π + v) and replay training
- 🚀 **OpenCL GPU acceleration** with full CPU fallback
- 🖥️ **Cross-platform** — macOS (Intel/Apple Silicon) and Linux x64
- 🎮 **UCI-compatible** for Arena, CuteChess, and external GUIs
- 📜 **Explainable AI** — no NNUE, complete transparency of heuristic scores

---

## 🧠 Engine Architecture

| Component | Description |
|------------|-------------|
| **Policy Network (p)** | Darknet CNN producing prior probabilities for legal moves |
| **Value Network (v)** | Estimates position outcome; guides back-propagation in MCTS |
| **PST Evaluator** | Context-aware positional heuristic based on handcrafted tables, tuned per game phase |
| **MCTS Core** | Parallel search with cpuct scaling, Dirichlet noise, and virtual loss handling |
| **Trainer** | Self-play data pipeline generating supervised targets for both networks |

### Behavioral Flow
1. **Self-play** produces (FEN, π, v) triplets.
2. **Policy/Value networks** update using accumulated replay data.
3. **PST tables** auto-rebalance per phase (opening, middlegame, endgame).
4. **MCTS** integrates priors + PST scores for move selection.

---

## 📈 Performance Characteristics

- Deterministic evaluation (< 2 ms per node)
- Efficient multi-threaded MCTS arena
- High throughput GPU kernels via OpenCL core
- Consistent play-strength growth during long self-play sessions

---

## 🧰 Build Instructions (macOS / Ubuntu 20.04+)

```bash
# Clone and prepare
mkdir iChess.io.en && cd iChess.io.en
git clone --recursive https://github.com/sowson/darknet

# Build libchess (used in the chess example)
cd darknet/cmake/libchess && mkdir build && cd build
cmake .. && make -j
cp shared/libchess.* ../../../3rdparty/libchess/

# Build (optional) clBLAS (used in the chess example)
cd darknet/cmake/clBLAS && mkdir build && cd build
cmake ../src -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_TEST=OFF -DBUILD_PERFORMANCE=OFF && make -j
cp shared/clBLAS.* ../../../3rdparty/clBLAS/

# Build engine with chess example enabled
cd ../../../.. && mkdir darknet/build && cd darknet/build
cmake -DDARKNET_ENG_CHESS=1 .. && make -j

# Copy example config and weights
cp ../cfg/chess.cfg ../../ && cp ../weights/chess.weights ../../
```

---

## 🧪 Example Run (UCI Mode)

```bash
./iChess.io.en
iChess.io v12.27 by Piotr Sowa

position startpos moves e2e4 b8c6 d2d4
go

info depth 1 pv e7e5
bestmove e7e5 ponder e7e5
```

---

## ⚙️ Dependencies

- [nlohmann/json](https://github.com/nlohmann/json)
- [libchess](https://github.com/sowson/libchess) (for chess logic)

---

## 🧠 Optimization Tips

- Use a **RAMDisk** for temporary training files to minimize I/O latency:
  - Linux: `sudo mount -t tmpfs -o size=4096M tmpfs /ramdisk`
  - macOS: `diskutil erasevolume HFS+ "ramdisk" $(hdiutil attach -nomount ram://8388608)`
- Replace clBLAS with **CLBlast** for optimized GEMM kernels:
  ```bash
  git apply patches/clblast.patch
  ```

---

## 🧩 Platform Support

- ✅ macOS (Intel / Apple Silicon)
- ✅ Ubuntu Linux 20.04 or newer
- ⚠️ Windows 10/11 (experimental OpenCL build)

Windows build guide: [Darknet on OpenCL on Windows 11 x64](https://iblog.io/2021/11/20/darknet-on-opencl-on-windows-11-x64)

---

## 📖 Scientific Foundation — Darknet v1.1.1

> *This project builds upon the Darknet v1.1.1 AI CNN Computer Vision Engine*  
> 📄 Reference paper: [https://doi.org/10.1002/cpe.6936](https://doi.org/10.1002/cpe.6936)

Darknet provided the original multi-GPU CNN core, which now serves as the foundation for policy/value training inside iChess.io v12.27.

---

## 🙏 Credits & Acknowledgements

Developed by **Piotr Sowa** — AI researcher, GPU software engineer, and creator of [iChess.io](https://iChess.io).  
More information and technical articles available at [https://iBlog.io](https://iBlog.io).

For academic citations or collaboration inquiries, please contact via [iSowa.io](https://iSowa.io) or LinkedIn.

---

**© 2025 Piotr Sowa / iSowa.io**  — All rights reserved.

