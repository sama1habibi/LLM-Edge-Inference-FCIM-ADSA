# An Incentive Mechanism for Edge Deployment of LLMs: Cost Optimization and Adaptive Scheduling

This repository contains the full implementation of the algorithms proposed in the paper:

> **An Incentive Mechanism for Edge Deployment of LLMs: Cost Optimization and Adaptive Scheduling**

---

## 📌 Overview

This project proposes two key mechanisms for efficient distributed LLM inference on edge devices:

- **FCIM**: Fair Cost-Efficient Incentive Mechanism for task and model layer allocation across heterogeneous edge devices.
- **ADSA**: Adaptive Dynamic Scheduling Algorithm for deadline-aware task execution on edge devices.

The framework is benchmarked against state-of-the-art methods to evaluate improvements in **cost**, **latency**, **fairness**, and **scheduling performance**.

---

## 🧠 Contents

| File             | Description |
|------------------|-------------|
| `FCIM.py`        | FCIM auction-based layer allocation algorithm |
| `HexGen.py`      | HexGen benchmark (asymmetric tensor partitioning) |
| `PipeDream.py`   | PipeDream static pipeline parallelism baseline |
| `PipeEdge.py`    | PipeEdge heuristic allocation algorithm |
| `ADSA.py`        | Adaptive Dynamic Scheduling Algorithm |
| `SRTF.py`        | Shortest Remaining Time First scheduling baseline |
| `FCFS.py`        | First-Come-First-Serve scheduling baseline |
| `LLF.py`         | Least Laxity First scheduling baseline |
| `optimal.py`     | Optimal scheduling with task deadline constraints |

---

## 🔧 Requirements

- Python 3.7+
- [transformers](https://huggingface.co/docs/transformers)
- [PuLP](https://coin-or.github.io/pulp/)
- NumPy
- Pandas
- Matplotlib (for plotting, optional)

Install dependencies with:

```bash
pip install numpy pandas matplotlib pulp transformers

## ▶️ How to Run

### 1. Allocation Mechanisms (Model Parallelism)
Run any of the following scripts:

```bash
python FCIM.py
python HexGen.py
python PipeDream.py
python PipeEdge.py

### 2. Scheduling Mechanisms
Run any of the following:
python ADSA.py
python SRTF.py
python FCFS.py
python LLF.py
python optimal.py
You can modify the task list inside each script to test different arrival times, deadlines, and output sizes.
## 📊 Evaluation Metrics

The framework evaluates:

- 💰 **Total Cost**
- ⏱️ **Completion Time**
- 🔁 **Communication Overhead**
- 📦 **Memory Utilization**
- 🎯 **Fairness (Layers & Rewards)**
- 📉 **Waiting Time**
- ✅ **Deadline Satisfaction**

---

## 🧪 Example Task Format

```python
tasks = [
    {"label": "Task 1", "prompt": "...", "arrival_time": 0, "output_words": 1200, "trans_time": 5, "deadline": 100},
    {"label": "Task 2", "prompt": "...", "arrival_time": 3, "output_words": 300, "trans_time": 2, "deadline": 20},
    ...
]





