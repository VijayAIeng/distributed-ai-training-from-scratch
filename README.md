# Distributed AI Training From Scratch

This repository is my hands-on exploration of distributed AI training, starting from the basic problem of training a model on one GPU and gradually moving toward multi-GPU and multi-node training.

The goal is to understand what actually happens when a model is trained across multiple GPUs and machines.

Instead of treating distributed training as something that I simply enable with a framework, I want to understand the underlying concepts, communication patterns, memory requirements, synchronization, parallelism strategies, failure handling, and performance tradeoffs.

The repository will move from simple experiments to production-oriented distributed training systems.

---

# Why Distributed Training?

Modern AI models can become too large or too expensive to train on a single GPU.

A simple training setup looks like:

```text
Dataset
   |
   v
GPU
   |
   v
Model
   |
   v
Forward
   |
   v
Backward
   |
   v
Optimizer
```

But when the model, dataset, or training workload becomes large:

```text
One GPU
   |
   X
   |
   v
Not Enough Memory / Compute
```

Distributed training allows the workload to be divided across multiple GPUs or machines.

```text
                    Training Job
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
        GPU 0          GPU 1          GPU 2
          |              |              |
          +--------------+--------------+
                         |
                         v
                    Synchronization
```

The main goal is not simply to use more GPUs.

The goal is to understand:

```text
How do GPUs communicate?

What does each GPU store?

How are gradients synchronized?

How is data divided?

How is the model divided?

How does memory change?

How does performance scale?

What happens when one worker fails?
```

---

# Single GPU vs Distributed Training

## Single GPU

```text
                    GPU

Dataset
   |
   v
Model
   |
   v
Forward
   |
   v
Loss
   |
   v
Backward
   |
   v
Gradients
   |
   v
Optimizer
   |
   v
Updated Model
```

Everything happens inside one device.

---

# Data Parallel Training

With data parallelism, each GPU has a copy of the model.

```text
                       Model
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
       GPU 0          GPU 1          GPU 2
       Model          Model          Model
          |              |              |
       Batch A         Batch B        Batch C
          |              |              |
          v              v              v
      Gradients       Gradients      Gradients
          |              |              |
          +--------------+--------------+
                         |
                    Synchronize
                         |
                         v
                   Model Update
```

Each GPU processes different data.

The gradients are synchronized so that the replicas remain consistent.

---

# The Core Distributed Training Problem

Suppose the global batch contains:

```text
Batch = 12 samples
```

With three GPUs:

```text
GPU 0 → 4 samples
GPU 1 → 4 samples
GPU 2 → 4 samples
```

Each worker performs:

```text
Forward
Backward
```

Then gradients must be combined.

Conceptually:

```text
GPU 0 → Gradients G0
GPU 1 → Gradients G1
GPU 2 → Gradients G2

            |
            v

        All-Reduce

            |
            v

       Average Gradient

            |
            v

      Optimizer Update
```

This communication step is one of the most important concepts in distributed training.

---

# What Happens During Distributed Training?

A simplified training step:

```text
                 Global Batch
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
        GPU 0       GPU 1       GPU 2
          |           |           |
          v           v           v
       Forward     Forward     Forward
          |           |           |
          v           v           v
       Loss         Loss        Loss
          |           |           |
          v           v           v
      Backward     Backward    Backward
          |           |           |
          v           v           v
     Gradients    Gradients   Gradients
          |           |           |
          +-----------+-----------+
                      |
                      v
                  All-Reduce
                      |
                      v
              Synchronized Gradients
                      |
                      v
                  Optimizer
                      |
                      v
                 Updated Model
```

I want to implement and inspect this process directly.

---

# Distributed Training Levels

The repository will progress through several levels.

```text
Single GPU
    |
    v
Multi-GPU
    |
    v
Data Parallelism
    |
    v
Distributed Data Parallel
    |
    v
Mixed Precision
    |
    v
Gradient Accumulation
    |
    v
FSDP
    |
    v
Tensor Parallelism
    |
    v
Pipeline Parallelism
    |
    v
3D Parallelism
    |
    v
Multi-Node Training
    |
    v
Large-Scale Training
```

---

# 1. Single GPU Fundamentals

Before distributed training, I will establish the basic training loop.

```text
Dataset
   |
   v
DataLoader
   |
   v
Model
   |
   v
Forward
   |
   v
Loss
   |
   v
Backward
   |
   v
Optimizer
   |
   v
Checkpoint
```

I will understand:

* Batch size
* Sequence length
* Model parameters
* Activations
* Gradients
* Optimizer states
* GPU memory
* Training throughput

---

# 2. Multi-GPU Basics

The first distributed experiment will use multiple GPUs.

```text
              Training Job
                   |
        +----------+----------+
        |          |          |
        v          v          v
      Rank 0     Rank 1     Rank 2
        |          |          |
        v          v          v
      GPU 0      GPU 1      GPU 2
```

I will understand:

```text
Rank
World Size
Local Rank
Global Rank
Process
Worker
Device
```

---

# 3. Distributed Process Groups

Distributed training requires workers to communicate.

```text
Process 0
    |
    |
Process Group
    |
    +---- Process 1
    |
    +---- Process 2
    |
    +---- Process 3
```

I will explore how distributed processes discover each other and communicate.

---

# 4. Distributed Data Parallel

Distributed Data Parallel is one of the most important foundations.

```text
                  DDP

             Model Replica
                   |
       +-----------+-----------+
       |           |           |
       v           v           v
     Rank 0      Rank 1      Rank 2
       |           |           |
       v           v           v
    Batch A      Batch B      Batch C
       |           |           |
       v           v           v
   Backward     Backward     Backward
       |           |           |
       +-----------+-----------+
                   |
               All-Reduce
                   |
                   v
             Same Gradients
```

I will build DDP examples from a simple model before moving to large models.

---

# 5. All-Reduce

All-Reduce is a core collective communication operation.

Conceptually:

```text
GPU 0 → G0
GPU 1 → G1
GPU 2 → G2

       ↓

All-Reduce

       ↓

(G0 + G1 + G2) / 3

       ↓

Every GPU receives result
```

I will understand:

* Reduce
* Broadcast
* All-Reduce
* All-Gather
* Reduce-Scatter
* Gather
* Scatter
* Barrier

---

# 6. Collective Communication

The main communication operations will be explored individually.

```text
Broadcast

GPU 0
  |
  +----> GPU 1
  |
  +----> GPU 2
```

```text
All-Reduce

GPU 0 ----+
GPU 1 ----+----> Combined Result
GPU 2 ----+
```

```text
All-Gather

GPU 0 → A
GPU 1 → B
GPU 2 → C

Result on every GPU:

A + B + C
```

```text
Reduce-Scatter

A + B + C
    |
    v
Different portions
to different GPUs
```

Understanding these operations is critical for understanding advanced parallelism.

---

# 7. Ring All-Reduce

I will study the ring communication pattern.

```text
GPU 0 → GPU 1
  ↑       |
  |       v
GPU 3 ← GPU 2
```

The goal is to understand how distributed systems move and aggregate tensors efficiently.

I will explore:

```text
Bandwidth
Latency
Number of GPUs
Tensor Size
Communication Time
```

---

# 8. Communication vs Computation

Distributed training introduces a fundamental tradeoff.

```text
GPU Computation
      +
GPU Communication
      |
      v
Training Step
```

If communication becomes too expensive:

```text
More GPUs
    |
    v
More Communication
    |
    v
Poor Scaling
```

Therefore:

```text
More GPUs
    !=
Always Faster Training
```

I will measure this experimentally.

---

# 9. Strong Scaling

Strong scaling asks:

```text
How much faster can the same workload become
when more GPUs are added?
```

Example:

```text
1 GPU  → 100 minutes
2 GPUs → 55 minutes
4 GPUs → 32 minutes
8 GPUs → 22 minutes
```

The speedup will not necessarily be perfectly linear.

---

# 10. Weak Scaling

Weak scaling increases the workload as GPUs increase.

```text
1 GPU  → 1x workload
2 GPUs → 2x workload
4 GPUs → 4x workload
8 GPUs → 8x workload
```

The objective is to understand whether training time remains approximately stable as the system grows.

---

# 11. Scaling Efficiency

I will calculate:

```text
Speedup = T1 / TN
```

and:

```text
Efficiency = Speedup / N
```

where:

```text
T1 = Single GPU training time
TN = N-GPU training time
N  = Number of GPUs
```

This gives a quantitative view of distributed training efficiency.

---

# 12. Batch Size

Distributed training changes the relationship between local and global batch size.

```text
Global Batch Size
       =
Local Batch Size
       ×
Number of Workers
```

For example:

```text
Local Batch = 8
GPUs = 4

Global Batch = 32
```

I will explore how batch size affects:

* Memory
* Throughput
* Convergence
* Learning rate
* Training stability

---

# 13. Gradient Accumulation

When GPU memory is limited:

```text
Small Micro Batch
       |
       v
Forward
       |
       v
Backward
       |
       v
Accumulate
       |
       v
Repeat
       |
       v
Optimizer Step
```

I will compare:

```text
Large Batch
```

against:

```text
Micro Batch + Gradient Accumulation
```

---

# 14. Mixed Precision

Distributed training is commonly combined with reduced precision.

```text
FP32
 |
 v
BF16 / FP16
 |
 v
Lower Memory
 |
 v
Higher Throughput
```

I will explore:

* FP32
* FP16
* BF16
* Automatic Mixed Precision
* Gradient Scaling
* Numerical Stability

---

# 15. Distributed Checkpointing

Checkpointing becomes more complicated when multiple processes are involved.

```text
Training
   |
   v
Multiple GPUs
   |
   v
Model State
Optimizer State
Scheduler State
RNG State
   |
   v
Checkpoint
```

I will explore:

```text
Full Checkpoint
Sharded Checkpoint
Distributed Checkpoint
Resume Training
Fault Recovery
```

---

# 16. Fault Tolerance

A large distributed training job may run for days or weeks.

If one worker fails:

```text
GPU 0 → Healthy
GPU 1 → Healthy
GPU 2 → Failed
GPU 3 → Healthy
```

the training job can potentially fail.

I will explore:

```text
Failure Detection
Checkpoint Recovery
Worker Restart
Elastic Training
Job Recovery
```

---

# 17. Fully Sharded Data Parallel

Data Parallelism keeps a complete model replica on every GPU.

FSDP changes this.

Instead of:

```text
GPU 0 → Full Model
GPU 1 → Full Model
GPU 2 → Full Model
GPU 3 → Full Model
```

parameters can be sharded:

```text
GPU 0 → Model Shard A
GPU 1 → Model Shard B
GPU 2 → Model Shard C
GPU 3 → Model Shard D
```

This can significantly reduce memory per GPU.

---

# 18. FSDP Flow

Conceptually:

```text
Sharded Parameters
       |
       v
All-Gather
       |
       v
Full Parameters
       |
       v
Forward
       |
       v
Backward
       |
       v
Reduce-Scatter
       |
       v
Sharded Gradients
```

I will study this communication pattern in detail.

---

# 19. Data Parallelism vs FSDP

Data Parallel:

```text
GPU 0 → Full Model
GPU 1 → Full Model
GPU 2 → Full Model
GPU 3 → Full Model
```

FSDP:

```text
GPU 0 → Model Shard
GPU 1 → Model Shard
GPU 2 → Model Shard
GPU 3 → Model Shard
```

I will compare:

```text
Memory
Communication
Throughput
Scalability
Checkpointing
```

---

# 20. Tensor Parallelism

Tensor parallelism splits model computation across GPUs.

For example:

```text
Large Linear Layer
        |
        +----------------+
        |                |
        v                v
      GPU 0            GPU 1
   Matrix Shard A    Matrix Shard B
        |                |
        +-------+--------+
                |
                v
             Output
```

Instead of each GPU holding the complete operation, the computation itself is partitioned.

---

# 21. Tensor Parallelism Flow

A simplified view:

```text
Input
  |
  +--------+
  |        |
  v        v
GPU 0    GPU 1
Shard A  Shard B
  |        |
  +--------+
      |
      v
Communication
      |
      v
Output
```

I will explore:

* Column parallelism
* Row parallelism
* Communication
* Tensor partitioning
* Matrix multiplication

---

# 22. Pipeline Parallelism

Pipeline parallelism divides the model into stages.

```text
Model
  |
  +---- Stage 1 → GPU 0
  |
  +---- Stage 2 → GPU 1
  |
  +---- Stage 3 → GPU 2
  |
  +---- Stage 4 → GPU 3
```

The input moves through the stages.

```text
Microbatch
   |
   v
GPU 0
   |
   v
GPU 1
   |
   v
GPU 2
   |
   v
GPU 3
```

---

# 23. Pipeline Bubbles

Naive pipeline execution can leave GPUs idle.

```text
GPU 0  ███████
GPU 1     ███████
GPU 2        ███████
GPU 3           ███████
```

These idle regions are pipeline bubbles.

I will explore how microbatching and scheduling reduce them.

---

# 24. Pipeline Scheduling

I will explore different scheduling strategies.

Conceptually:

```text
Forward
   |
   v
Microbatches
   |
   v
Pipeline
   |
   v
Backward
```

The objective is to improve device utilization.

---

# 25. 3D Parallelism

Large-scale model training can combine multiple forms of parallelism.

```text
3D Parallelism

Data Parallelism
        +
Tensor Parallelism
        +
Pipeline Parallelism
```

Conceptually:

```text
                Training Cluster
                       |
        +--------------+--------------+
        |              |              |
        v              v              v
   Data Parallel   Data Parallel   Data Parallel
        |              |              |
     Pipeline       Pipeline       Pipeline
        |              |              |
      Tensor         Tensor         Tensor
    Parallel       Parallel       Parallel
```

I will explore why combining these approaches is useful for large models.

---

# 26. Expert Parallelism

For Mixture of Experts models:

```text
Experts
   |
   +---- GPU 0
   |
   +---- GPU 1
   |
   +---- GPU 2
   |
   +---- GPU 3
```

Tokens are routed to experts distributed across devices.

```text
Tokens
   |
   v
Router
   |
   v
Expert Assignment
   |
   v
All-to-All
   |
   v
Distributed Experts
```

This connects distributed training with the MoE concepts explored in my MoE repository.

---

# 27. Distributed LLM Training

The repository will eventually connect distributed training to Transformer and LLM training.

```text
Dataset
   |
   v
Tokenizer
   |
   v
Distributed DataLoader
   |
   v
Transformer
   |
   +--> Data Parallelism
   |
   +--> Tensor Parallelism
   |
   +--> Pipeline Parallelism
   |
   +--> FSDP
   |
   v
Loss
   |
   v
Distributed Backward
   |
   v
Optimizer
   |
   v
Checkpoint
```

---

# 28. Memory Breakdown

I will analyze GPU memory rather than treating it as one number.

A training job can consume memory for:

```text
Model Parameters
Gradients
Optimizer States
Activations
Temporary Buffers
Communication Buffers
CUDA Runtime
```

Conceptually:

```text
GPU Memory
   |
   +--> Parameters
   +--> Gradients
   +--> Optimizer
   +--> Activations
   +--> Temporary Memory
   +--> Communication
```

---

# 29. Memory Optimization

I will explore:

```text
Mixed Precision
Gradient Accumulation
Gradient Checkpointing
Activation Checkpointing
Parameter Sharding
Optimizer Sharding
Offloading
```

The objective is to understand exactly which memory component each technique reduces.

---

# 30. Activation Checkpointing

Instead of storing every activation:

```text
Forward
   |
   +--> Store Everything
   |
   v
Backward
```

activation checkpointing stores selected activations and recomputes others.

```text
Forward
   |
   +--> Store Selected Activations
   |
   v
Backward
   |
   v
Recompute Missing Activations
```

This trades computation for memory.

---

# 31. CPU Offloading

Some state can potentially be moved from GPU memory to CPU memory.

```text
GPU
 |
 | Parameters / States
 |
 v
CPU Memory
```

This can reduce GPU memory usage but introduces data-transfer overhead.

I will measure this tradeoff.

---

# 32. Communication Topology

Distributed training performance depends on hardware topology.

I will explore:

```text
GPU
GPU Interconnect
PCIe
NVLink
Network
NIC
```

Conceptually:

```text
GPU 0
  |
NVLink / PCIe
  |
GPU 1
  |
Network
  |
GPU 2
```

The communication path can strongly affect scaling.

---

# 33. Intra-Node vs Inter-Node

Inside one machine:

```text
Node 0
+-----------------------+
| GPU 0 GPU 1 GPU 2 GPU 3 |
+-----------------------+
```

Across machines:

```text
Node 0
   |
Network
   |
Node 1
   |
Network
   |
Node 2
```

Inter-node communication is often significantly more challenging.

---

# 34. Multi-Node Training

The repository will progress from:

```text
1 GPU
```

to:

```text
1 Node
4 GPUs
```

and eventually:

```text
Node 0 → GPUs
Node 1 → GPUs
Node 2 → GPUs
Node 3 → GPUs
```

with distributed communication across the network.

---

# 35. Distributed Data Loading

The dataset must also be distributed correctly.

```text
Dataset
   |
   v
Distributed Sampler
   |
   +---- GPU 0 → Data Partition 0
   |
   +---- GPU 1 → Data Partition 1
   |
   +---- GPU 2 → Data Partition 2
```

I will ensure that workers do not unnecessarily process the same samples.

---

# 36. Randomness and Reproducibility

Distributed training introduces additional randomness.

I will explore:

```text
Python Seed
NumPy Seed
PyTorch Seed
CUDA Seed
Worker Seed
Distributed Rank
```

and how these affect reproducibility.

---

# 37. Distributed Debugging

Debugging multiple processes is different from debugging a single program.

I will investigate:

```text
Deadlocks
Hangs
Timeouts
NCCL Errors
CUDA Errors
Rank Failures
Communication Failures
Out-of-Memory Errors
```

The goal is to learn how to identify which process and operation caused the problem.

---

# 38. Deadlocks

A distributed system can become stuck if workers wait for each other incorrectly.

```text
GPU 0
  |
  v
Waiting for GPU 1

GPU 1
  |
  v
Waiting for GPU 0
```

Neither can continue.

I will create controlled experiments to understand distributed deadlocks and synchronization problems.

---

# 39. NCCL

For GPU communication, I will explore NCCL concepts and how frameworks use collective communication.

Important areas include:

```text
Process Groups
Collectives
All-Reduce
All-Gather
Reduce-Scatter
Topology
Communication Performance
```

The goal is to understand what happens underneath high-level distributed APIs.

---

# 40. Distributed Training Performance

I will measure:

```text
Training Time
Step Time
Samples/sec
Tokens/sec
GPU Utilization
GPU Memory
Communication Time
Scaling Efficiency
```

For LLM training:

```text
Tokens / Second / GPU
Tokens / Second / Cluster
```

will be important metrics.

---

# 41. Throughput

For a training system:

```text
Throughput =
Processed Samples / Time
```

For language models:

```text
Throughput =
Processed Tokens / Time
```

I will compare throughput across:

```text
1 GPU
2 GPUs
4 GPUs
8 GPUs
```

and investigate why scaling may become inefficient.

---

# 42. Communication Profiling

I will separate:

```text
Computation Time
+
Communication Time
+
Data Loading Time
+
Synchronization Time
```

Example:

```text
Training Step

Forward       → 40 ms
Backward      → 60 ms
Communication → 25 ms
Optimizer     → 10 ms

Total         → 135 ms
```

This makes performance bottlenecks measurable.

---

# 43. Training Efficiency

The objective is not only:

```text
Training Faster
```

but:

```text
Training Faster
       +
Using Hardware Efficiently
       +
Controlling Cost
       +
Maintaining Model Quality
```

---

# 44. Checkpoint and Resume

A production training job should survive interruptions.

```text
Training
   |
   v
Checkpoint
   |
   v
Failure
   |
   v
Restart
   |
   v
Load Checkpoint
   |
   v
Continue Training
```

I will verify that training can continue without losing important state.

---

# 45. Production Training Workflow

The final training system will follow a flow similar to:

```text
Dataset
   |
   v
Data Validation
   |
   v
Distributed Data Loading
   |
   v
Model Initialization
   |
   v
Distributed Setup
   |
   v
Training
   |
   +---- Metrics
   +---- Logs
   +---- Profiling
   |
   v
Checkpoint
   |
   v
Evaluation
   |
   v
Resume / Scale / Deploy
```

---

# Complete Distributed Training Flow

```text
                         TRAINING CLUSTER

                            Dataset
                               |
                               v
                       Distributed Loader
                               |
                               v
                         Global Batch
                               |
             +-----------------+-----------------+
             |                 |                 |
             v                 v                 v
          GPU 0             GPU 1             GPU 2
             |                 |                 |
             v                 v                 v
          Forward           Forward           Forward
             |                 |                 |
             v                 v                 v
           Loss              Loss              Loss
             |                 |                 |
             v                 v                 v
         Backward           Backward           Backward
             |                 |                 |
             +-----------------+-----------------+
                               |
                               v
                          Communication
                               |
                 +-------------+-------------+
                 |             |             |
                 v             v             v
              All-Reduce    All-Gather   Reduce-Scatter
                 |             |             |
                 +-------------+-------------+
                               |
                               v
                           Optimizer
                               |
                               v
                        Updated Parameters
                               |
                               v
                          Checkpoint
                               |
                               v
                           Evaluation
```

---

# Parallelism Comparison

I will study the main parallelism strategies separately before combining them.

| Strategy             | What is Split?                          | Main Goal                    |
| -------------------- | --------------------------------------- | ---------------------------- |
| Data Parallelism     | Data                                    | Increase training throughput |
| DDP                  | Data + model replicas                   | Multi-GPU training           |
| FSDP                 | Parameters, gradients, optimizer states | Reduce memory                |
| Tensor Parallelism   | Model tensors                           | Train larger layers          |
| Pipeline Parallelism | Model layers                            | Train larger models          |
| Expert Parallelism   | Experts                                 | Scale MoE models             |
| 3D Parallelism       | Data + tensors + layers                 | Large-scale training         |

---

# Distributed Training Tradeoffs

Distributed training introduces several tradeoffs.

```text
More GPUs
    |
    +--> More Compute
    |
    +--> More Memory
    |
    +--> More Communication
    |
    +--> More Complexity
```

Therefore:

```text
Performance
   =
Compute
+
Communication
+
Memory
+
Synchronization
+
Data Loading
```

---

# What I Want to Measure

Every important experiment should produce measurable results.

```text
Model Parameters
GPU Memory
Training Time
Step Time
Tokens/sec
Samples/sec
Communication Time
GPU Utilization
Scaling Efficiency
Checkpoint Size
Failure Recovery Time
```

---

# Experiments

Each experiment will follow the same process.

```text
Question
   |
   v
Hypothesis
   |
   v
Implementation
   |
   v
Run
   |
   v
Measure
   |
   v
Compare
   |
   v
Analyze
   |
   v
Conclusion
```

Example experiments:

```text
How does DDP scale from 1 to 8 GPUs?

What happens to communication time as tensor size increases?

How does batch size affect throughput?

How much memory does FSDP save?

When does FSDP become communication-bound?

How does tensor parallelism affect latency?

How much pipeline bubble exists?

How does gradient checkpointing trade compute for memory?

What happens when one worker fails?

How does multi-node training compare with single-node training?

How does communication topology affect scaling?

What is the maximum useful GPU count for a workload?

```

---

# Repository Structure

```text
distributed-ai-training-from-scratch/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
│
├── 01_single_gpu/
│   ├── training_loop/
│   ├── memory/
│   ├── profiling/
│   └── baseline/
│
├── 02_distributed_fundamentals/
│   ├── processes/
│   ├── ranks/
│   ├── world_size/
│   ├── process_groups/
│   └── synchronization/
│
├── 03_collective_communication/
│   ├── broadcast/
│   ├── reduce/
│   ├── all_reduce/
│   ├── all_gather/
│   ├── reduce_scatter/
│   └── barriers/
│
├── 04_ddp/
│   ├── basic_ddp/
│   ├── distributed_sampler/
│   ├── gradient_sync/
│   ├── checkpointing/
│   └── experiments/
│
├── 05_training_optimization/
│   ├── gradient_accumulation/
│   ├── mixed_precision/
│   ├── bf16/
│   ├── fp16/
│   └── gradient_scaling/
│
├── 06_memory_optimization/
│   ├── activation_checkpointing/
│   ├── gradient_checkpointing/
│   ├── optimizer_states/
│   ├── cpu_offloading/
│   └── memory_analysis/
│
├── 07_fsdp/
│   ├── parameter_sharding/
│   ├── gradient_sharding/
│   ├── optimizer_sharding/
│   ├── communication/
│   └── checkpoints/
│
├── 08_tensor_parallelism/
│   ├── column_parallel/
│   ├── row_parallel/
│   ├── linear_layers/
│   ├── attention/
│   └── communication/
│
├── 09_pipeline_parallelism/
│   ├── model_partitioning/
│   ├── microbatches/
│   ├── scheduling/
│   ├── pipeline_bubbles/
│   └── experiments/
│
├── 10_expert_parallelism/
│   ├── moe_distribution/
│   ├── token_dispatch/
│   ├── all_to_all/
│   └── expert_placement/
│
├── 11_3d_parallelism/
│   ├── data_parallel/
│   ├── tensor_parallel/
│   ├── pipeline_parallel/
│   └── combined/
│
├── 12_multi_node/
│   ├── networking/
│   ├── rendezvous/
│   ├── node_configuration/
│   ├── communication/
│   └── experiments/
│
├── 13_fault_tolerance/
│   ├── failure_detection/
│   ├── checkpoint_recovery/
│   ├── worker_restart/
│   └── elastic_training/
│
├── 14_profiling/
│   ├── gpu/
│   ├── communication/
│   ├── memory/
│   ├── pytorch_profiler/
│   └── bottlenecks/
│
├── 15_scaling/
│   ├── strong_scaling/
│   ├── weak_scaling/
│   ├── throughput/
│   ├── efficiency/
│   └── benchmarks/
│
├── 16_llm_training/
│   ├── transformer/
│   ├── tokenizer/
│   ├── pretraining/
│   ├── distributed_dataloader/
│   └── distributed_checkpoints/
│
├── src/
│   └── distributed_training/
│       ├── data/
│       ├── communication/
│       ├── parallelism/
│       ├── memory/
│       ├── training/
│       ├── checkpointing/
│       ├── profiling/
│       └── evaluation/
│
├── configs/
├── scripts/
├── notebooks/
├── benchmarks/
├── dashboards/
├── tests/
├── experiments/
└── artifacts/
```

---

# Technology

The main technologies explored in this repository include:

### Programming

```text
Python
PyTorch
NumPy
```

### Distributed Training

```text
PyTorch Distributed
DDP
FSDP
DistributedDataParallel
DistributedSampler
Process Groups
Collective Communication
```

### Parallelism

```text
Data Parallelism
Tensor Parallelism
Pipeline Parallelism
Expert Parallelism
3D Parallelism
```

### Performance

```text
Mixed Precision
BF16
FP16
Gradient Checkpointing
Activation Checkpointing
GPU Profiling
Communication Profiling
```

### Infrastructure

```text
CUDA
NCCL
Multi-GPU
Multi-Node Training
Containerized Training
Cluster-Based Training
```

---

# From Scratch to Production

The complete learning path is:

```text
Single GPU Training
        |
        v
GPU Memory Understanding
        |
        v
Multi-Process Training
        |
        v
Distributed Communication
        |
        v
All-Reduce
        |
        v
DDP
        |
        v
Mixed Precision
        |
        v
Gradient Accumulation
        |
        v
Memory Optimization
        |
        v
FSDP
        |
        v
Tensor Parallelism
        |
        v
Pipeline Parallelism
        |
        v
Expert Parallelism
        |
        v
3D Parallelism
        |
        v
Multi-Node Training
        |
        v
Fault Tolerance
        |
        v
Profiling
        |
        v
Scaling
        |
        v
LLM Training
        |
        v
Production Training System
```

---

# Final Goal

The final goal is to understand distributed AI training at three levels.

## Level 1: Training

```text
Dataset
   |
   v
Model
   |
   v
Forward
   |
   v
Backward
   |
   v
Optimizer
```

## Level 2: Distributed Systems

```text
Multiple GPUs
      |
      v
Communication
      |
      v
Synchronization
      |
      v
Parallelism
      |
      v
Scaling
```

## Level 3: Production

```text
Large Dataset
      |
      v
Distributed Training
      |
      v
Multi-GPU / Multi-Node
      |
      v
Checkpointing
      |
      v
Fault Recovery
      |
      v
Monitoring
      |
      v
Performance Optimization
      |
      v
Reliable Training Infrastructure
```

The objective is to move from:

```text
"I can train a model on one GPU."
```

to:

```text
"I understand how multiple GPUs coordinate during training."
```

and finally:

```text
"I can design, implement, profile, debug, optimize,
and operate distributed AI training systems at scale."
```

This repository is my exploration of distributed AI training from the fundamentals of GPU communication to production-oriented multi-GPU and multi-node training systems.
