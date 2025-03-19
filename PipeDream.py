import numpy as np
import random
import torch
import time
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Constants for BLOOM-176B
BYTES_PER_WORD = 4  # Assume 4 bytes per word
BLOOM_MODEL_SIZE = 176 * 1e9  # Model size for BLOOM-176B in bytes
BLOOM_NUM_LAYERS = 70  # BLOOM-176B has 70 transformer layers

# Fixed GPU Bids
fixed_bids = [
    {"gpu_id": "GPU 1", "type": "RTX 4090", "capacity": "capacity_1", "T_comp": "time_1", "P_ijm": "cost_1"},
    {"gpu_id": "GPU 2", "type": "RTX 4090", "capacity": "capacity_2", "T_comp": "time_2", "P_ijm": "cost_2"},
    {"gpu_id": "GPU 3", "type": "RTX 4090", "capacity": "capacity_3", "T_comp": "time_3", "P_ijm": "cost_3"},
    {"gpu_id": "GPU 4", "type": "T4", "capacity": "capacity_4", "T_comp": "time_4", "P_ijm": "cost_4"},
    {"gpu_id": "GPU 5", "type": "T4", "capacity": "capacity_5", "T_comp": "time_5", "P_ijm": "cost_5"},
    {"gpu_id": "GPU 6", "type": "T4", "capacity": "capacity_6", "T_comp": "time_6", "P_ijm": "cost_6"},
]


# Number of GPUs
gpu_ids = ["GPU 1", "GPU 2", "GPU 3", "GPU 4", "GPU 5", "GPU 6"]

# Generate a Rayleigh distributed communication matrix
scale_param = 3  # Adjust scale parameter for different delay characteristics
communication_matrix = {}

for gpu in gpu_ids:
    communication_matrix[gpu] = {
        other_gpu: round(np.random.rayleigh(scale_param), 2) if gpu != other_gpu else 0
        for other_gpu in gpu_ids
    }

# Generate client-GPU communication times using Rayleigh distribution
client_times = {gpu: round(np.random.rayleigh(scale_param), 2) for gpu in gpu_ids}

# Print the generated values
print("Communication Matrix:")
for gpu, connections in communication_matrix.items():
    print(f"{gpu}: {connections}")

print("\nClient Communication Times:")
print(client_times

# Partition layers among GPUs
def partition_layers_dynamic(total_layers, fixed_bids):
    stages = []
    remaining_layers = total_layers
    total_capacity = sum(bid['capacity'] for bid in fixed_bids)

    for i, bid in enumerate(fixed_bids):
        allocatable_layers = int(total_layers * (bid['capacity'] / total_capacity))
        if i == len(fixed_bids) - 1:  # Assign all remaining layers to the last GPU
            allocatable_layers = remaining_layers
        if remaining_layers < allocatable_layers:
            allocatable_layers = remaining_layers

        stages.append({
            "gpu_id": bid['gpu_id'],
            "allocated_layers": allocatable_layers,
            "comp_time": bid['T_comp'] * (allocatable_layers / bid['capacity']),
            "comm_time": client_times[bid['gpu_id']] if len(stages) == 0 else communication_matrix[stages[-1]['gpu_id']][bid['gpu_id']],
            "reward": (bid['P_ijm'] / bid['capacity']) * allocatable_layers
        })

        remaining_layers -= allocatable_layers

    return stages

# Simulate pipeline execution with overlapping communication and computation
def simulate_pipeline_execution(stages, num_micro_batches=10):
    total_latency = 0
    stage_times = [stage['comp_time'] + stage['comm_time'] for stage in stages]

    for batch in range(num_micro_batches):
        if batch == 0:
            # First micro-batch goes through all stages sequentially
            total_latency += sum(stage_times)
        else:
            # Subsequent micro-batches overlap computation and communication
            total_latency += max(stage_times)

    return total_latency

# Main allocation routine
stages = partition_layers_dynamic(BLOOM_NUM_LAYERS, fixed_bids)
total_latency = simulate_pipeline_execution(stages)

# Compute fairness and efficiency metrics
total_allocated_layers = sum(stage['allocated_layers'] for stage in stages)
total_cost = sum(stage['reward'] for stage in stages)
total_communication_time = sum(stage['comm_time'] for stage in stages)
total_computation_time = sum(stage['comp_time'] for stage in stages)

total_allocated_rewards = total_cost
fairness_index_layers = (
    (total_allocated_layers ** 2) /
    (len(stages) * sum(stage['allocated_layers'] ** 2 for stage in stages))
    if stages else 0
)

fairness_index_rewards = (
    (sum(stage['reward'] for stage in stages) ** 2) /
    (len(stages) * sum(stage['reward'] ** 2 for stage in stages))
    if stages else 0
)

cost_efficiency = total_allocated_layers / total_cost if total_cost > 0 else 0

communication_overhead = total_communication_time / (total_communication_time + total_computation_time)

# Output results
print("Pipeline Stages and Allocation for bloom:")
for stage in stages:
    print(f"GPU: {stage['gpu_id']}, Allocated Layers: {stage['allocated_layers']}, Comp Time: {stage['comp_time']:.2f}, Comm Time: {stage['comm_time']:.2f}, Reward: {stage['reward']:.2f}")

print(f"\n Estimated Total Latency for bloom: {total_latency:.2f} seconds")

print("\nOverall Optimization Results for bloom:")
print(f"  Total Allocated Layers for bloom: {total_allocated_layers}")
print(f"  Total Allocated Rewards for bloom: {total_allocated_rewards:.2f}")
print(f"  Fairness Index for bloom(Layers): {fairness_index_layers:.4f}")
print(f"  Fairness Index for bloom(Rewards): {fairness_index_rewards:.4f}")
print(f"  Cost Efficiency for bloom: {cost_efficiency:.4f}")
print(f"  Estimated Total Communication Time for bloom: {total_communication_time:.2f}")
print(f"  Estimated Total Computation Time for bloom: {total_computation_time:.2f}")
print(f"  Estimated Communication Overhead for bloom: {communication_overhead:.4f}")

#####################################################  After determining the optimal layer allocation for the selected/ winner GPUs, run:

#########################################################

import torch
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load BLOOM-176B Model
model_name = "bigscience/bloom"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
print(f"Model loaded on {device}")

# Step 2: Retrieve GPU Allocation from Auction Results
selected_gpus = []  # This should be retrieved from the auction process

# Populate selected_gpus list from previous auction results
print("\nSelected GPUs for the task for BLOOM:")
for gpu in selected_gpus:
    print(f"GPU: {gpu['gpu_id']}, Type: {gpu['type']}, Allocated Layers: {gpu['allocated_layers']}, Cost: {gpu['reward']:.2f}, Total Time Spent: {gpu['total_time_spent']:.2f}")

print(f"\nTotal Cost for BLOOM: {total_cost:.2f}")

# Assign GPUs dynamically
gpu_allocations = {
    gpu["gpu_id"]: {
        "allocated_layers": gpu["allocated_layers"],
        "device": f"cuda:{i}"  # Assigning each GPU a CUDA device dynamically
    }
    for i, gpu in enumerate(selected_gpus)
}

# Get total number of layers in the model
total_layers = len(model.transformer.h)

# Print allocations
print("\nFinal GPU Allocations:")
for gpu, details in gpu_allocations.items():
    print(f"{gpu} -> Layers: {details['allocated_layers']} | Device: {details['device']}")

# Step 3: Function to Load Model Sections Per GPU
def load_model_section(start_layer, end_layer, device):
    """ Extracts and loads only the allocated model layers dynamically on the assigned GPU. """
    model_section = torch.nn.Sequential(*list(model.transformer.h[start_layer:end_layer]))
    model_section.to(device)
    return model_section, model.lm_head, model.transformer.wte, model.transformer.ln_f, model.transformer.drop

# Step 4: Pipeline Parallelism Execution
def pipeline_parallel_inference(tasks):
    """
    Implements pipeline parallelism where:
    - The first GPU processes the original task input.
    - Each subsequent GPU processes the output from the previous GPU.
    - The final GPU produces the complete output.
    """
    for task in tasks:
        prompt = task["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        hidden_states = inputs.input_ids  # Initial input for the first GPU

        for i, (gpu_id, details) in enumerate(gpu_allocations.items()):
            device = details["device"]
            allocated_layers = details["allocated_layers"]

            # Determine start and end layers
            start_layer = sum(gpu_allocations[g]["allocated_layers"] for g in gpu_allocations if g < gpu_id)
            end_layer = min(start_layer + allocated_layers, total_layers)

            # Load GPU-specific layers
            model_section, lm_head, wte, ln_f, drop = load_model_section(start_layer, end_layer, device)

            # Move data to current GPU
            hidden_states = hidden_states.to(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Run inference on allocated layers
            with torch.no_grad():
                hidden_states = wte(hidden_states)
                hidden_states = drop(hidden_states)
                for layer in model_section:
                    hidden_states = layer(hidden_states)[0]  # Output used for next GPU
                hidden_states = ln_f(hidden_states)

            # Save the output for next GPU
            if i < len(gpu_allocations) - 1:
                next_gpu = list(gpu_allocations.keys())[i + 1]
                hidden_states = hidden_states.to(gpu_allocations[next_gpu]["device"])

        # Store final output
        task["output"] = hidden_states

    return tasks  # Return processed tasks

# Step 5: Generate Rayleigh-Distributed Communication Matrix
scale_param = 3
communication_matrix = {
    gpu1: {gpu2: round(np.random.rayleigh(scale_param), 2) if gpu1 != gpu2 else 0 for gpu2 in gpu_allocations}
    for gpu1 in gpu_allocations
}
client_times = {gpu: round(np.random.rayleigh(scale_param), 2) for gpu in gpu_allocations}

# Step 6: FCFS Scheduling with Pipeline Parallelism
def fcfs_scheduling_pipeline(tasks):
    t = 0
    schedule = []
    gpu_computation_times = {}
    gpu_communication_times = {}

    previous_gpu = None

    for task in tasks:
        start_time = t

        # Run pipeline inference
        processed_task = pipeline_parallel_inference([task])[0]  # Get processed output

        # Measure total processing time
        comp_time = sum(communication_matrix[gpu_id][next_gpu] for gpu_id, next_gpu in zip(gpu_allocations.keys(), list(gpu_allocations.keys())[1:]))
        t += comp_time
        end_time = t

        # Store task execution details
        task["start_time"] = start_time
        task["computation_time"] = comp_time
        task["completion_time"] = end_time
        task["allocated_gpus"] = list(gpu_allocations.keys())

        schedule.append(task)
        gpu_computation_times[task["allocated_gpus"][-1]] = comp_time  # Last GPU execution time

        # Compute communication times
        if previous_gpu is not None:
            gpu_communication_times[(previous_gpu, task["allocated_gpus"][-1])] = communication_matrix[previous_gpu][task["allocated_gpus"][-1]]
            t += gpu_communication_times[(previous_gpu, task["allocated_gpus"][-1])]
        previous_gpu = task["allocated_gpus"][-1]

    # Compute total processing time and communication overhead
    total_processing_time = sum(gpu_computation_times.values())
    total_communication_time = sum(gpu_communication_times.values()) + sum(client_times.values())
    communication_overhead = total_communication_time / (total_processing_time + total_communication_time)

    print(f"Measured Communication Overhead: {communication_overhead:.4f}")
    print(f"Total Processing Time:** {total_processing_time:.2f} seconds")

    return schedule, t, communication_overhead, total_processing_time

# Step 7: Generate Initial Task List (Used for First GPU)
tasks = [
    {"label": "Task 1", "prompt": "Analyze the implications of quantum computing on cybersecurity."},
    {"label": "Task 2", "prompt": "Discuss the key milestones in the history of artificial intelligence."},
    {"label": "Task 3", "prompt": "Summarize the effects of climate change on polar ecosystems."},
    {"label": "Task 4", "prompt": "Explain the process of photosynthesis in detail."},
    {"label": "Task 5", "prompt": "Analyze the economic impacts of the COVID-19 pandemic."},
]

# Step 8: Compute Metrics
fcfs_schedule, final_time, communication_overhead, total_processing_time = fcfs_scheduling_pipeline(tasks.copy())

# Print Final Task Outputs
print("Final Outputs from Last GPU in the Pipeline:")
for task in fcfs_schedule:
    print(f"Task: {task['label']}, Completion Time: {task['completion_time']:.2f}s")





