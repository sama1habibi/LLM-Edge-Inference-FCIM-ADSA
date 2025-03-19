import torch
import time
import pickle
import matplotlib.pyplot as plt
import heapq
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to load the full model
def load_full_model():
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return model

# Function to measure computation time, memory usage, and generate text
def measure_time_and_memory(model, tokenizer, prompt, output_words):
    start_time = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    memory_before = torch.cuda.memory_allocated(device) / (1024 ** 2)

    tokens_per_word = 1.33
    max_new_tokens = int(output_words * tokens_per_word)
    max_model_tokens = 2048
    prompt_length = inputs.input_ids.shape[1]
    max_allowed_new_tokens = max_model_tokens - prompt_length
    if max_new_tokens > max_allowed_new_tokens:
        max_new_tokens = max_allowed_new_tokens

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    end_time = time.perf_counter()
    torch.cuda.synchronize()

    memory_after = torch.cuda.memory_allocated(device) / (1024 ** 2)
    computation_time = end_time - start_time
    memory_used = memory_after - memory_before

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return computation_time, memory_used, generated_text

# Define tasks
tasks = [
    {"label": "Task 1", "prompt": "Analyze the implications of quantum computing on cybersecurity.", "arrival_time": "arrival_time1", "output_words": "Word_lenght1", "trans_time": "trans_time1", "deadline": "deadline1"},
    {"label": "Task 2", "prompt": "Discuss the key milestones in the history of artificial intelligence.", "arrival_time": "arrival_time2", "output_words": "Word_lenght2", "trans_time": "trans_time2", "deadline": "deadline2"},
    {"label": "Task 3", "prompt": "Summarize the effects of climate change on polar ecosystems.", "arrival_time": "arrival_time3", "output_words": "Word_lenght3", "trans_time": "trans_time3", "deadline": "deadline3"},
    {"label": "Task 4", "prompt": "Analyze the economic impacts of the COVID-19 pandemic.", "arrival_time": "arrival_time4", "output_words": "Word_lenght4", "trans_time": "trans_time4", "deadline": "deadline4"},
    {"label": "Task 5", "prompt": "Explain the process of photosynthesis in detail.", "arrival_time": "arrival_time5", "output_words": "Word_lenght5", "trans_time": "trans_time5", "deadline": "deadline5"},
]


# Preemptive SRTF Scheduling Simulation
def process_tasks_srtf(tasks):
    current_time = 0
    # Dictionaries to store intervals and completion info
    task_intervals = {task['label']: [] for task in tasks}
    transmission_tracker = {}
    task_completion_time = {}
    last_transmission_end = 0
    remaining_execution = {}
    
    # Load the model once and pre-compute each task's execution time & memory usage
    model = load_full_model()
    for task in tasks:
        comp_time, memory_used, generated_text = measure_time_and_memory(model, tokenizer, task["prompt"], task["output_words"])
        # Round the computed time to at least 1 time unit
        task["execution_time"] = max(1, int(round(comp_time)))
        task["memory_used"] = memory_used
        task["generated_text"] = generated_text
        remaining_execution[task['label']] = task["execution_time"]
    
    # Build an arrival queue (min-heap) based on arrival time
    task_queue = []
    for i, task in enumerate(tasks):
        heapq.heappush(task_queue, (task["arrival_time"], i, task))
    
    # Processing queue: min-heap keyed by remaining execution time
    processing_queue = []  # (remaining_time, arrival_time, task)
    
    # Simulation loop runs until all tasks are processed
    while task_queue or processing_queue:
        # Add tasks that have arrived by current_time
        while task_queue and task_queue[0][0] <= current_time:
            arrival, _, new_task = heapq.heappop(task_queue)
            heapq.heappush(processing_queue, (remaining_execution[new_task['label']], arrival, new_task))
        
        # If no task is available, jump to next arrival time
        if not processing_queue:
            if task_queue:
                current_time = task_queue[0][0]
                continue
            else:
                break
        
        # Peek at the task with the shortest remaining time
        current_remaining, task_arrival, running_task = processing_queue[0]
        next_arrival_time = task_queue[0][0] if task_queue else float('inf')
        time_until_next_arrival = next_arrival_time - current_time
        
        # Determine time slice: the lesser of remaining time and time until next arrival
        time_slice = min(current_remaining, time_until_next_arrival)
        
        # Pop the running task to update its remaining execution
        heapq.heappop(processing_queue)
        start_interval = current_time
        current_time += time_slice  # Advance simulation time by time_slice
        
        # Record this execution interval
        task_intervals[running_task['label']].append((start_interval, current_time))
        
        # Update remaining execution time for the running task
        remaining_execution[running_task['label']] -= time_slice
        
        # Check if the task is finished
        if remaining_execution[running_task['label']] > 0:
            # Not finished: reinsert the task with its updated remaining time
            heapq.heappush(processing_queue, (remaining_execution[running_task['label']], task_arrival, running_task))
        else:
            # Task has completed its computation; schedule its transmission
            finish_time = current_time
            transmission_start = max(last_transmission_end, finish_time)
            transmission_end = transmission_start + running_task['trans_time']
            transmission_tracker[running_task['label']] = (transmission_start, transmission_end)
            task_completion_time[running_task['label']] = transmission_end
            last_transmission_end = transmission_end

    return task_intervals, transmission_tracker, task_completion_time

# Run SRTF scheduling
task_intervals, transmission_tracker, task_completion_time = process_tasks_srtf(tasks)

# Save results
with open("task_results_SRTF.pkl", 'wb') as f:
    pickle.dump({
        'schedule': task_intervals,
        'transmission_tracker': transmission_tracker,
        'task_completion_time': task_completion_time,
    }, f)

# Print intervals for verification
print("\nFinal Computation Intervals:", task_intervals)
print("Final Transmission Intervals:", transmission_tracker)

# Function to plot computation and transmission intervals in a Gantt chart
def plot_gantt_chart(task_intervals, transmission_tracker, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['blue', 'green', 'red', 'orange', 'yellow']
    task_order = list(task_intervals.keys())

    for i, task_label in enumerate(task_order):
        # Plot each computation interval for the task
        for interval in task_intervals[task_label]:
            ax.broken_barh([(interval[0], interval[1] - interval[0])],
                           (i - 0.4, 0.8), facecolors=colors[i % len(colors)],
                           label=f'{task_label} (Computation)')
        # Plot transmission interval with a striped pattern
        if task_label in transmission_tracker:
            trans_start, trans_end = transmission_tracker[task_label]
            ax.broken_barh([(trans_start, trans_end - trans_start)],
                           (i - 0.4, 0.8), facecolors=colors[i % len(colors)],
                           hatch='/', edgecolor='black', linestyle='dashed',
                           label=f'{task_label} (Transmission)')

    ax.set_yticks(range(len(task_order)))
    ax.set_yticklabels(task_order)
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.grid(True)
    plt.legend()
    plt.show()

# Plot the preemptive SRTF Gantt chart
plot_gantt_chart(task_intervals, transmission_tracker, "Task Scheduling Gantt Chart (SRTF)")
