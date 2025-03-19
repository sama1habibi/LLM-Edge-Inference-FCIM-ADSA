import torch
import time
import pickle
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set pad_token to eos_token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to load the full model
def load_full_model():
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return model

# Function to measure computation time, memory usage, and generate text
def measure_time_and_memory(model, tokenizer, prompt, output_words):
    # Start timing and memory measurement
    start_time = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    memory_before = torch.cuda.memory_allocated(device) / (1024 ** 2)

    # Estimate the number of tokens to generate
    tokens_per_word = 1.33
    max_new_tokens = int(output_words * tokens_per_word)

    # Ensure max_new_tokens does not exceed model's maximum capacity
    max_model_tokens = 2048
    prompt_length = inputs.input_ids.shape[1]
    max_allowed_new_tokens = max_model_tokens - prompt_length
    if max_new_tokens > max_allowed_new_tokens:
        max_new_tokens = max_allowed_new_tokens

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
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


# Function to process tasks with FCFS scheduling and allow transmission overlap
def process_tasks_fcfs_with_transmission_overlap(tasks):
    current_time = 0
    task_queue = sorted(tasks, key=lambda x: x['arrival_time'])  # Sort by arrival time (FCFS)
    task_intervals = {task['label']: [] for task in tasks}  # Store intervals for each task
    transmission_tracker = {}
    task_start_time = {}
    task_completion_time = {}  # To store completion times for each task
    task_stats = {}  # To store the requested task statistics
    last_transmission_end = 0

    for task in task_queue:
        # Start computation immediately after the previous task's computation finishes
        task_start_time[task['label']] = current_time

        # Clear memory and reload the model for each task
        torch.cuda.empty_cache()  # Clear cache
        model = load_full_model()  # Reload model

        # Measure time, memory usage, and generate text
        comp_time, memory_used, generated_text = measure_time_and_memory(model, tokenizer, task["prompt"], task["output_words"])

        task["computation_time"] = int(round(comp_time))  # Ensure computation time is rounded to an integer
        task["memory_used"] = memory_used
        task["generated_text"] = generated_text  # Store the generated text

        # Simulate computation
        end_computation_time = current_time + task["computation_time"]
        task_intervals[task['label']].append((current_time, end_computation_time))

        # Simulate transmission overlapping with next task's computation
        transmission_start_time = max(last_transmission_end, end_computation_time)
        transmission_end_time = transmission_start_time + task['trans_time']
        transmission_tracker[task['label']] = (transmission_start_time, transmission_end_time)
        task_completion_time[task['label']] = transmission_end_time

        # Store statistics
        task_stats[task['label']] = {
            'arrival_time': task['arrival_time'],
            'start_time': task_start_time[task['label']],
            'end_computation': end_computation_time,
            'wait_time': task_start_time[task['label']] - task['arrival_time'],
            'task_completion_time': transmission_end_time,
            'processing_time': end_computation_time - task_start_time[task['label']],
            'memory_used': memory_used,
        }

        # Allow the next task's computation to start after this task's computation
        current_time = end_computation_time  # Update current time to when the computation finishes
        last_transmission_end = transmission_end_time  # Transmission overlaps with the next task's computation

    return task_intervals, transmission_tracker, task_completion_time, task_stats

# Function to plot Gantt chart
def plot_gantt_chart(task_intervals, transmission_tracker, title):
    fig, ax = plt.subplots()
    colors = ['blue', 'green', 'red', 'orange', 'yellow']
    task_order = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5']

    for i, task_label in enumerate(task_order):
        if task_label in task_intervals and task_intervals[task_label]:  # Check if intervals exist
            intervals = task_intervals[task_label]
            for interval in intervals:
                ax.broken_barh([(interval[0], interval[1] - interval[0])], (i - 0.4, 0.8), facecolors=colors[i % len(colors)])

    last_trans_end = 0
    for i, task_label in enumerate(task_order):
        if task_label in transmission_tracker and transmission_tracker[task_label]:  # Check if transmission exists
            trans_start, trans_end = transmission_tracker[task_label]
            ax.broken_barh([(trans_start, trans_end - trans_start)], (i - 0.4, 0.8), facecolors=colors[i % len(colors)], hatch='/', edgecolor='black', linestyle='dashed')
            if trans_end > last_trans_end:
                last_trans_end = trans_end

    max_time = max(
        max((interval[1] for interval_list in task_intervals.values() for interval in interval_list), default=0),
        max((trans_end for trans_start, trans_end in transmission_tracker.values()), default=0)
    )

    ax.set_yticks(range(len(task_order)))
    ax.set_yticklabels(task_order)
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.set_xlim(0, max_time)
    ax.set_xticks(range(0, int(max_time) + 1, 5))

    # Add text showing when the last task finished
    ax.text(last_trans_end, -0.5, f'finished at {last_trans_end}', va='center', ha='left', fontsize=5, color='red', fontweight='bold')

    plt.show()

# Run the task processing with transmission overlapping computation
task_intervals, transmission_tracker, task_completion_time, task_stats = process_tasks_fcfs_with_transmission_overlap(tasks)

# Print and save memory usage and waiting times for each task
memory_usage = []
waiting_times = []
task_labels = []

# Save data in a structured format (for later comparisons and analysis)
results_data = {
    'schedule': task_intervals,
    'waiting_times': {},  # Store waiting times
    'memory_usage': {},   # Store memory usage
    'transmission_tracker': transmission_tracker,
    'task_completion_time': task_completion_time, 
    'task_stats': task_stats  # Store full task statistics
}

for task_label, stats in task_stats.items():
    trans_end_time = transmission_tracker[task_label][1]
    waiting_time = trans_end_time - stats['arrival_time']

    memory_usage.append(stats['memory_used'])
    waiting_times.append(waiting_time)
    task_labels.append(task_label)

    # Save to results_data
    results_data['waiting_times'][task_label] = waiting_time
    results_data['memory_usage'][task_label] = stats['memory_used']

    print(f"Task: {task_label}")
    print(f"  Memory Used: {stats['memory_used']} MB")
    print(f"  Waiting Time: {waiting_time}")

# Save results to pickle file
with open('task_results_FCFS.pkl', 'wb') as f:
    pickle.dump(results_data, f)


# **Print all task intervals**
for task_label, intervals in task_intervals.items():
    print(f"Task {task_label} computation intervals: {intervals}")

# Save the results for future use
with open('task_results_FCFS.pkl', 'wb') as f:
    pickle.dump({
        'schedule': task_intervals,
        'task_intervals': task_intervals,
        'transmission_tracker': transmission_tracker,
        'task_completion_time': task_completion_time,  # Save completion times
        'task_stats': task_stats,  # Save the task statistics
    }, f)

# Print the statistics for each task
for task_label, stats in task_stats.items():
    print(f"Task: {task_label}")
    print(f"  Arrival Time: {stats['arrival_time']}")
    print(f"  Start Time: {stats['start_time']}")
    print(f"  End of Computation: {stats['end_computation']}")
    print(f"  Task Completion Time: {stats['task_completion_time']}")
    print(f"  Processing Time: {stats['processing_time']}")
    print("-" * 40)

# Plot the Gantt chart for the task scheduling
plot_gantt_chart(task_intervals, transmission_tracker, "Task Scheduling Gantt Chart (FCFS with Transmission Overlap)")

# Print the generated texts for each task
for task in tasks:
    print(f"Generated text for {task['label']}:")
    print(task['generated_text'])
    print("=" * 80)
