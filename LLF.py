import torch
import time
import pickle
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize model and tokenizer (load once for efficiency)
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_full_model():
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return model

def measure_time_and_memory(model, tokenizer, prompt, output_words):
    start_time = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
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

# Load model once
model = load_full_model()

# Pre-compute each task's execution time and memory usage
for task in tasks:
    comp_time, memory_used, generated_text = measure_time_and_memory(model, tokenizer, task["prompt"], task["output_words"])
    print(f"Generated Output for {task['label']}:")
    print(generated_text)
    print("=" * 80)
    # Use at least one time unit for execution
    task["execution_time"] = max(1, int(round(comp_time)))
    task["memory_used"] = memory_used
    task["generated_text"] = generated_text

##############################################################################
# Preemptive LLF Scheduler Simulation for the Execution Phase
##############################################################################

def process_tasks_llf_preemptive(tasks):
    current_time = 0
    # Record execution intervals for each task
    task_intervals = {task['label']: [] for task in tasks}
    # Add fields needed for simulation
    for task in tasks:
        task['remaining_time'] = task['execution_time']
        task['finish_time'] = None  # will be set when execution completes
        task['laxity'] = None
    # Sort tasks by arrival time
    tasks_sorted = sorted(tasks, key=lambda t: t['arrival_time'])
    ready_tasks = []
    completed_tasks = []
    next_task_index = 0

    current_running_task = None
    current_interval_start = None

    # Simulation loop: run until all tasks complete execution
    while len(completed_tasks) < len(tasks):
        # Add tasks that have arrived at current_time
        while next_task_index < len(tasks_sorted) and tasks_sorted[next_task_index]['arrival_time'] <= current_time:
            ready_tasks.append(tasks_sorted[next_task_index])
            next_task_index += 1

        if not ready_tasks:
            # No tasks are ready; idle time
            current_time += 1
            current_running_task = None
            current_interval_start = None
            continue

        # Update laxity for all ready tasks: laxity = deadline - (current_time + remaining_time)
        for task in ready_tasks:
            task['laxity'] = task['deadline'] - (current_time + task['remaining_time'])

        # Select the task with the smallest laxity
        chosen_task = min(ready_tasks, key=lambda t: t['laxity'])

        # Handle preemption: if a different task is chosen than the one running, record the previous interval
        if current_running_task is None:
            current_running_task = chosen_task
            current_interval_start = current_time
        elif chosen_task['label'] != current_running_task['label']:
            # Record the interval for the task that was preempted
            task_intervals[current_running_task['label']].append((current_interval_start, current_time))
            current_running_task = chosen_task
            current_interval_start = current_time

        # Execute the chosen task for one time unit
        chosen_task['remaining_time'] -= 1

        # If the chosen task finishes execution, record its finish time and final interval
        if chosen_task['remaining_time'] == 0:
            task_intervals[chosen_task['label']].append((current_interval_start, current_time + 1))
            chosen_task['finish_time'] = current_time + 1
            ready_tasks.remove(chosen_task)
            completed_tasks.append(chosen_task)
            current_running_task = None
            current_interval_start = None

        # Advance simulation time by one unit
        current_time += 1

    # Transmission Phase (non-preemptive)
    transmission_tracker = {}
    last_transmission_end = 0
    # Schedule transmissions in order of task finish times
    for task in sorted(tasks, key=lambda t: t['finish_time']):
        transmission_start = max(last_transmission_end, task['finish_time'])
        transmission_end = transmission_start + task['trans_time']
        transmission_tracker[task['label']] = (transmission_start, transmission_end)
        last_transmission_end = transmission_end

    # Compile task statistics
    task_stats = {}
    for task in tasks:
        first_run = task_intervals[task['label']][0][0] if task_intervals[task['label']] else None
        task_stats[task['label']] = {
            'arrival_time': task['arrival_time'],
            'first_execution_time': first_run,
            'finish_time': task['finish_time'],
            'transmission_interval': transmission_tracker[task['label']],
            'execution_intervals': task_intervals[task['label']],
            'memory_used': task["memory_used"],
        }

    return task_intervals, transmission_tracker, task_stats

# Run preemptive LLF scheduling simulation
exec_intervals, transmission_tracker, task_stats = process_tasks_llf_preemptive(tasks)

print("Execution Intervals per Task:", exec_intervals)
print("Transmission Intervals:", transmission_tracker)

# Save results for verification
#with open('task_results_LLF.pkl', 'wb') as f:
    #pickle.dump({
        #'execution_schedule': exec_intervals,
        #'transmission_tracker': transmission_tracker,
        #'task_stats': task_stats,
    #}, f)
    
with open('task_results_LLF.pkl', 'wb') as f:
    pickle.dump({
        'schedule': exec_intervals,
        'transmission_tracker': transmission_tracker,
        'task_stats': task_stats,
    }, f)
    

with open('task_results_LLF.pkl', 'rb') as f:
    saved_data = pickle.load(f)

print("Loaded Computation Intervals:", saved_data['schedule'])
print("Loaded Transmission Intervals:", saved_data['transmission_tracker'])

#with open('task_results_LLF.pkl', 'rb') as f:
    #saved_data = pickle.load(f)

#print("Loaded Execution Intervals:", saved_data['execution_schedule'])
#print("Loaded Transmission Intervals:", saved_data['transmission_tracker'])

##############################################################################
# Gantt Chart Plotting Function
##############################################################################
def plot_gantt_chart(execution_intervals, transmission_tracker, title):
    fig, ax = plt.subplots()
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    task_order = list(execution_intervals.keys())

    # Plot execution intervals for each task
    for i, task_label in enumerate(task_order):
        for interval in execution_intervals[task_label]:
            ax.broken_barh([(interval[0], interval[1] - interval[0])],
                           (i - 0.4, 0.8), facecolors=colors[i % len(colors)])
        # Plot transmission interval with a hatched pattern
        if task_label in transmission_tracker:
            trans_start, trans_end = transmission_tracker[task_label]
            ax.broken_barh([(trans_start, trans_end - trans_start)],
                           (i - 0.4, 0.8), facecolors=colors[i % len(colors)],
                           hatch='/', edgecolor='black', linestyle='dashed')

    ax.set_yticks(range(len(task_order)))
    ax.set_yticklabels(task_order)
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.grid(True)
    plt.show()

plot_gantt_chart(saved_data['schedule'], saved_data['transmission_tracker'],
                 "Preemptive LLF Task Scheduling Gantt Chart")