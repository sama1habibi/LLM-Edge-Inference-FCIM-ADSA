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

# Function to compute completion times for each task in the queue
def compute_completion_times(queue, current_time):
    completion_times = []
    cumulative_computation = 0
    for task in queue:
        cumulative_computation += task['computation_time']
        completion_time = cumulative_computation + task['trans_time']
        completion_times.append(completion_time)
    return completion_times

# Function to compute deltas for each task
def compute_deltas(queue, completion_times, current_time):
    deltas = []
    for task, completion_time in zip(queue, completion_times):
        updated_deadline = task['deadline'] - current_time
        delta = updated_deadline - completion_time
        deltas.append(delta)
    return deltas

# Function to reorder tasks based on arrival time or transmission time
def reorder_tasks(queue):
    if not queue:
        return queue

    trans_times = [task['trans_time'] for task in queue]

    if len(trans_times) > 1:
        variance = max(trans_times) - min(trans_times)
    else:
        variance = 0

    if variance < 0.1:
        queue = sorted(queue, key=lambda x: x['arrival_time'])
    else:
        queue = sorted(queue, key=lambda x: x['trans_time'], reverse=True)

    return queue

# Main function to process tasks
def process_tasks(tasks):
    current_time = 0
    task_queue = []
    task_intervals = {task['label']: [] for task in tasks}
    transmission_tracker = {}
    task_start_time = {}
    task_completion_time = {}
    task_stats = {}
    in_process_task = None
    task_interruption_tracker = {}
    small_value_threshold = 1e-10
    upcoming_times = [task['arrival_time'] for task in tasks]
    processed_tasks = []

    while tasks or task_queue or in_process_task:
        if upcoming_times:
            next_event_time = min(upcoming_times)
            upcoming_times.remove(next_event_time)
        else:
            if in_process_task:
                next_event_time = current_time + in_process_task['computation_time']
            else:
                break

        current_time = max(current_time, next_event_time)

        # Step 1: Add newly arrived tasks to the queue
        newly_arrived_tasks = [task for task in tasks if task['arrival_time'] <= current_time]
        for task in newly_arrived_tasks:
            tasks.remove(task)

            torch.cuda.empty_cache()
            model = load_full_model()

            comp_time, memory_used, generated_text = measure_time_and_memory(model, tokenizer, task["prompt"], task["output_words"])

            task["computation_time"] = int(round(comp_time))
            task["memory_used"] = memory_used
            task["generated_text"] = generated_text
            task_queue.append(task)
            processed_tasks.append(task)

            task_stats[task['label']] = {
                'arrival_time': task['arrival_time'],
                'start_time': None,
                'end_computation': None,
                'wait_time': None,
                'task_completion_time': None,
                'processing_time': None,
                'memory_used': memory_used,
            }

        if in_process_task:
            elapsed_time = current_time - task_start_time[in_process_task['label']]
            remaining_time = in_process_task['computation_time'] - elapsed_time
            if remaining_time > small_value_threshold:
                task_interruption_tracker[in_process_task['label']].append((task_start_time[in_process_task['label']], current_time))
                task_intervals[in_process_task['label']].append((task_start_time[in_process_task['label']], current_time))
                in_process_task['computation_time'] = remaining_time

                torch.cuda.empty_cache()
                model = load_full_model()
                memory_used_reload = torch.cuda.memory_allocated(device) / (1024 ** 2)
                in_process_task['memory_used'] += memory_used_reload

                task_queue.append(in_process_task)
            else:
                transmission_start_time = current_time
                transmission_end_time = transmission_start_time + in_process_task['trans_time']
                transmission_tracker[in_process_task['label']] = (transmission_start_time, transmission_end_time)
                task_completion_time[in_process_task['label']] = transmission_end_time
                task_interruption_tracker[in_process_task['label']].append((task_start_time[in_process_task['label']], current_time))
                task_intervals[in_process_task['label']].append((task_start_time[in_process_task['label']], current_time))

                task_stats[in_process_task['label']]['end_computation'] = current_time
                task_stats[in_process_task['label']]['task_completion_time'] = transmission_end_time
                task_stats[in_process_task['label']]['processing_time'] = current_time - task_stats[in_process_task['label']]['start_time']

            in_process_task = None

        task_queue = reorder_tasks(task_queue)

        if not task_queue:
            continue

        completion_times = compute_completion_times(task_queue, current_time)
        deltas = compute_deltas(task_queue, completion_times, current_time)

        if any(delta < 0 for delta in deltas):
            critical_task = min(zip(task_queue, deltas), key=lambda x: x[1])[0]
        else:
            critical_task = task_queue[0]

        task_queue.remove(critical_task)
        task_start_time[critical_task['label']] = current_time

        if critical_task['label'] not in task_interruption_tracker:
            task_interruption_tracker[critical_task['label']] = []
            task_intervals[critical_task['label']] = []

        in_process_task = critical_task

        if task_stats[critical_task['label']]['start_time'] is None:
            task_stats[critical_task['label']]['start_time'] = current_time
            task_stats[critical_task['label']]['wait_time'] = current_time - task_stats[critical_task['label']]['arrival_time']

        next_completion_time = current_time + in_process_task['computation_time']
        upcoming_times.append(next_completion_time)

    return task_interruption_tracker, transmission_tracker, task_completion_time, task_intervals, processed_tasks, task_stats

# Function to plot Gantt chart
def plot_gantt_chart(task_intervals, transmission_tracker, title):
    fig, ax = plt.subplots()
    colors = ['blue', 'green', 'red', 'orange', 'yellow']
    task_order = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5']

    for i, task_label in enumerate(task_order):
        if task_label in task_intervals and task_intervals[task_label]:
            intervals = task_intervals[task_label]
            for interval in intervals:
                ax.broken_barh([(interval[0], interval[1] - interval[0])], (i - 0.4, 0.8), facecolors=colors[i % len(colors)])

    last_trans_end = 0
    for i, task_label in enumerate(task_order):
        if task_label in transmission_tracker:
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
    ax.text(last_trans_end, -0.5, f'finished at {last_trans_end}', va='center', ha='left', fontsize=5, color='red', fontweight='bold')
    plt.show()

# Run the task processing
task_interruption_tracker, transmission_tracker, task_completion_time, task_intervals, updated_tasks, task_stats = process_tasks(tasks)

# Print and save memory usage and waiting times for each task
memory_usage = {}
waiting_times = {}
task_labels = []

# Save data for later comparison
results_data = {
    'waiting_times': waiting_times,
    'memory_usage': memory_usage,
    'task_intervals': task_intervals,
    'transmission_tracker': transmission_tracker,
    'task_completion_time': task_completion_time,
    'task_stats': task_stats
}

for task_label, stats in task_stats.items():
    if task_label in transmission_tracker:
        trans_end_time = transmission_tracker[task_label][1]
        waiting_time = trans_end_time - stats['arrival_time']
    else:
        waiting_time = None

    memory_usage[task_label] = stats['memory_used']
    waiting_times[task_label] = waiting_time  # Correctly assign waiting time to the dictionary
    task_labels.append(task_label)

# Debugging: Print waiting_times and memory_usage before saving
#print("Waiting Times:", waiting_times)
#print("Memory Usage:", memory_usage)

# Save results to a pickle file for comparison
with open('task_results_ADSA.pkl', 'wb') as f:
    pickle.dump(results_data, f)


# Print all task intervals
for task_label, intervals in task_intervals.items():
    print(f"Task {task_label} computation intervals: {intervals}")

# Save the task intervals and stats for comparison
with open('task_results_ADSA.pkl', 'wb') as f:
    pickle.dump({
        'schedule': updated_tasks,
        'task_intervals': task_intervals,
        'transmission_tracker': transmission_tracker,
        'task_completion_time': task_completion_time,
        'task_stats': task_stats,
        'waiting_times': waiting_times,  # Now saving waiting times
        'memory_usage': memory_usage     # Now saving memory usage
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

# Plot the Gantt chart for task scheduling
plot_gantt_chart(task_intervals, transmission_tracker, "Task Scheduling Gantt Chart")

# Print generated texts
for task in updated_tasks:
    print(f"Generated text for {task['label']}:")
    print(task['generated_text'])
    print("=" * 80)
