import torch
import pulp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoModelForCausalLM
import tracemalloc
import pickle

# Define the tasks and transmission deadlines
transmissions = [
    {"group": 1, "length": "trans_time1", "deadline": "deadline1", "group_length": "Comp_time_ADSA1", "arrival_time": "arrival_time1"},
    {"group": 2, "length": "trans_time2", "deadline": "deadline2", "group_length": "Comp_time_ADSA2", "arrival_time": "arrival_time2"},
    {"group": 3, "length": "trans_time3", "deadline": "deadline3", "group_length": "Comp_time_ADSA3", "arrival_time": "arrival_time3"},
    {"group": 4, "length": "trans_time4", "deadline": "deadline4", "group_length": "Comp_time_ADSA4", "arrival_time": "arrival_time4"},
    {"group": 5, "length": "trans_time5", "deadline": "deadline5", "group_length": "Comp_time_ADSA5", "arrival_time": "arrival_time5"},
]



# Dynamically generate tasks based on group lengths
tasks = []
task_id_counter = 1
for transmission in transmissions:
    for _ in range(transmission['group_length']):
        tasks.append({
            "task_ids": [task_id_counter],
            "group": transmission['group'],
            "arrival_time": transmission["arrival_time"]  # Use the assigned arrival times
        })
        task_id_counter += 1

# Create the model in PuLP
model = pulp.LpProblem("Task_Scheduling_Problem", pulp.LpMinimize)

# Variables: x[t][k] = 1 if task t is scheduled at time k, 0 otherwise
x = pulp.LpVariable.dicts("x", 
                          ((t["task_ids"][0], k) for t in tasks for k in range(t["arrival_time"], transmissions[t["group"] - 1]["deadline"])), 
                          cat="Binary")

# New decision variable: task_finish_time[t] for each task
task_finish_time = pulp.LpVariable.dicts("finish_time", (t["task_ids"][0] for t in tasks), lowBound=0, cat="Continuous")

# New decision variable: transmission start time for each group
transmission_start_time = pulp.LpVariable.dicts("transmission_start_time", (trans["group"] for trans in transmissions), lowBound=0, cat="Continuous")

# T_end is the time when the last transmission is completed, to be minimized
T_end = pulp.LpVariable("T_end", lowBound=0, cat="Continuous")

# Constraints
for t in tasks:
    task_id = t["task_ids"][0]
    model += pulp.lpSum(x[task_id, k] for k in range(t["arrival_time"], transmissions[t["group"] - 1]["deadline"])) == 1

for k in range(max(transmission["deadline"] for transmission in transmissions)):
    model += pulp.lpSum(x[t["task_ids"][0], k] for t in tasks if k >= t["arrival_time"] and k < transmissions[t["group"] - 1]["deadline"]) <= 1

for t in tasks:
    task_id = t["task_ids"][0]
    model += task_finish_time[task_id] == pulp.lpSum(k * x[task_id, k] for k in range(t["arrival_time"], transmissions[t["group"] - 1]["deadline"]))

for transmission in transmissions:
    group_tasks = [task["task_ids"][0] for task in tasks if task["group"] == transmission["group"]]
    for t in group_tasks:
        model += transmission_start_time[transmission["group"]] >= task_finish_time[t]
    model += transmission_start_time[transmission["group"]] + transmission["length"] <= transmission["deadline"]
    model += T_end >= transmission_start_time[transmission["group"]] + transmission["length"]

# Objective: minimize total transmission time
model += T_end

# Solve the problem
model.solve(pulp.PULP_CBC_CMD(timeLimit=60, gapRel=0.01))


# Output the schedule
schedule = []
if pulp.LpStatus[model.status] == 'Optimal':
    print("Optimal Schedule:")
    for t in tasks:
        task_id = t["task_ids"][0]
        for k in range(t["arrival_time"], transmissions[t["group"] - 1]["deadline"]):
            if pulp.value(x[task_id, k]) == 1:
                print(f"Task {task_id} from group {t['group']} is scheduled at time {k}")
                schedule.append({"task_id": task_id, "start_time": k, "finish_time": k + 1, "group": t["group"]})
    print(f"Total completion time (T_end): {pulp.value(T_end)}")
else:
    print("No optimal solution found.")
    print(f"Solver status: {pulp.LpStatus[model.status]}")

# Add transmission times after the last task in each group and calculate waiting times
waiting_times = {}
group_intervals = {}  # To store intervals for each group

for transmission in transmissions:
    group_tasks = [task for task in schedule if task["group"] == transmission["group"]]
    if not group_tasks:
        print(f"No tasks were scheduled for Group {transmission['group']}.")
        continue
    last_task_finish_time = max([task["finish_time"] for task in group_tasks])
    transmission["start_time"] = last_task_finish_time
    transmission["end_time"] = transmission["start_time"] + transmission["length"]
    print(f"Group {transmission['group']} starts transmission at time {transmission['start_time']} and ends at time {transmission['end_time']}")

    # Calculate waiting time for each group as the difference between the end of transmission time and the arrival time of the group
    waiting_time = transmission["end_time"] - transmission["arrival_time"]
    waiting_times[transmission["group"]] = waiting_time

    # Calculate the interval (start, end) for tasks in this group
    first_task_start_time = min([task["start_time"] for task in group_tasks])
    group_intervals[transmission["group"]] = (first_task_start_time, last_task_finish_time)
    print(f"Group {transmission['group']} interval: {group_intervals[transmission['group']]}")

# Save model download counts, memory usage, waiting times, and intervals to different files
with open("optimal_task_memory_waiting_times.txt", "w") as file:
    for group, memory in total_memory_usage_per_group.items():
        file.write(f"Group {group}: Memory used {memory:.2f} MB (Total model downloads: {model_download_count[group]})\n")
    for group, wait_time in waiting_times.items():
        file.write(f"Group {group}: Waiting time {wait_time} units\n")
    for group, interval in group_intervals.items():
        file.write(f"Group {group}: Interval {interval} (start, end)\n")

# Group consecutive tasks from the same group together for plotting
merged_schedule = []
current_group = None
current_start = None
current_end = None

for entry in schedule:
    if entry['group'] == current_group and entry['start_time'] == current_end:
        current_end = entry['finish_time']
    else:
        if current_group is not None:
            merged_schedule.append({"group": current_group, "start_time": current_start, "finish_time": current_end})
        current_group = entry['group']
        current_start = entry['start_time']
        current_end = entry['finish_time']

merged_schedule.append({"group": current_group, "start_time": current_start, "finish_time": current_end})


# Add transmission times with hashed color
fig, ax = plt.subplots(figsize=(10, 6))

# Define a color map for different tasks
group_colors = {1: 'green', 2: 'blue', 3: 'red', 4: 'yellow', 5: 'purple'}
legend_patches = []

# Create a bar for each merged task block
for entry in merged_schedule:
    color = group_colors[entry['group']]
    ax.barh(entry['group'], entry['finish_time'] - entry['start_time'], left=entry['start_time'], height=0.4, color=color, align='center')
    if color not in [patch.get_facecolor() for patch in legend_patches]:
        legend_patches.append(mpatches.Patch(color=color, label=f'Group {entry["group"]}'))

# Add transmission times with hashed color
for transmission in transmissions:
    color = group_colors[transmission["group"]]
    ax.barh(transmission["group"] + 0.5, transmission["length"], 
            left=transmission["start_time"], 
            height=0.2, 
            color=color, 
            hatch='///',  # Hashed pattern for transmission
            edgecolor=color, 
            align='center')

    # Annotate the end time of the transmission
    transmission_end_time = transmission["end_time"]
    ax.text(transmission_end_time + 0.2, transmission["group"] + 0.5, f'{transmission_end_time}', 
            va='center', ha='left', fontsize=9, color='black')

# Set y-ticks for group IDs only
ax.set_yticks(list(group_colors.keys()))
ax.set_yticklabels([f"Task {group}" for group in group_colors.keys()])

# Add labels and customize the plot
ax.set_xlabel('Time')
ax.set_ylabel('Task')
ax.set_title('Optimal Task Scheduling Gantt Chart with Transmission Times')
ax.set_xlim(0, max([t['deadline'] for t in transmissions]) + 5)
ax.grid(True)

# Add the legend for the task groups
ax.legend(handles=legend_patches, title="Task Groups / Transmission")

# Save the Gantt chart plot
plt.tight_layout()
plt.savefig('optimal_gantt_chart_with_transmissions.png')  # Save the plot as a PNG file
plt.show()



    
with open('optimal_task_results.pkl', 'wb') as f:
    pickle.dump({
        'schedule': schedule,  # Save the schedule
        'group_intervals': group_intervals,  # Save the group intervals
        'transmission_tracker': {trans['group']: (trans['start_time'], trans['end_time']) for trans in transmissions}  # Save transmission times
    }, f)

# Save the schedule to a file
with open("optimal_task_schedule.txt", "a") as file:
    for entry in schedule:
        file.write(f"Task {entry['task_id']} from Group {entry['group']} scheduled at time {entry['start_time']} and ends at time {entry['finish_time']}\n")
    file.write(f"Total completion time (T_end): {pulp.value(T_end)}\n") 