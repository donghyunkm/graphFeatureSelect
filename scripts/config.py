# Open a file to write
with open("config.txt", "w") as file:
    # Write the header
    file.write("ArrayTaskID Seed Fold\n")
    
    # Initialize variables
    array_task_id = 1
    seeds = range(10)
    folds = range(5)
    
    # Generate all combinations and write to the file
    for seed in seeds:
        for fold in folds:
            file.write(f"{array_task_id} {seed} {fold}\n")
            array_task_id += 1

print("config.txt has been generated with 500 rows.")
