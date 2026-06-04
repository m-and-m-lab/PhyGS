# Define the session name
SESSION="infinigen_GPU_batch"

# 1. Create the session (detached) with an initial monitor window
tmux new-session -d -s $SESSION -n "overview_gpu_batch"

# 2. Loop to create 3 windows, each running a unique instance
for i in {0..3}; do
    # Define the unique command for this instance
    # Note: We added a 'read' at the end so the window stays open even if the script finishes/fails
    CMD="python -m infinigen_examples.generate_indoors --seed $i --task coarse --output_folder outputs/indoors/coarse/indoor_$i -g fast_solve.gin overhead.gin -p compose_indoors.terrain_enabled=False; echo 'Execution finished.'; read"
    
    # Create the new window and run the command
    tmux new-window -t $SESSION -n "indoor_$i" "$CMD"
done

# 3. Attach to the session to see the work
tmux attach -t $SESSION
