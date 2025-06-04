import wandb
import os

# Set up wandb API
os.environ["WANDB_API_KEY"] = "28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
api = wandb.Api()
entity = "mahdikallel"
project = "GAE_LAMBDA_ABLATIONS"

# Get all runs from the project
runs = api.runs(entity + "/" + project)

print("Checking runs for env_name configurations...")
runs_to_update = []

# Check which runs need updating
for run in runs:
    if 'env_name' in run.config:
        env_name = run.config['env_name']
        if not env_name.endswith('-v5'):
            runs_to_update.append((run, env_name))
            print(f"Run {run.name}: '{env_name}' -> '{env_name}-v5'")
        else:
            print(f"Run {run.name}: '{env_name}' (already correct)")
    else:
        print(f"Run {run.name}: No env_name config found")

print(f"\nFound {len(runs_to_update)} runs that need updating.")

if runs_to_update:
    print("\nUpdating runs...")
    
    for run, old_env_name in runs_to_update:
        try:
            # Update the config
            new_env_name = old_env_name + '-v5'
            run.config['env_name'] = new_env_name
            run.update()
            print(f"✓ Updated run {run.name}: '{old_env_name}' -> '{new_env_name}'")
        except Exception as e:
            print(f"✗ Failed to update run {run.name}: {e}")
    
    print(f"\nCompleted updating {len(runs_to_update)} runs.")
else:
    print("\nNo runs need updating.")

print("\nFinal verification:")
runs = api.runs(entity + "/" + project)  # Refresh the runs
for run in runs:
    if 'env_name' in run.config:
        env_name = run.config['env_name']
        status = "✓" if env_name.endswith('-v5') else "✗"
        print(f"{status} Run {run.name}: '{env_name}'") 