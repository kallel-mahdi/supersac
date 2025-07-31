import pandas as pd
import wandb
import os
import numpy as np
import itertools



import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("/home/mahdi/Desktop/supersac/notebooks/plots/ablations_tmlr/", exist_ok=True)
sns.set_theme()

# Set up matplotlib defaults for consistent plots
plt.rcParams['figure.figsize'] = (10, 6)  # Standard figure size
plt.rcParams['figure.dpi'] = 100  # Standard resolution

# Font sizes
plt.rcParams['font.size'] = 14  # Base font size
plt.rcParams['axes.titlesize'] = 22  # Plot title
plt.rcParams['axes.labelsize'] = 18  # Axis labels
plt.rcParams['xtick.labelsize'] = 14  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 14  # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 18  # Legend text

# Line and marker styles
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8

# Grid settings
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Legend settings
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.8
plt.rcParams['legend.edgecolor'] = 'gray'

# Save figure settings
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.2


os.environ["WANDB_API_KEY"]="28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
api = wandb.Api()
entity= "mahdikallel"


runs = api.runs(entity + "/" + "GAE_LAMBDA_ABLATIONS")




def filter_fn(run,env_name, filter_dict):

    
    if run.config["env_name"]==env_name:
        for key, value in filter_dict.items():
            if key not in run.config or run.config[key] != value:
                return False
        return True
    return False



# dc = { 
#         #"PPO+":{"algo_name":"superppo","gamma":0.99,"on_policy_data":False,"temperature":1.,"buffer_size":50000,"use_layer_norm":True,"clipping_ratio":0.25,"num_critics":2,"bound_actions":True,"gae_lambda":0.5,"min_target":False},
# "PPO (Lambda=0.95)":{"algo_name":"ppo","gamma":0.99,"gae_lambda":0.95},
# "PPO (Lambda=0.7)":{"algo_name":"ppo","gamma":0.99,"gae_lambda":0.7},
# "PPO (Lambda=0.5)":{"algo_name":"ppo","gamma":0.99,"gae_lambda":0.5},
# "PPO (Lambda=0.0)":{"algo_name":"ppo","gamma":0.99,"gae_lambda":0.0},
# }


dc = {  
        #"PPO+":{"algo_name":"superppo","gamma":0.99,"on_policy_data":False,"temperature":1.,"buffer_size":50000,"use_layer_norm":True,"clipping_ratio":0.25,"num_critics":2,"bound_actions":True,"gae_lambda":0.5,"min_target":False},
"PPO+ (Lambda=0.95)":{"algo_name":"superppo","gae_lambda":0.95},
"PPO+ (Lambda=0.7)" :{"algo_name":"superppo","gamma":0.99,"on_policy_data":False,"temperature":1.,"buffer_size":50000,"use_layer_norm":True,"clipping_ratio":0.25,"num_critics":2,"bound_actions":True,"gae_lambda":0.7,"min_target":False},
"PPO+ (Lambda=0.5)" :{"algo_name":"superppo","gamma":0.99,"on_policy_data":False,"temperature":1.,"buffer_size":50000,"use_layer_norm":True,"clipping_ratio":0.25,"num_critics":2,"bound_actions":True,"gae_lambda":0.5,"min_target":False},
"PPO+ (Lambda=0.0)" :{"algo_name":"superppo","gamma":0.99,"on_policy_data":False,"temperature":1.,"buffer_size":50000,"use_layer_norm":True,"clipping_ratio":0.25,"num_critics":2,"bound_actions":True,"gae_lambda":0.,"min_target":False},
}
      
env_names = ["Hopper-v5","Walker2d-v5","HalfCheetah-v5"]
max_steps = [1e6,1e6,1e6]

fig, axs = plt.subplots(1, 3, figsize=(24, 8))
# Reset runs chain for the outer loop
for ax, env_name, max_step in zip(axs.flatten(), env_names, max_steps):
    # Reset the runs chain for each environment
    runs = api.runs(entity + "/" + "GAE_LAMBDA_ABLATIONS")

    
    for algo in dc.keys():
        print(algo, env_name)
        
        df = pd.DataFrame()
        config_list, name_list, run_list = [], [], []
        for run in runs:
            print(run.config["algo_name"],run.config["gae_lambda"],run.config["env_name"])
            if filter_fn(run, env_name, dc[algo]):
                
                config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
                name_list.append(run.name)
                full_df = run.history(samples=10000)
                tmp_df = full_df[['evaluation/undisc_policy_return', '_step']].dropna(axis=0)
                tmp_df["average return"] = tmp_df['evaluation/undisc_policy_return'].rolling(window=5).mean()
                tmp_df['step'] = np.round(tmp_df['_step'] / 10000) * 10000
                tmp_df = tmp_df[["step", "average return"]]
                df = pd.concat([df, tmp_df], ignore_index=True)
                print(run.name)
                
        df2 = df[df["step"] < int(max_step)]
        x = df2.groupby("step")["step"].mean() / 1e6
        mean = df2.groupby("step")["average return"].mean()
        stderror = df2.groupby("step")["average return"].std() / np.sqrt(df2.groupby("step")["average return"].count())
        
        ax.plot(x, mean, label=algo.upper(), linewidth=3)
        ax.fill_between(x, mean - stderror, mean + stderror, alpha=0.3)
        ax.set_xlabel('Million Steps')
        ax.set_ylabel('Policy Return')
        ax.set_title(env_name)
        
     

# Create one common legend on top of the figure
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(dc.keys()), bbox_to_anchor=(0.5, 1.05))
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("/home/mahdi/Desktop/supersac/notebooks/plots/ablations_tmlr/lambda_ppo_plus.pdf", format="pdf", bbox_inches="tight")
plt.show()


