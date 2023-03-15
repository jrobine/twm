# Transformer-based World Models (TWM)

WIP

Install packages from `requirements.txt`. Also make sure you have installed the Atari environments correctly.  
For more information, see: https://github.com/openai/gym/releases/tag/v0.21.0

Execute the following command to run an experiment:  
```
python -O twm/main.py --game Breakout --seed 0 --device cuda:0 --cpu_p 1.0 --wandb disabled 
```

Use `--wandb online` to log the metrics in weights and biases.  
To use other hyperparameters, edit the file `twm/config.py`.
