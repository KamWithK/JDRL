# Reinforcement Learning: Jelly Drift
## AI Playing Games - Racing Time
Driving a car within a video game is a relatively easy and relaxing process for a human.
Here we try and train an AI to race a car around a race track as fast as possible!

This project uses the environment [created here](https://github.com/EuclideanEncabulator/gym-jd)!

## Project Structure
To install and use `environment.yml` do the following:
1. Install Anaconda or Miniconda
2. Create a new environemtn from the file `conda env create -f environment.yml`
3. When packages need to be added or modified update the `environment.yml` file and run `conda env update -f environment.yml --prune`
4. Once satisfied activate the new environment through `conda activate jd_rl` and (optionally) deactivate/close afterwards with `conda deactivate`

*For an update the above isn't necessary - simply run `conda update --all` for packages and `conda update -n base -c defaults conda` for conda itself!*

## Config
This project has been set up using Hydra.
To start a run please create a run config file under `config/runs/` and specify it as a command line argument - `python src/main.py ++runs=test_run`.
For reproducibility avoid modifying the default configuration and instead override it within the run config file.
