# Code for the Bayesian Behaviors framework

Link to the paper: <https://arxiv.org/abs/2304.05008>

credit: Dongqi Han

## Installation

Tested using Python 3.7.7 on Ubuntu 20.04 and Windows 11

### Install Requirements (typically a few minutes)

```bash
pip install -r requirements.txt 
```

And you also need to install PyTorch. Please install PyTorch >= 1.11 that matches your CUDA version according to <https://pytorch.org/>.

## How to use

----------------------------------

### Learn habitual behaviors

To train the agent to learn habitual behaviors, you can run a command like (see the python file for the arguments)

```bash
python train_habitual_behaviors.py --seed 42 --verbose 1 --gui 0
```

Set `--gui 1` if you want to see the visualized environment.

The default arguments (hyperparameters) are the same as used in the paper. For the information of the arguments in training the habitual behavior, see `train_habitual_behaviors.py`

To run the models for ablation study, use the argument --abalation.

### Data format

After training the agent's habitual behavior (less than 1 day with a computer with a descent GPU), the result data will be saved at `./data/` and `./details/` in .mat files, for which you can load using MATLAB or scipy:

```python
import scipy.io as sio
data = sio.loadmat("xxx.mat")
```

The PyTorch model of the trained agent will also be saved at `./data/`, which can be loaded by `torch.load()`.

----------------------------------

### Performing goal-directed behaviors

After acquiring the habitual bebahviors by self-exploration (the model saved at `./data/`), the agent can perform goal-directed planning with the goal provided.

We provide a pre-trained model at ./data/ if you want to skip training, to use the pre-trained model:

```bash
python test_goal_directed.py --seed 0 --goal_marker red --gui 1
```

where the goal_marker can be "full", "red", "blue", "less_blue" or "less_red" (see the paper).

The result data will be save at `./aif/` containing 6 goal-directed episodes.
