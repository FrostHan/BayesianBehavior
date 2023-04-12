# Code for the Bayesian Behaviors framework

Paper preprint at:
https://arxiv.org/abs/2304.05008 

credit: Dongqi Han

## How to use

### Learn habitual behaviors

To train the agent to learn habitual behaviors, you can run a command like (see the python file for the arguments)

```bash
python train_habitual_behaviors.py --seed 0 --verbose 1 --gui 0
```

The result data will be saved at ./data/ and ./details/ in .mat files, for which you can load using MATLAB or scipy:

```python
import scipy.io as sio
data = sio.loadmat("xxx.mat")
```

The trained PyTorch model will also be saved at ./data/.

### Performing goal-directed behaviors

After acquiring the habitual bebahviors by self-exploration (we provided an trained model at ./data/ if you want to skip training), the agent can perform goal-directed planning with goal provided

```bash
python test_goal_directed.py --seed 0 --goal_marker red --gui 1
```

where the goal_marker can be "full", "red", "blue", "less_blue" or "less_red" (see the paper).

The data will be save at ./aif/ containing 6 goal-directed episodes.
