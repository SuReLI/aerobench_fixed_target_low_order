# Reliability assessment of off-policy deep reinforcement learning: a benchmark for aerodynamics  


**Note:** This codebase was utilized to generate the results presented in the paper titled *"Reliability assessment of off-policy deep reinforcement learning: a benchmark for aerodynamics."* The repository includes implementations of three reinforcement learning algorithms—DDPG, TD3, and SAC—along with the necessary setup to reproduce and analyze the benchmark results. For detailed information on the experiments, methodology, and findings, please refer to the associated paper.
  
This project examines three existing reinforcement learning algorithms which store collected samples in a replay buffer: **DDPG**, **TD3**, and **SAC**. These are evaluated and compared on a fluid mechanics benchmark which consists in **controlling an airfoil to reach a target**. The problem is solved with two different levels of data collection complexity: either a **low-cost low-order model** or with a high-fidelity **Computational Fluid Dynamics** (CFD) approach.  
  
In practice, two different control tasks are performed. First, both the starting and target points are kept in a fixed position during both the learning and testing of the policy, whereas in the second task, the target may be anywhere in a given domain. The code allows to evaluate the three DRL algorithms on both tasks, when solving the physics with either a low-order or a high-fidelity model, and with various DRL hyperparameters, reward formulations, and environment parameters controlling the dynamics.  
  
In order to facilitate the reproducibility of our results without requiring an in-depth understanding of the code, each case study is stored in a separate repository containing all the necessary code and setup to execute the case directly. The code for the following tasks can be found in the respective repositories:  
- [First task with fixed target and low-order model](https://github.com/SuReLI/aerobench_fixed_target_low_order)  
- [First task with fixed target and CFD model](https://github.com/SuReLI/aerobench_fixed_target_star)  
- [Second task with variable target and low-order model](https://github.com/SuReLI/aerobench_variable_target_low_order)  
- [Second task with variable target and CFD model](https://github.com/SuReLI/aerobench_variable_target_star).  
  
  
## Available algorithms  
- **DDPG** : Deep Deterministic Policy Gradient presented in [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971).  
- **TD3** : Twin Delayed Deep Deterministic policy gradient [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf)  
- **SAC** : Soft Actor-Critic presented in [Soft Actor-Critic:Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf)  
  
  
## Installation  
Ensure you have Python 3.7 or older versions installed, and then install the necessary Python libraries:  
  
```bash  
pip install torch torchvision imageio gym matplotlib PyYAML numpy  
```  
Clone the repository:  
```bash  
git clone https://github.com/SuReLI/aerobench_fixed_target_low_order
```  
  
## Usage  
  
### Training phase  
Navigate to the directory to run the first task with the low-order model:  
```bash  
cd aerobench_fixed_target_low_order
```  
Initiate training using the following command:
```bash
python3 -u train <agent> --appli='flatplate'  
```  
Replace `<agent>` with one of the following values: DDPG, TD3, or SAC. Note that the 'flatplate' application refers to the usage of the low-order model to solve the physics at stake.  
  
Optional parameters for loading pre-existing models and memory replay buffers and continue the training from there are available:  
- `--load`: Load a pre-existing model.  
- `--loadrm`: Load a pre-existing memory buffer.
  
Example:  
```bash  
python3 -u train SAC --appli='flatplate' --load='results/SAC/flatplate_2023-12-13_16-46-40' --loadrm='results/SAC/flatplate_2023-12-13_16-46-40'  
```  
This command trains the specified reinforcement learning agent (SAC in this case) on the 'flatplate' application (low-order model) with the option to load a pre-existing model (soft_actor.pth, critic_target and critic.pth) from the folder `results/SAC/flatplate_2023-12-13_16-46-40/models/` and a pre-existing memory buffer (replay_memory.yaml) from the folder `results/SAC/flatplate_2023-12-13_16-46-40/`.


Alternatively, if you are on a supercomputer, you can launch the training using a slurlm file. An example slurlm file, called **submit_example**, is provided  in the repository.


  
### Testing phase
Navigate to the directory root:
```bash 
cd aerobench_fixed_target_low_order
``` 
Initiate testing using the following command:
```bash
python3 -u test <agent> --appli='flatplate'
``` 
Just like in the training phase, replace `<agent>` with one of the following values: DDPG, TD3, or SAC.

Optional parameters for testing are available:

-   `-n` or `--nb_tests`: Set the number of test episodes.
-   `-f` or `--folder`: Specify the path to a specific result folder to test. If not provided, the default folder tested is the most recent one with a format similar to `flatplate_2023-12-13_16-46-40` inside the `/results/<agent>/` directory. 
Note: the model tested is the one contained in the `/models/` subdirectory of the specified result folder.

Example:
```bash
python3 -u test SAC --appli='flatplate' -n 10 -f='results/SAC/first_trial'
``` 
This command tests the pre-trained model stored in the folder `results/SAC/first_trial/models/`, on the 'flatplate' application, running 10 test episodes.


Alternatively, if you are on a supercomputer, you can launch the testing using a slurlm file. An example slurlm file, called **submit_example**, is provided  in the repository.
 

## Outputs

After running the training or testing phases, the code generates various outputs and results. Below is an overview of the key directories and files you can expect:

### Training Outputs:

For each training, results are stored in a directory of the form `results/<agent>/flatplate_date/`, where date is the date at which the training started. The folder contains the following outputs:
- **training plot (`train_output.png`):** a visual representation of the training (return, specific trajectories and location of point B)
- **model checkpoints (`models/*.pth`)**,
- **memory buffer (`replay_memory.yaml`)**
- **additional variable files (`variables/*.csv`):** contain CSV files with the values of various variables during the training episodes.
- **configuration File (`config.yaml`):** a copy of the configuration file used for the specific training run.

### Testing Outputs:

For each testing, results are stored in a sub-directory of the results directory tested :`results/<agent>/flatplate_date/test`. The `test` folder contains the following outputs:

- **testing plot (`test_output.png`):** a visual representation of the testing (return, specific trajectories and location of point B)
- **additional variable files (`variables/*.csv`):** contain CSV files with the values of various variables during the testing episodes.


  
## Run the different cases from the paper (or customized cases)

To customize the case, one can adjust the values of various parameters in the **config.yaml** file, or if necessary (i.e. if the config.yaml file does not allow it), modify the code in the **flatplate.py** file.

Specifically, all the cases documented in the article *Reliability assessment of off-policy deep reinforcement learning: a benchmark for aerodynamics* can be reproduced by modifying the parameters in config.yaml, with the exception of setting the c<sub>2</sub> constant and the R<sup>-1</sup> case, which must be adjusted in the flatplate.py file, within the update_reward_if_done and compute_reward functions, respectively.
  
  
## Acknowledgments  
  
The reinforcement learning algorithms implemented in this project have been adapted from the [Pytorch-RL-Agents](https://github.com/SuReLI/Pytorch-RL-Agents) repository.  
  
## Contact  
For any questions or comments, feel free to contact Sandrine Berger at [sand.qva@gmail.com](mailto:sand.qva@gmail.com).
