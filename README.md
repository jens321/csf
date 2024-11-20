# Can a MISL Fly? Analysis and Ingredients for Mutual Information Skill Learning
Official code repo for the paper "Can a MISL Fly? Analysis and Ingredients for Mutual Information Skill Learning". This paper introduces a new method which we call **Contrastive Successor Features (CSF)**, which achieves compareable performance to current SOTA unsupervised skill discovery methods while at its core relying on mutual information maximization.

## Installation 🔌

After cloning this repo, please run the following commands at the root of the project:
```
# Setting up the conda environment
conda create --name csf python=3.9
conda activate csf

# Installing dependencies
pip install -r requirements.txt --no-deps
pip install -e .
pip install -e garaged
pip install --upgrade joblib
pip install patchelf
```

> [!NOTE] 
> Pip might complain about incompatible versions -- this is expected and can be safely ignored.

Next, we need to do some Mujoco setup.
```
conda activate csf
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c anaconda mesa-libgl-cos6-x86_64
conda install -c menpo glfw3
```

We also need to tell Mujoco which backend to use. This can be done by setting the appropriate environment variables.
```
conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
conda deactivate && conda activate csf
```

If you don't already have Mujoco, you will need it. Install Mujoco in a folder called `.mujoco`. More instructions on how to do so are linked [here](https://pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html).

Finally, you may want to add the following environment variables to your `.bashrc` file:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CPATH=$CONDA_PREFIX/include
```

Remember to source your `.bashrc` file after changing it: `source ~/.bashrc`.

## Running Experiments 🏃‍♂️

For **unsupervised pretraining** (state coverage), you can use the following commands. Make sure to run these from the root of the project.
```
# Ant
sh scripts/pretrain/csf_ant.sh

# HalfCheetah
sh scripts/pretrain/csf_halfcheetah.sh

# Humanoid
sh scripts/pretrain/csf_humanoid.sh

# Quadruped
sh scripts/pretrain/csf_quadruped.sh

# Kitchen 
sh scripts/pretrain/csf_kitchen.sh

# Robobin
sh scripts/pretrain/csf_robobin.sh
```

> [!NOTE] 
> All experiments were run on a single GPU, usually with between 8 - 10 workers (see the `--n_parallel` flag).
> In addition, we found we needed 32GB of CPU memory (RAM) for all state-based experiments (Ant and HalfCheetah), while
> we needed 40GB of CPU memory for all image-based experiments (Humanoid, Quadruped, Kitchen, Robobin).

Once experiments are running, they will be logged under the `exp` folder.

## Videos of learned policies

Our key theoretical result is that a prominent skill learning algorithm (METRA) can be *reinterpreted* as doing mutual information maximization. This opens the door to a new skill learning method (CSF) that is simpler, has the same objective as a long line of prior work, and achieves results that are on par with the current SOTA. 

The videos below indeed show that our method (CSF) qualitatively learns similar skills to the prior method (METRA).

### Robobin
CSF
https://github.com/user-attachments/assets/dd60c6c2-caf9-4d23-84ad-ceb84e47e34b

METRA
https://github.com/user-attachments/assets/8b97e54f-4adf-4dd5-b849-e7860d5c7cfe

### Ant
CSF
https://github.com/user-attachments/assets/7676677c-235c-49d9-a400-6f88ba999e44

METRA
https://github.com/user-attachments/assets/69ca8d58-13a4-4c02-b48e-d6b230bfd201


## Acknowledgements
This code repo was built on the original [METRA repo](https://github.com/seohongpark/METRA).
