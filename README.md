# Inductive Moment Matching

This repository is an open-source implementation of the Inductive Moment Matching (IMM) paper. 
We compare IMM with diffusion models, flow matching and consistancy models on some standard generative benchmarks.

Models and utils were taken from Lummalabs.ai repo: https://github.com/lumalabs/imm


The original Inducitve Moment Matching paper can be found here:
@misc{zhou2025inductivemomentmatching,
      title={Inductive Moment Matching}, 
      author={Linqi Zhou and Stefano Ermon and Jiaming Song},
      year={2025},
      eprint={2503.07565},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.07565}, 
}


---
### Installation
To install IMM, use the following steps:

```bash
# Clone the repository
git clone git@github.com:owenhowell20/IMM.git
cd IMM

# Install dependencies
conda env create -f environment.yml
```

To check the installation, run:
```bash
python3 -m pytest tests
```


To compare with flow matching and diffusion models


