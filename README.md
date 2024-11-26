# PAttern-Based LOcal explanations (PABLO) for Outcome-Based Predictive Process Monitoring

This repository provides the implementation for generating pattern-ware local explanations for Outcome-Base PPM,
using the PABLO framework introduced in the paper.

## Usage
To rerun the experiments presented in the paper, we first need to download and unzip the datasets archive file found at this link: https://drive.google.com/file/d/1Jr4RM4dRlbi-aXAxNaifRVGLDOnhlIdN/view?usp=sharing.
Make sure that the datasets are placed within the project PABLO folder inside datasets folder.
Afterwards, a Conda/virtualenv environment has to be setup (see ```_virtualenv``` for instructions), then the requirements can be installed by opening a Terminal instance, going to the PABLO folder and running following command: ```pip install -e .```

The simple-index results can be obtained by running the ```python impressed_pipeline_simple_index.py```.

To adjust the datasets to be run, or the prefix lengths considered, update the ```datasets.json```.  
```

@TODO: Additional instructions will be included in the README.md as soon as possible. 


@TODO: timestamps in dice_impressed functions