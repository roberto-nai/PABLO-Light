# PAttern-Based LOcal explanations (PABLO) - Light

## Usage
To rerun the experiments presented in the paper, we first need to download and unzip the datasets archive file found at this link: https://drive.google.com/file/d/1Jr4RM4dRlbi-aXAxNaifRVGLDOnhlIdN/view?usp=sharing.
Make sure that the datasets are placed within the project PABLO folder inside datasets folder.
Afterwards, a Conda/virtualenv environment has to be setup (see ```_virtualenv``` for instructions), then the requirements can be installed by opening a Terminal instance, going to the PABLO folder and running following command: ```pip install -e .```

The simple-index pipeline can by processed by running the ```python impressed_pipeline_simple_index.py```.

To adjust the datasets to be run, or the prefix lengths considered, update JSON configuration file ```datasets.json```.  

## Output

### /datasets_encoded  
Contains prefixes (P) and its encoding (E).  

### /datasets_positions
Contains encoding of activity positions after synthesised data generation (/datasets_synth).  

### /datasets_sublog  
Contains the sublogs of each case id, created on the basis of the generated prefix (/datasets_encoded).  

### /datasets_synth
Contains the synthetic dataset, generated from the sublogs (/datasets_sublog).  



@TODO: timestamps in dice_impressed functions