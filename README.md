# KnowledgeNet
Create sparse neural networks to be used as knowledge graphs


## Use KnowledgeNet

### Setup an experiment
#### Setup the experiment directory

1. **Create a new directory with the name of the experiment to store the data 
files and results of the experiment**. We suggest doing this one level 
up from the knowledge-net directory, but the choice of where to keep the 
directory is irrelevant as long as the correct path is specified in the 
`config.py` file. For example, we create a directory called `regression` 
within the`tests` directory in the KnowledgeNet master folder. The absolute path 
to the folder is then `KnowledgeNet/tests/regression`. 

2. **Within the experiment directory you just created, create two new
   directories called** `data` **and** `results`. These directories already
exist in the regression example experiment. 

3. **Create a new file called** `features.tsv` **in the** `data` **directory**.
The file should be structured as a two-column TSV file, with the first
column titled `ID`, and the second column titled `Feature` (ensure first letter
is capitalized). The `id` column
should begin with `0` and continue until the number of features - 1. The
`feature` column should contain the name of each feature used by the model. See
the `features.tsv` file within `tests/regression/data` folder for an example. 





