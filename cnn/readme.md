# Search Phase
Search phase currently supports the following algorithms:
* DARTS
* EDARTS (exponentiated first order DARTS)

## Architects
Each search method has its own associated architect:
* DARTS (`architect.py`)
* EDARTS (`architect_edarts.py`)

## Training
Two scripts are used for training in the search phase:
* `train_search.py` is standard DARTS training with additional options.
* `train_search_alternating.py` alternates between steps on supernet and steps on architecture parameters.

# Evaluation Phase
Training script has been modified with additional options for AutoAugment and soft cross entropy training loss.  See `argument_manager.py` for additional info on possible arguments.
