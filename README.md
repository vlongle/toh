# Tower of Hanoi

## About
A OpenAI gym environment.

## Installation
(**Developer**) Run
```
rm requirements.txt && pip freeze > requirements.txt
```
to dump pip dependencies to `requirements.txt` file.

(**User**) Run (inside a conda environment)
```
pip install -r requirements.txt
```
to install the correct versions of dependencies.



## Scratch


                    [0, 0]
                /           \
            [1, 0] -------  [2, 0]
            /                  \
          [1, 2]              [2, 1] 
          /   \               /      \
    [2, 2]---- [0, 2] ----- [0, 1] -- [1, 1]