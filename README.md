# RecipeMAG
Metapath Aggregated Graph Neural Network for Recipe Recommendation


## Create and Activate the Conda Environment

To set up the environment for this project, follow these steps:

### 1. Create the environment from the `environment.yml` file:
```bash
conda env create -f environment.yml
```

### 2. Activate the environment:
```bash
conda activate recipemag
```

### 3. Verify installation:
```bash
conda list
```

## Updating the Environment

If new dependencies are added, update your local environment with:
```bash
conda env update --file environment.yml --prune
```

## Exporting the Environment (for contributors)

If you add dependencies, save them to `environment.yml` for others:
```bash
conda env export --name recipemag > environment.yml
```

## Deactivating and Removing the Environment (if needed)

To deactivate:
```bash
conda deactivate
```

To remove the environment:
```bash
conda env remove --name recipemag
```
