# MediaVerse-CMR
Cross-Modal retrieval module for T3.2 of MediaVerse

## Installation
Create a conda virtual environment and activate it
```
conda create --name mediaverse python=3.8 anaconda
conda activate mediaverse
```

Install requirements
```
pip install -r requirements
```

Install spacy vocabulary
```
python -m spacy download en_core_web_sm
```

Install clip from [clip repository](https://github.ckom/openai/CLIP)

Install faiss from conda forge following [faiss install](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

Install common package in editable mode
```
pip install -e .
```

## Usage
You have to download MSCOCO dataset on your computer.
See the bash scripts located in `scripts/` to have an example of how to run computations.
