# MediaVerse-CMR
Cross-Modal retrieval module for T3.2 of MediaVerse

## Installation
Create a Python virtual environment and activate it
```
python3 -m venv venv_mediaverse
source venv_mediaverse/bin/activate
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

Install common package in editable mode
```
pip install -e .
```

## Usage    
Use bash scripts located in `scripts/` to run computations.
