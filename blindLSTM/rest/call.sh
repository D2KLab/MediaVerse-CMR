#!/bin/bash

#todo remember to activate the server using `python api.py --port 6000`

curl -i -H "Content-Type: application/json" -X POST -d\
            '{"query": "A man on a bicycle",
              "items": [{"id": 721, "annotations": ["person", "car", "bicycle"]},
                        {"id": 199, "annotations": ["boat", "dog", "car"]}]
             }'\
            http://localhost:6000/cmr/v0.1/scorer