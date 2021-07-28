import argparse
import json
import logging
import os
import pdb
import time

import requests
import torch
#from config import SetupParameters
from flask import Flask, abort, jsonify, make_response, request
from flask_cors import CORS
#from logger import CustomLogger
#from logstash_formatter import LogstashFormatterV1

from model import CMRNet

app     = Flask(__name__)
cors    = CORS(app)

os.makedirs('log/', exist_ok=True)

flasklog    = open('log/flask.log', 'a+')
handler     = logging.StreamHandler(stream=flasklog)
#handler.setFormatter(LogstashFormatterV1())
logging.basicConfig(handlers=[handler], level=logging.INFO)

_MAX_LEN = 150


@app.route('/cmr/v0.1/scorer', methods=['POST'])
def scorer():
    """This API returns the score assigned to a <text, annotations> pair.
    
    Returns:
        [list] -- list of dictionaries containing, for each input string, the list of entities with type, value and offset.
        e.g.
        {'sentence': 'Mr. Robinson lives in Reeds.'
        'entities':[
            {
                'offset': 0,
                'type': PER,
                'value': Mr. Robinson
            },
            {
                'offset': 22,
                'type': LOC,
                'value': Reeds 
            }
        ]} 
    """
    start         = time.time()
    input_params  = request.get_json()
    query, items  = input_params['query'], input_params['items']

    model               = app.config['Scorer']

    instances           = [item['annotations'] for item in items]
    scores              = model.infer(query, instances)
    assert len(items) == len(scores)
    results             = [{'id': curr_item['id'], 'score': round(curr_score, 4)} for curr_item, curr_score in zip(items, scores)]
    end                 = time.time()
    execution_t         = end-start

    log_dict = {'remote_addr': request.remote_addr,
                'start': start,
                'response_time': execution_t,
                'request': input_params,
                'response': scores}
    #app.config['logger'].log(log_dict)

    return jsonify(results), 200


@app.route('/cmr/v0.1/max_len', methods=['GET'])
def get_max_len():
    return jsonify({'max_len': _MAX_LEN})


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'Size too large. Max size {} words'.format(_MAX_LEN)}), 400)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        type=int,
        required=True,
        help='Port to allocate for the rest api'
    )
    parser.add_argument(
        '--cuda',
        required=False,
        default=False,
        action='store_true',
        help='Flag to enable inference on gpu'        
    )

    args = parser.parse_args()
    with open('word2id.json') as fp:
        word2id = json.load(fp)
    state_dict = torch.load('best_model.pth', map_location=torch.device('cpu'))
    app.config['Scorer'] = CMRNet(word2id=word2id)
    app.config['Scorer'].load_state_dict(state_dict)
    #app.config['logger'] = CustomLogger('log/cmr.log')
    app.run(host='0.0.0.0', debug=False, port=args.port)
