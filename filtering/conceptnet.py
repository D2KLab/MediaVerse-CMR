import requests
import pdb

concept      = 'dog'
lang         = 'en'
go_next_page = True  

#each page contains up to 20 relations for that node. To see more go to next page using obj['view']['nextPage']
while go_next_page:
    concept = input('Enter the name of the concept to search for: ')
    concept = concept.lower()
    obj     = requests.get('http://api.conceptnet.io/c/{}/{}'.format(lang, concept)).json()
    if 'error' in obj:
        print(obj['error'])
        continue
    pdb.set_trace()
    edges_l = ['* {} {} {}'.format(edge['start']['label'], edge['rel']['label'], edge['end']['label']) for edge in obj['edges']]
    print('\n'.join(edges_l))
    user_will    = input('\n -- Go to the next page? (y/n)')
    go_next_page = (user_will in ('y', 'yes'))
        

