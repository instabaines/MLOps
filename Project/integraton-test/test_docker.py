# pylint: disable=duplicate-code

import json

import requests
from deepdiff import DeepDiff

with open('example_test.json', 'rt', encoding='utf-8') as f_in:
    test = json.load(f_in)


url = 'http://192.168.23.104:9696/predict'
actual_response = requests.post(url, json=test).json()
print('actual response:')

print(json.dumps(actual_response, indent=2))

expected_response = {'Predicted Sales': 4445.678823678149}
    

diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')

assert 'type_changes' not in diff
assert 'values_changed' not in diff
