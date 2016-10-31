import json
from pprint import pprint
res = {}
with open('data/Abhishek_0.json') as infile:
	res = json.load(infile)

pprint(res)