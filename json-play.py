import json

data = json.load(open('/Users/nmoran/pearl_river_raw_dump.json'))

for thing in data['bot23_fleet22_20240715T174656']['vehicle']:
    print(thing)