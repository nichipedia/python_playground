import requests
import json
import ast

base_url = 'http://10.23.22.13:40010'

#path_opts = ['jaiabot::helm_ivp', 'jaiabot::imu', 'jaiabot::intervehicle_subscribe_request', 'jaiabot::low_control', 'jaiabot::metadata', 'jaiabot::pressure_adjusted', 'jaiabot::pressure_temperature', 'jaiabot::salinity']
path_opts = ['jaiabot::imu', 'jaiabot::low_control', 'jaiabot::metadata', 'jaiabot::pressure_adjusted', 'jaiabot::pressure_temperature', 'jaiabot::salinity']

def scrape_jaia_data():

    logs_raw = requests.get(base_url + '/logs')

    logs = json.loads(logs_raw.text)

    log_json = {}

    for bot in logs['logs']:
        filename = bot['filename']
        log_data = requests.get(f'{base_url}/task-packet?log={filename}')
        log_json[filename] = json.loads(log_data.text)
        
    return log_json

def get_logs():
    logs_raw = requests.get(base_url + '/logs')
    if logs_raw.status_code != 200:
        return None
    return json.loads(logs_raw.text)

def scrape_depth_contours(logs):
    log_json = {}
    for bot in logs['logs']:
        log_name = bot['filename']
        depth_data = requests.get(f'{base_url}/depth-contours?log={log_name}')
        if depth_data.status_code == 200:
            log_json[log_name] = json.loads(depth_data.text)
        else:
            log_json[log_name] = {}

def scrape_interpolated_drift(logs):
    log_json = {}
    for bot in logs['logs']:
        log_name = bot['filename']
        drift_data = requests.get(f'{base_url}/interpolated-drifts?log={log_name}')
        if drift_data.status_code == 200:
            log_json[log_name] = json.loads(drift_data.text)
        else:
            log_json[log_name] = {}

def get_raw_paths(log_name):
    root_paths = requests.get(f'{base_url}/paths?log={log_name}&root_path=')
    if root_paths.status_code == 200:
        return ast.literal_eval(root_paths.text)
    else:
        return []


def scrape_raw_data(logs):
    raw_json = {}
    for bot in logs['logs']:
        log_name = bot['filename']
        raw_json[log_name] = {}
        map_position = requests.get(f'{base_url}/map?log={log_name}')
        if (map_position.status_code == 200):
            raw_json[log_name]['position'] = map_position.text
        else:
            print(f'Not able to read position for {log_name}')
        for path in path_opts:
            requests.get(f'{base_url}/convert-if-needed')
            proto_check = requests.get(f'{base_url}/paths?log={log_name}&root_path=')
            if (proto_check.status_code == 200 and len(proto_check.text) > 0):
                proto_name_resp = requests.get(f'{base_url}/paths?log={log_name}&root_path={path}')
                if proto_name_resp.status_code == 200:
                    proto_names = ast.literal_eval(proto_name_resp.text)
                    for proto_name in proto_names:
                        data_available_resp = requests.get(f'{base_url}/paths?log={log_name}&root_path={path}/{proto_name}')
                        if (data_available_resp.status_code == 200):
                            data_available = ast.literal_eval(data_available_resp.text)
                            data_available.remove('_datenum_')
                            data_available.remove('_scheme_')
                            data_available.remove('_utime_')
                            for data_series in data_available:
                                #print(f'Reading {log_name}/{path}/{proto_name}/{data_series}')
                                series_resp = requests.get(f'{base_url}/series?log={log_name}&path={path}/{proto_name}/{data_series}')
                                if series_resp.status_code == 200:
                                    series = json.loads(series_resp.text)
                                    raw_json[log_name][data_series] = series
                                else:
                                    print(f'Issue reading daw for {log_name}/{path}/{proto_name}/{data_available}: {series_resp.text}')
                        else:
                            print(f'Issue reading daw for {log_name}/{path}/{proto_name}: {data_available_resp.text}')
                else:
                    print(f'Issue reading daw for {log_name}/{path}: {proto_name_resp.text}')
            else:
                print('No Logs available ')
    print(json.dumps(raw_json))

def format_data():
    #collects = json.load(open('/Users/nmoran/pearl-river-dump.json'))
    collects = scrape_jaia_data()
    header = 'bot_id,bottom_dive,estimated_bottom_type,depth_achieved,dive_rate,duration_to_acquire_gps,max_acceleration,mean_depth,mean_salinity,mean_temp,powered_rise_rate,unpowered_rise_rate,start_lat,start_lon,drift_duration,drift_end_lat,drift_end_lon,drift_start_lat,drift_start_lon,est_drift,end_time,start_time'

    collect_gen = (collect for collect in collects if len(collects[collect]) != 0)
    for collect in collect_gen:
        file = open(f'/Users/nmoran/tmp/jaia/box1/{collect}.csv', 'w')
        file.write(header + '\n')
        task_packets = (collects[collect])
        for dive_task in task_packets:
            bot_id = dive_task['bot_id']
            start_time = dive_task['start_time']
            end_time = dive_task['end_time']

            #dive data
            dive = dive_task['dive']
            bottom_dive = dive['bottom_dive']
            bottom_type = dive['bottom_type']
            depth_achieved = dive['depth_achieved']
            dive_rate = dive['dive_rate']
            duration_to_acquire_gps = dive['duration_to_acquire_gps']
            max_acc = dive['max_acceleration']
            powered_rise_rate = dive.get('powered_rise_rate', None)
            start_lat = dive['start_location']['lat']
            start_lon = dive['start_location']['lon']
            unpowered_rise_rate = dive['unpowered_rise_rate']
            drift_duration = dive_task['drift']['drift_duration']
            estimated_drift = dive_task['drift']['estimated_drift']['speed']
            drift_end_lat = dive_task['drift']['end_location']['lat']
            drift_end_lon = dive_task['drift']['end_location']['lon']
            drift_start_lat = dive_task['drift']['start_location']['lat']
            drift_start_lon = dive_task['drift']['start_location']['lon']
        
            measurements = dive['measurement']
            for point in measurements:
                mean_depth = point['mean_depth']
                mean_salinity = point['mean_salinity']
                mean_temp = point['mean_temperature']
                row = f'{bot_id},{bottom_dive},{bottom_type},{depth_achieved},{dive_rate},{duration_to_acquire_gps},{max_acc},{mean_depth},{mean_salinity},{mean_temp},{powered_rise_rate},{unpowered_rise_rate},{start_lat},{start_lon},{drift_duration},{drift_end_lat},{drift_end_lon},{drift_start_lat},{drift_start_lon},{estimated_drift},{end_time},{start_time}'
                file.write(row)
                file.write('\n')
        file.close()
        

logs = get_logs()
scrape_raw_data(logs)