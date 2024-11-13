# Generates random time series of length 10.000 with contamination of different anomaly types
import numpy as np
import pandas as pd
from copy import deepcopy
import random
from ruamel.yaml import YAML
from gutenTAG import GutenTAG, TrainingType, LABEL_COLUMN_NAME
from tqdm import tqdm

# Define general parameters for time series
timeseries_template = {
    'length': 1000,  # Length of entire time series
    'semi-supervised': False,  # Whether a train file without anomalies should be generated
    'supervised': True,  # Whether the train file should contain labels
}


# Define general anomaly properties wit all possibilities
anomaly_template = {
    'kinds': ['amplitude','extremum','frequency','mean','pattern','pattern-shift','platform','trend','variance'],
    'position': ['beginning','middle','end'],
}

anomaly_template_2 = {
    'kinds': ['amplitude','extremum','frequency','mean','pattern','platform','trend','variance'],
    'position': ['beginning','middle','end'],
}

anomaly_length = {
    'amplitude': [100],
    'extremum': [1],
    'frequency': range(50, 100, 10),
    'mean': range(50, 100, 10),
    'pattern': range(50, 100, 10),
    'pattern-shift': range(50, 100, 10),
    'platform': range(50, 100, 10),
    'trend': range(50, 1000, 10),
    'variance': range(50, 100, 10)
}


# Define all anomaly types with example (ref. https://github.com/TimeEval/GutenTAG/blob/main/doc/introduction/anomaly-types.md)
def get_anomaly_type():
    anomaly_types = {
        'amplitude': {
            'amplitude_factor': np.arange(0.5, 2, 0.1, dtype=np.float32)},
        'extremum': {
            'min': [True, False],
            'local': [True, False]},
        'frequency': {'frequency_factor': np.arange(1.1, 2.5, 0.1, dtype=np.float32)},
        'mean': {'offset': np.arange(0.2, 1, 0.1, dtype=np.float32)},
        'pattern': {
            'sine': {'sinusoid_k': np.arange(5.0, 10.0, 0.2, dtype=np.float32)},  # Ramming factor for changing sine waves.
            'cosine': {'sinusoid_k': np.arange(5.0, 10.0, 0.2, dtype=np.float32)},
            'square': {'square_duty': np.arange(0.2, 0.7, 0.1, dtype=np.float32)},  # New duty of the square wave.
            'sawtooth': {'sawtooth_width': np.arange(0.7, 1.0, 0.1, dtype=np.float32)},  # New width
            'cylinder_bell_funnel': {'cbf_pattern_factor': np.arange(1.4, 2.2, 0.1, dtype=np.float32)}  # Pattern variance factor for change in CBF wave.
        },
        'pattern-shift': {
            'shift_by': [5],  # Size of the shift length to the right. Can be negative for shift to the left.
            'transition_window': [5],  # Number of points to the left and right used for transition.
        },
        'platform': {'value': np.arange(-0.3, 0.3, 0.1, dtype=np.float32)},  # Value of the platform on Y-axis
        'trend': generate_random_base_oscillation(),  # any form of the base oscillations
        'variance': {'variance': np.arange(0.1, 0.5, 0.1, dtype=np.float32)}  # Value of the new variance
    }

    return anomaly_types



def get_base_oscillation_template():
    # Define allowed kinds for base oscillations with corresponding parameters based on GutenTAG documentation
    base_oscillations_template = {
        'variance': np.arange(0, 0.5, 0.01, dtype=np.float32),  # Noise factor dependent on amplitude
        'offset': [0],  # Gets added to the generated time series
        'kind': list(base_oscillations_types.keys()),
    }
    return deepcopy(base_oscillations_template)


base_oscillations_types = {
    'sine': {
        'frequency': np.arange(1, 15, 0.2, dtype=np.float32),  # Number of square waves per 100 points
        'amplitude': np.arange(0.5, 2.5, 0.5, dtype=np.float32),  # +/- deviation from 0
        'freq-mod': np.arange(0.01, 0.5, 0.01, dtype=np.float32)  # Factor (of base frequency) of the frequency modulation that changes the amplitude of the wave over time. The carrier wave always has an amplitude of 1.
    },
    'cosine': {
        'frequency': np.arange(1, 15, 0.2, dtype=np.float32),
        'amplitude': np.arange(0.5, 2.5, 0.5, dtype=np.float32),
        'freq-mod': np.arange(0.01, 0.5, 0.01, dtype=np.float32)
    },
    'square': {
        'frequency': np.arange(1, 15, 0.2, dtype=np.float32),
        'amplitude': np.arange(0.5, 2.5, 0.5, dtype=np.float32),
        'freq-mod': np.arange(0.01, 0.5, 0.01, dtype=np.float32),
        'duty': range(2, 5, 1)
    },
    'polynomial': {
        'polynomial': [
        [1, 1, -8, -4, 1],
        [-8, 8, 2, -10],
        [8, 4, 2, 8]]
    }
}


# Function to generate random base oscillations
def generate_random_base_oscillation(is_main_oscilation=True):
    base_oscillation_template = get_base_oscillation_template()

    base_oscillations = {k: random.choice(p) for k, p in base_oscillation_template.items()}

    for k, p in base_oscillations_types[base_oscillations['kind']].items():
        base_oscillations[k] = random.choice(p)

    # Randomly add optional trend to base oscillation with probability
    if random.random() < 0.1 and is_main_oscilation:
        base_oscillations["trend"] = generate_random_base_oscillation(is_main_oscilation=False)

    return base_oscillations


def shuffel_anomaly(oscillation_type):
    random_anomaly = {k: random.choice(p) for k, p in deepcopy(anomaly_template).items()}

    if (oscillation_type == 'polynomial') and (random_anomaly['kinds'] in ['amplitude', 'frequency', 'pattern', 'pattern-shift']):
        # Shuffle again if incompatible type (ref. https://github.com/TimeEval/GutenTAG/blob/main/doc/introduction/index.md)
        return shuffel_anomaly(oscillation_type)

    return random_anomaly

def generate_random_anomaly(base_osc):
    anomaly = shuffel_anomaly(base_osc[0]['kind'])

    kind = anomaly['kinds']

    anomaly_types = get_anomaly_type()

    anomaly_characteristics = {}
    for k, p in anomaly_types[kind].items():
        if kind == 'trend':
            sub = {k: p}
            sub['oscillation'] = generate_random_base_oscillation(is_main_oscilation=False)
        elif kind == 'pattern':
            sub = {list(p.keys())[0]: random.choice(list(p.values())[0])}
        else:
            sub = {k: random.choice(p)}
        anomaly_characteristics.update(sub)

    anomaly_characteristics['kind'] = kind
    anomaly['kinds'] = [anomaly_characteristics]

    if kind == 'extremum':
        anomaly['length'] = 1
        if anomaly['kinds'][0]['local']:
            anomaly['kinds'][0]['context_window'] = random.choice([20, 30, 50])
    else:
        anomaly['length'] = random.choice([10, 50, 100])
    return anomaly


def generate_configurations(num_series):
    anomaly_probability = 1.1  # Probability of generating a time series with an anomaly: e.g. 0.2 for 20% chance of anomaly

    all_timeseries = []
    for i in tqdm(range(num_series), desc='Generating time series configs'):
        timeseries = deepcopy(timeseries_template)

        timeseries['base-oscillations'] = [generate_random_base_oscillation()]
        
        # Decide if this series will have anomalies based on probability
        if random.random() < anomaly_probability:
            timeseries['anomalies'] = [generate_random_anomaly(timeseries['base-oscillations'])]
        else:
            timeseries['anomalies'] = []
        timeseries['name'] = f'ts_{i}'
        all_timeseries.append(timeseries)

    return {'timeseries': all_timeseries}


def save_to_yaml(data, output_path='generated_config.yaml'):
    yaml = YAML()

    def represent_float32(representer, data):
        return representer.represent_float(float(data))

    
    yaml.representer.add_representer(np.float32, represent_float32)

    with open(output_path, 'w') as outfile:
        yaml.default_flow_style = False
        yaml.dump(data, outfile)

    print(f"Configuration file saved to {output_path}")


if __name__ == "__main__":
    # get_template("C:/Users/HUD2FH/OneDrive - Bosch Group/01_Useful_Libs/GutenTAG/synthetic_dataset/GutenTAG/overview.yaml")
    
    # ----- Generate Configuration File -----

    configuration = generate_configurations(10000)  # Generate x configurations/ time series
    save_to_yaml(configuration)


    # ----- Generate Time Series -----
    gutentag = GutenTAG()
    gutentag.load_config_dict(configuration)
    gutentag.generate(return_timeseries=False, output_folder="datasets/self_generated")
