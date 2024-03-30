import h5py
import json
import numpy as np
import argparse
    

def load_weights(path):
    f = h5py.File(path, 'r')
    weights = {}

    for layer in f['model_weights']:
        weights[layer] = {}

        if (len(np.array(f['model_weights'][layer])) == 0):
            continue
        
        else:
            for param in f['model_weights'][layer][layer]:
                if (np.array(f['model_weights'][layer][layer][param]).shape == ()):

                    weights[layer][param] = float(np.array(f['model_weights'][layer][layer][param])[()])
                
                else:
                    weights[layer][param] = np.array(f['model_weights'][layer][layer][param]).astype(float).tolist()
                
    return weights

if __name__ == '__main__':

    arg = argparse.ArgumentParser()
    arg.add_argument('-p', '--path', help='Path to the h5 file', required=True)

    args = arg.parse_args()

    path = arg.parse_args().path
    
    weights = load_weights(path)   

    # Saving the weights in a json file
    output_path = path.rsplit('/', 1)[-1].split('.')[0] + '.json'
    
    with open(output_path, 'w') as f:
        json.dump(weights, f, indent=4)
    