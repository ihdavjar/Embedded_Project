import json
import argparse
import numpy as np


def get_weights_base_from_json(data):
    
    output = {}
    
    for key in data.keys():

        if ("input" not in key):
            splitted_key = key.split("_")
            layer_name = splitted_key[1]

            if (len(splitted_key)>2):

                if (len(splitted_key)<=3):

                    if (splitted_key[2].isdigit()):

                        layer_name = layer_name + "_" + str(int(splitted_key[2])+1)
        
                        
                        if (("bias:0" not in data[key].keys()) and ("kernel:0" in data[key].keys())):
                            weight = data[key]["kernel:0"]

                            output[layer_name] = {"bias": list(np.zeros(np.array(weight).shape)),"weights":data[key]["kernel:0"]}
                        

                        elif ("bias:0" in data[key].keys() and "kernel:0" in data[key].keys()):
                            output[layer_name] = {"bias":data[key]["bias:0"], "weights":data[key]["kernel:0"]}

                    
            else:
                layer_name = layer_name + "_1"
                
                if (("bias:0" not in data[key].keys()) and ("kernel:0" in data[key].keys())):
                    weight = data[key]["kernel:0"]

                    output[layer_name] = {"bias": list(np.zeros(np.array(weight).shape)),"weights":data[key]["kernel:0"]}
                    
                        

                elif ("bias:0" in data[key].keys() and "kernel:0" in data[key].keys()):
                    output[layer_name] = {"bias":data[key]["bias:0"], "weights":data[key]["kernel:0"]}

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get weights and bias from json file')
    parser.add_argument('json_file', type=str, help='path to json file')
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)
    
    output = get_weights_base_from_json(data)
    
    path = args.json_file

    print(path.rsplit('/',1)[-1].rsplit('.',1)[0]+'.h')

    f = open(path.rsplit('/',1)[-1].rsplit('.',1)[0]+'.h', "w")

    f.write("#include <stdint.h>\n")
    f.write("using namespace std;")

    for key in output.keys():
        weights = output[key]["weights"]
        bias = output[key]["bias"]

        weights_shape = np.array(weights).shape
        bias_shape = np.array(bias).shape

        #converting weights shape to a string
        weights_shape_str = str(weights_shape).replace("(","").replace(")","").replace(", ","*")


        weights = np.array(weights).reshape(-1)
        bias = np.array(bias).reshape(-1)   

        f.write("\n\nstatic const double " + key + "_weights[" + weights_shape_str + "] = {")
        for i in range(len(weights)):
            f.write(str(weights[i].round(5))+ ", ")
        f.write("};")

        f.write("\n\nstatic const double " + key + "_bias[" + str(len(bias)) + "] = {")
        for i in range(len(bias)):
            f.write(str(bias[i].round(5)) + ", ")
        f.write("};")

    f.close()

