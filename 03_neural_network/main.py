#!/usr/bin/env python3
import argparse
import numpy as np
from typing import Tuple
from utils import read_input, save_data_to_buffer, load_data_from_buffer
from linear_layer import LinearLayer
from relu_layer import ReLULayer
from softmax_layer import SoftmaxLayer
from losses import LossCrossEntropy, LossCrossEntropyForSoftmaxLogits
from mlp import MLP

parser = argparse.ArgumentParser(description="Process input and output file paths.")
parser.add_argument('input_file_path', type=str, help='Path to the input file')
parser.add_argument('output_file_path', type=str, nargs='?', default=None, help='Path to the output file')
# These arguments will be set appropriately by Brute, even if you change them.
parser.add_argument('--brute', action='store_true', help='Evaluation in Brute')

def test_linear(args):
    # Read the input data as a dictionary
    data = read_input(args.input_file_path)['linear'] 
    # Build the layer
    layer = LinearLayer(data['n_inputs'], data['n_units'], np.random.RandomState(data['rng']), data['name'])
    # Compute outputs of the student implemented methods
    forward_out = layer.forward(**data['forward'])
    delta_out = layer.delta(**data['delta'])
    grad_out = layer.grad(**data['grad'])
    student_output = {'forward': forward_out, 'delta': delta_out, 'grad_W': grad_out[0], 'grad_b': grad_out[1]}
     
    # Evaluate on public instances
    if not args.brute:
        print("\nLinear:")
        reference_output =  load_data_from_buffer(args.input_file_path.replace('instances', 'solutions').replace('.json', ''))['linear']
         
        for name in reference_output.keys():
            are_identical = np.allclose(student_output[name], reference_output[name], rtol=1e-05, atol=1e-05)

            if are_identical:
                print(f"\t{name}: Test OK")
            else:
                print(f"\t{name}: Test Failed")    
            
    return student_output

def test_relu(args):
    # Read the input data as a dictionary
    data = read_input(args.input_file_path)['relu'] 
    # Build the layer
    layer = ReLULayer(data['name'])
    # Compute outputs of the student implemented methods
    forward_out = layer.forward(**data['forward'])
    delta_out = layer.delta(**data['delta'])
    student_output = {'forward': forward_out, 'delta': delta_out}
     
    # Evaluate on public instances
    if not args.brute:
        print("\nReLU:")
        reference_output =  load_data_from_buffer(args.input_file_path.replace('instances', 'solutions').replace('.json', ''))['relu']
        
        for name in reference_output.keys():
            are_identical = np.allclose(student_output[name], reference_output[name], rtol=1e-05, atol=1e-05)

            if are_identical:
                print(f"\t{name}: Test OK")
            else:
                print(f"\t{name}: Test Failed")    
            
    return student_output

def test_softmax(args):
    # Read the input data as a dictionary
    data = read_input(args.input_file_path)['softmax'] 
    # Build the layer
    layer = SoftmaxLayer(data['name'])
    # Compute outputs of the student implemented methods
    forward_out = layer.forward(**data['forward'])
    delta_out = layer.delta(**data['delta'])
    student_output = {'forward': forward_out, 'delta': delta_out}
     
    # Evaluate on public instances
    if not args.brute:
        print("\nSoftmax:")
        reference_output =  load_data_from_buffer(args.input_file_path.replace('instances', 'solutions').replace('.json', ''))['softmax']
        
        for name in reference_output.keys():
            are_identical = np.allclose(student_output[name], reference_output[name], rtol=1e-05, atol=1e-05)
            
            if are_identical:
                print(f"\t{name}: Test OK")
            else:
                print(f"\t{name}: Test Failed")    
            
    return student_output

def test_cross_entropy(args):
    # Read the input data as a dictionary
    data = read_input(args.input_file_path)['cross_entropy'] 
    # Build the layer
    layer = LossCrossEntropy(data['name'])
    # Compute outputs of the student implemented methods
    forward_out = layer.forward(**data['forward'])
    delta_out = layer.delta(**data['delta'])
    student_output = {'forward': forward_out, 'delta': delta_out}
     
    # Evaluate on public instances
    if not args.brute:
        print("\nCross Entropy:")
        reference_output =  load_data_from_buffer(args.input_file_path.replace('instances', 'solutions').replace('.json', ''))['cross_entropy']
        
        for name in reference_output.keys():
            are_identical = np.allclose(student_output[name], reference_output[name], rtol=1e-05, atol=1e-05)

            if are_identical:
                print(f"\t{name}: Test OK")
            else:
                print(f"\t{name}: Test Failed")    
            
    return student_output

def test_cross_entropy_for_logits(args):
    # Read the input data as a dictionary
    data = read_input(args.input_file_path)['cross_entropy_for_logits'] 
    # Build the layer
    layer = LossCrossEntropyForSoftmaxLogits(data['name'])
    # Compute outputs of the student implemented methods
    forward_out = layer.forward(**data['forward'])
    delta_out = layer.delta(**data['delta'])
    student_output = {'forward': forward_out, 'delta': delta_out}
     
    # Evaluate on public instances
    if not args.brute:
        print("\nCross Entropy For Softmax Logits:")
        reference_output =  load_data_from_buffer(args.input_file_path.replace('instances', 'solutions').replace('.json', ''))['cross_entropy_for_logits']
        
        for name in reference_output.keys():
            are_identical = np.allclose(student_output[name], reference_output[name], rtol=1e-05, atol=1e-05)

            if are_identical:
                print(f"\t{name}: Test OK")
            else:
                print(f"\t{name}: Test Failed")    
            
    return student_output


def test_MLP(args):
    # Read the input data as a dictionary
    data = read_input(args.input_file_path)['mlp'] 
    # Build the layer
    net = MLP(n_inputs=data['n_inputs'],
              layers=[
                  LinearLayer(n_inputs=data['n_inputs'], n_units=data['n_units'],
                              rng=np.random.RandomState(data['rng']), name='Linear_1'),
                  ReLULayer(name='ReLU_1'),
                  LinearLayer(n_inputs=data['n_units'], n_units=data['n_units'],
                              rng=np.random.RandomState(data['rng']), name='Linear_2'),
                  ReLULayer(name='ReLU_2'),
                  LinearLayer(n_inputs=data['n_units'], n_units=10,
                              rng=np.random.RandomState(data['rng']), name='Linear_3'),
                  SoftmaxLayer(name='Softmax_OUT')
              ],
              loss=LossCrossEntropy(name='CE')
              )
    # Compute outputs of the student implemented methods
    gradient_out = net.gradient(**data['gradient'])
    student_output = {'gradient': gradient_out}
     
    # Evaluate on public instances
    if not args.brute:
        print("\nMLP:")
        reference_output =  load_data_from_buffer(args.input_file_path.replace('instances', 'solutions').replace('.json', ''))['mlp']
        
        for name in reference_output['gradient'].keys():
            are_identical = all([np.allclose(student_output['gradient'][name][i], reference_output['gradient'][name][i], rtol=1e-05, atol=1e-05) for i in range(len(student_output['gradient'][name]))])

            if are_identical:
                print(f"\t{name} gradient: Test OK")
            else:
                print(f"\t{name} gradient: Test Failed")    
            
    return student_output



def main(args):
    linear_out = test_linear(args)
    relu_out = test_relu(args)
    softmax_out = test_softmax(args)
    cross_entropy_out = test_cross_entropy(args)
    cross_entropy_for_logits_out = test_cross_entropy_for_logits(args)
    mlp_out = test_MLP(args)
    
    if args.brute:
        save_data_to_buffer({'linear': linear_out,
                             'relu': relu_out,
                             'softmax': softmax_out,
                             'cross_entropy': cross_entropy_out,
                             'cross_entropy_for_logits': cross_entropy_for_logits_out,
                             'mlp': mlp_out}, 
                            args.output_file_path)
    
if __name__ == "__main__":            
    args = parser.parse_args()    
    main(args)
