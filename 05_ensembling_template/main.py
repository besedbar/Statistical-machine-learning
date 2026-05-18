#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tree import RegressionTree
from ensembles import GradientBoostedTrees, RandomForest
from data import generate_sin_data, generate_boston_housing, Dataset
from utils import generate_plot, rmse
from utils import read_input, save_data_to_buffer, load_data_from_buffer

parser = argparse.ArgumentParser(description="Process input and output file paths.")
parser.add_argument('input_file_path', type=str, help='Path to the input file')
parser.add_argument('output_file_path', type=str, nargs='?', default=None, help='Path to the output file')
# These arguments will be set appropriately by Brute, even if you change them.
parser.add_argument('--brute', action='store_true', help='Evaluation in Brute')

def experiment_tree_sin(show: bool = True) -> None:
    """
    Conducts an experiment with regression trees on a sine wave dataset.

    Parameters
    ----------
    show : bool, default=True
        Whether to display the plot.
    """
    data_sin_train, _ = generate_sin_data(n=100, scale=0.2)
    data_sin_test, sin_test_rmse = generate_sin_data(n=1000, scale=0.2)
    rng = np.random.RandomState(1)
    generate_plot(
        data_sin_train,
        data_sin_test,
        tattr="t",
        model_cls=RegressionTree,
        iterate_over="max_depth",
        iterate_values=list(range(30)),
        title="Regression Tree (sin)",
        xlabel="max depth",
        bayes_rmse=sin_test_rmse,
        rng=rng,
    )
    plt.savefig("regression_tree_sin.pdf")
    if show:
        plt.show()


def experiment_tree_housing(show: bool = False) -> None:
    """
    Conducts an experiment with regression trees on the Boston housing dataset.

    Parameters
    ----------
    show : bool, default=False
        Whether to display the plot.
    """
    data_housing_train, data_housing_test = generate_boston_housing()
    rng = np.random.RandomState(1)
    generate_plot(
        data_housing_train,
        data_housing_test,
        tattr="medv",
        model_cls=RegressionTree,
        iterate_over="max_depth",
        iterate_values=list(range(30)),
        title="Regression Tree (housing)",
        xlabel="max depth",
        rng=rng,
    )
    plt.savefig("regression_tree_housing.pdf")
    if show:
        plt.show()
        
def experiment_rf_housing(show: bool = True) -> None:
    """
    Conducts an experiment with Random Forests on the Boston housing dataset, varying the number of trees.

    Parameters
    ----------
    show : bool, default=True
        Whether to display the plot.
    """
    data_housing_train, data_housing_test = generate_boston_housing()
    rng = np.random.RandomState(1)
    n_trees_list = [1, 2, 5, 10, 20, 50, 100]

    generate_plot(
        data_housing_train,
        data_housing_test,
        tattr="medv",
        model_cls=RandomForest,
        iterate_over="n_trees",
        iterate_values=n_trees_list,
        title="RF (housing)",
        xlabel="# trees",
        rng=rng,
        max_depth=np.infty,
    )
    plt.ylim([0.0, 10.0])
    plt.savefig("rf_housing.pdf")
    if show:
        plt.show()
        

def experiment_gbm_housing(show: bool = False, beta: float = 0.1) -> None:
    """
    Conducts an experiment with Gradient Boosted Trees (GBM) on the Boston housing dataset.

    Parameters
    ----------
    show : bool, default=False
        Whether to display the plot.
    beta : float, default=0.1
        The learning rate for the Gradient Boosted Trees model.
    """
    data_housing_train, data_housing_test = generate_boston_housing()
    rng = np.random.RandomState(1)
    n_trees_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    generate_plot(
        data_housing_train,
        data_housing_test,
        tattr="medv",
        model_cls=GradientBoostedTrees,
        iterate_over="n_trees",
        iterate_values=n_trees_list,
        title=f"GBM (housing) beta = {beta}",
        xlabel="# trees",
        rng=rng,
        max_depth=1,
        beta=beta,
    )
    plt.ylim([0.0, 10.0])
    plt.savefig(f"gbm_housing_beta{beta}.pdf")
    if show:
        plt.show()

def test_gbm(args):
    # Read the input as a dictionary
    data = read_input(args.input_file_path)
    # Load the training and test data
    train_array, test_array = data['train_array'], data['test_array']
    data_housing_train = Dataset(pd.DataFrame(train_array, columns=data['array_columns']))
    data_housing_test = Dataset(pd.DataFrame(test_array, columns=data['array_columns']))
    
    # Build the GBM
    # By fixing the random number generator, the output of the training should be deterministic
    gbm = GradientBoostedTrees(data_housing_train, 
                         tattr=data['tattr'],
                         xattrs=None,
                         n_trees=data['n_trees'],
                         max_depth=1,
                         beta=data['beta'],
                         rng=np.random.RandomState(data['rng']),
    )
    # Compute outputs of the student implemented methods and evaluate the root-mean-square-error
    train_rmse = rmse(gbm, data_housing_train)
    test_rmse = rmse(gbm, data_housing_test)
    student_output = {'train_rmse': train_rmse, 'test_rmse': test_rmse}
     
    # Evaluate on public instances
    if not args.brute:
        print("\nGradient Boosted Machine:")
        reference_output =  load_data_from_buffer(args.input_file_path.replace('instances', 'solutions').replace('.json', ''))['gbm']
        
        for name in reference_output.keys():
            are_identical = np.allclose(student_output[name], reference_output[name], rtol=1e-05, atol=1e-05)

            if are_identical:
                print(f"\t{name}: Test OK")
            else:
                print(f"\t{name}: Test Failed")    
            
    return student_output

def test_rf(args):
    # Read the input as a dictionary
    data = read_input(args.input_file_path)
    # Load the training and test data
    train_array, test_array = data['train_array'], data['test_array']
    data_housing_train = Dataset(pd.DataFrame(train_array, columns=data['array_columns']))
    data_housing_test = Dataset(pd.DataFrame(test_array, columns=data['array_columns']))
    
    # Build the Random Forest
    # By fixing the random number generator, the output of the training should be deterministic
    rf = RandomForest(data_housing_train, 
                         tattr=data['tattr'],
                         xattrs=None,
                         n_trees=data['n_trees'],
                         max_depth=data['max_depth'],
                         rng=np.random.RandomState(data['rng']),
    )
    # Compute outputs of the student implemented methods and evaluate the root-mean-square-error
    train_rmse = rmse(rf, data_housing_train)
    test_rmse = rmse(rf, data_housing_test)
    student_output = {'train_rmse': train_rmse, 'test_rmse': test_rmse}
     
    # Evaluate on public instances
    if not args.brute:
        print("\nRandom Forest:")
        reference_output =  load_data_from_buffer(args.input_file_path.replace('instances', 'solutions').replace('.json', ''))['rf']
        
        for name in reference_output.keys():
            are_identical = np.allclose(student_output[name], reference_output[name], rtol=1e-05, atol=1e-05)

            if are_identical:
                print(f"\t{name}: Test OK")
            else:
                print(f"\t{name}: Test Failed")    
            
    return student_output    
    
def main(args):
    if not args.brute:
        # TODO: If you are interested, you can uncomment the following experiments, modify their parameters and observe the performance changes
        #experiment_tree_sin(show=True)
        #experiment_tree_housing(show=True)
        #experiment_rf_housing(show=True)
        #experiment_gbm_housing(show=True)
        pass

    gbm_output = test_gbm(args)
    rf_output = test_rf(args)

    if args.brute:
        save_data_to_buffer({'gbm': gbm_output,
                            'rf': rf_output}, 
                            args.output_file_path)
    
if __name__ == "__main__":
    args = parser.parse_args()    
    main(args)