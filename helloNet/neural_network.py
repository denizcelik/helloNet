import numpy as np
import helloNet.hello_utils as hello_utils
from helloNet.metrics import compute_accuracy_binary
from time import time


def train_model(
    X_train,
    Y_train,
    X_test=None,
    Y_test=None,
    dims_hidden_layers=[10],
    dims_output_layer=1,
    learning_rate=0.001,
    num_epochs=100,
    size_mini_batch=64,
    initialization="he",
    regularization="l2",
    lambda_val=0.7,
    optimizer="gradient-descent",
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    activation_hidden_layer="relu",
    activation_output_layer="sigmoid",
    print_cost=True,
    plot_learning_curve=True,
    period_print=100,
    period_stack=1,
    threshold_cost=None,
    random_seed=None,
    batch_normalization=False,
    lr_decay=False,
    decay_rate=None,
    precision="epochs",
):

    # Classification information
    classification = "binary"
    if dims_output_layer > 1:
        classification = "multi-class"

    # Get start time
    t1 = time()

    # Shallow copy of the variable to prevent modifying by a function
    dims_hidden_layers_copy = dims_hidden_layers.copy()

    # Lambda correction
    if regularization is None:
        lambda_val = None

    # Random seed assignment
    if random_seed is not None:
        np.random.seed(random_seed)

    # Initialize adam counter to use in optimizer == "adam" case
    t_adam_counter = 0

    # Create model layers dimension list
    dims_hidden_layers_copy.append(dims_output_layer)
    dims_all_layers = dims_hidden_layers_copy

    # Get number of input features to use in parameter initializing
    num_input_feat = X_train.shape[0]

    # Get number of example in "training" set
    m_train_set = X_train.shape[1]

    # Get number of example in "validation" set
    m_test_set = Y_test.shape[1]

    # If mini batch size = -1, take full batch as one single mini batch
    if size_mini_batch == -1:
        size_mini_batch = m_train_set

    # Get number of mini batches and round it to floor
    num_mini_batches = int(np.floor(m_train_set / size_mini_batch))

    # Create empty list to stack cost values of both "training" and "validation" sets
    costs_model = []
    costs_test = []

    # Create empty list to stack accuracy values of both "training" and "validation" sets
    accuracy_model = []
    accuracy_test = []

    # Create empty list to stack epoch iterations
    list_epochs = []

    # Print informations
    hello_utils.print_initial_info(
        dims_hidden_layers_copy,
        dims_output_layer,
        classification,
        learning_rate,
        regularization,
        lambda_val,
        size_mini_batch,
        num_mini_batches,
        optimizer,
        beta1,
        beta2,
    )

    # Initialize model parameters
    parameters_model = hello_utils.initialize_parameters(
        dims_all_layers, num_input_feat, initialization
    )

    # If batch normalization is intended, initialize and add gamma and beta parameters to model parameters
    if batch_normalization:

        parameters_model, param_bn = hello_utils.initialize_batch_normalization(
            parameters_model
        )
    else:
        param_bn = None
        # add return for

    # Initialize optimizer parameters
    parameters_optimizer = hello_utils.initialize_optimizer(
        optimizer, parameters_model, batch_normalization
    )

    # If learning rate decay is intended
    if lr_decay:
        learning_rate0 = learning_rate

    # Information Formatting
    print("\n", end="")

    # TRAINING LOOP

    # Iterate over the epochs
    for epoch in range(0, num_epochs):

        # Shuffle training set randomly and get mini batches as X, Y pairs stack
        stack_mini_batches = hello_utils.get_random_mini_batches(
            X_train, Y_train, size_mini_batch
        )

        # Define an accumulator to keep total cost values of mini batches in one epoch
        cost_epoch_total_train = 0

        # Create variables to stack activations and corresponding labels
        A_epoch_stack = np.array([])
        Y_epoch_stack = np.array([])

        #  Iterate over the mini batches
        for t in range(0, num_mini_batches):

            # Pick the mini batch
            X_train_mini, Y_train_mini = stack_mini_batches[t]

            # Compute the model activations (training predictions) for current iteration with Forward Propagation
            A_output_iter, caches_model_iter = hello_utils.forward_propagation(
                X_train_mini,
                parameters_model,
                activation_hidden_layer,
                activation_output_layer,
                batch_normalization,
                param_bn,
                train=True,
            )

            # Compute cross-entropy cost for model
            # *In next two statements, It calculates the total cost for mini batches iterations
            cost_minibatch_total = hello_utils.compute_cost(
                A_output_iter,
                Y_train_mini,
                classification,
                regularization,
                parameters_model,
                lambda_val,
                batch_normalization,
            )

            # Add the cost to current epoch total cost accumulator
            cost_epoch_total_train += cost_minibatch_total

            # Compute model gradients with Back Propagation
            gradients_model_iter = hello_utils.back_propagation(
                A_output_iter,
                Y_train_mini,
                caches_model_iter,
                m_set=m_train_set,
                params_model=parameters_model,
                reg_type=regularization,
                lambda_val=lambda_val,
                act_hidden_layers=activation_hidden_layer,
                act_output_layer=activation_output_layer,
                batch_normalization=batch_normalization,
            )

            # If learning rate decay is intended
            if lr_decay:
                # Get decayed learning rate for current iteration
                learning_rate = hello_utils.get_decayed_learning_rate(
                    learning_rate0, decay_rate, precision, epoch, t
                )

            # Iteration counter for optimizer == "adam"
            if optimizer == "adam":
                t_adam_counter += 1

            # Update parameters with selected optimize method
            parameters_model = hello_utils.update_parameters(
                parameters_model,
                gradients_model_iter,
                learning_rate,
                optimizer,
                parameters_optimizer,
                beta1,
                beta2,
                t_adam_counter,
                epsilon,
                batch_normalization,
            )

            # Rectify dimensions
            A_minibatch_train_vec = np.squeeze(A_output_iter.T)
            Y_train_mini_vec = np.squeeze(Y_train_mini.T)

            # Stack epoch cost for information
            A_epoch_stack = np.concatenate((A_epoch_stack, A_minibatch_train_vec))
            Y_epoch_stack = np.concatenate((Y_epoch_stack, Y_train_mini_vec))

        # [VALIDATION SET PROCESS]: Compute the output activations of "validation" set (predictions) with Forward Propagation
        (A_test, caches_model_test,) = hello_utils.forward_propagation(
            X_test,
            parameters_model,
            activation_hidden_layer,
            activation_output_layer,
            batch_normalization,
            param_bn,
            train=False,
        )

        # [VALIDATION SET PROCESS]: Compute the latest cross-entropy cost for "validation" (or test) set
        cost_epoch_total_test = hello_utils.compute_cost(
            A_test, Y_test, classification, batch_normalization=batch_normalization
        )

        # Average cost of "validation" set for current epoch
        cost_epoch_test = cost_epoch_total_test / m_test_set

        # Average cost of "training" set for current epoch
        cost_epoch_average_train = cost_epoch_total_train / m_train_set

        #  Compute training and validation accuracy
        if classification == "binary":

            # Get all of epoch activations as column vector
            A_epoch_col = A_epoch_stack.reshape((1, A_epoch_stack.shape[0]))
            Y_epoch_col = Y_epoch_stack.reshape((1, Y_epoch_stack.shape[0]))

            accuracy_epoch_train = compute_accuracy_binary(A_epoch_col, Y_epoch_col)
            accuracy_epoch_test = compute_accuracy_binary(A_test, Y_test)

        elif classification == "multi-class":
            raise ValueError("Undefined branch")

        else:
            raise ValueError("Undefined branch")

        # Save cost and accuracy values on every "period_stack" epochs for plotting
        if epoch % period_stack == 0:
            costs_model.append(cost_epoch_average_train)
            costs_test.append(cost_epoch_test)
            accuracy_model.append(accuracy_epoch_train)
            accuracy_test.append(accuracy_epoch_test)
            list_epochs.append(epoch)

        # Print cost (loss) value for every "period_print" iteration
        if print_cost and epoch % period_print == 0:
            print(
                f"Current loss for epoch {epoch}/{num_epochs} is - {cost_epoch_average_train:.4f} -- train accuracy: {accuracy_epoch_train:.4f} -- validation accuracy: {accuracy_epoch_test:.4f}"
            )

        # Check threshold value for converging break if it's set
        if (threshold_cost is not None) and (cost_epoch_average_train < threshold_cost):
            break

    # TRAINING LOOP END

    # Turn the last epoch activations and labels stacks to numpy arrays
    A_epoch_stack = A_epoch_stack.reshape((1, A_epoch_stack.shape[0]))
    Y_epoch_stack = Y_epoch_stack.reshape((1, Y_epoch_stack.shape[0]))

    # Get finish time to write execution time
    t2 = time()

    # Print training accuracy
    hello_utils.print_final_info(
        A_epoch_stack,
        Y_epoch_stack,
        cost_epoch_average_train,
        m_train_set,
        print_cost,
        t2,
        t1,
    )

    # Plot learning curve
    if plot_learning_curve:
        hello_utils.plot_learning_curve_func(
            costs_model, costs_test, accuracy_model, accuracy_test, list_epochs
        )

    # Return final "trained" parameters of model
    return parameters_model, param_bn


def predict(set_X, labels_Y, params_model, batch_normalization=False, param_bn=None):

    # Get start time
    t1 = time()

    # Get number of example in input set
    m_set = set_X.shape[1]

    preds = np.zeros((1, m_set))

    # Get predictions with forward prop.
    activations, caches_pred = hello_utils.forward_propagation(
        set_X,
        params_model,
        batch_normalization=batch_normalization,
        param_bn=param_bn,
        train=False,
    )

    # activations as probabilities
    probabilities = activations

    # Probabilities -> Predictions (vectorized)
    probs_bools = probabilities[0] > 0.5
    ones_to_convert = np.ones(probabilities.shape[1])
    preds = probs_bools * ones_to_convert
    preds = preds.reshape(1, preds.shape[0])

    if labels_Y is not None:
        print(
            f"Set size: {m_set}, Number of true predictions: {str(np.sum(preds == labels_Y))}"
        )
        print(f"Model prediction accuracy: {(np.sum(preds == labels_Y) / m_set):.4f}")

    # Get finish time
    t2 = time()
    print(f"The prediction process took {(t2 - t1):.6f} seconds.")

    return preds, probabilities


def search_learning_rate():
    pass


def search_hidden_layers():
    pass
