import numpy as np
import matplotlib.pyplot as plt
from helloNet.metrics import compute_accuracy_binary


def initialize_parameters(dims_all_layers, num_features, initialization="he"):

    # np.random.seed(1)  # debug

    # Print Information
    # print(f"{'-'*50}\nInitialization Type: '{initialization}'")
    print(f"\nInitialization Method:\t '{initialization}'")

    # Add input features dimension to dimensions list
    dims_all_layers.insert(0, num_features)

    # Get length of layers
    len_layers = len(dims_all_layers)

    # Create empty dictionary to save parameters
    parameters_initialized = {}

    # Iterate over number of layers
    for l in range(1, len_layers):

        # Select initialization method to modify coefficient
        if initialization == "random":
            method_coef = 0.01

        elif initialization == "xavier":
            method_coef = np.sqrt(1.0 / dims_all_layers[l - 1])

        elif initialization == "he":
            method_coef = np.sqrt(2.0 / dims_all_layers[l - 1])

        else:
            raise ValueError(
                "Undefined initialization method. Please check 'initialization' parameter."
            )

        # print("CHECK: layer coefficient ", method_coef)

        # Create weights matrix for current layer
        parameters_initialized["W" + str(l)] = (
            np.random.randn(dims_all_layers[l], dims_all_layers[l - 1]) * method_coef
        )  # 0.01 , # / np.sqrt(dims_all_layers[l - 1]
        # Create bias vector for current layer
        parameters_initialized["b" + str(l)] = np.zeros((dims_all_layers[l], 1))
        # Print information
        print(
            f"Created: W{l}, b{l}, Shapes: {parameters_initialized['W'+str(l)].shape}, {parameters_initialized['b'+str(l)].shape}"
        )

    # Return initialized model parameters
    return parameters_initialized


def forward_pass_linear(A_prev, W, b):

    # Calculate linear hypothesis
    Z = np.dot(W, A_prev) + b
    # Stack variables in a cache
    cache_linear = (A_prev, W, b)

    # Return variables
    return Z, cache_linear


def forward_pass_activation_for_one_layer(
    A_previous,
    W,
    b,
    type_activation,
    batch_normalization=False,
    gamma=None,
    norm_beta=None,
    param_bn=None,
    train=False,
    layer=None,
):

    # Control activation type (THIS BLOCK WILL BE DELETED SOON)
    if type_activation not in ["sigmoid", "relu"]:
        raise ValueError("Undefined activation type.")

    # Apply linear part of forward pass
    Z, cache_linear = forward_pass_linear(A_previous, W, b)  # classical linear part: Z

    #  Apply batch normalization to linear part Z if batch_normalization param. is True
    if batch_normalization:
        Z, cache_bn = apply_batch_normalization(  # batch normalized lineer part: Z
            Z, gamma, norm_beta, param_bn, train, layer
        )

    # print("TRAIN", train)
    # print("LAYER", layer)
    # print("LINEAR", Z)

    # Apply selected activation
    if type_activation == "sigmoid":
        # Apply sigmoid activation
        A, cache_activation = activate_with_sigmoid(Z)

    elif type_activation == "relu":
        # Apply RelU activation
        A, cache_activation = activate_with_relu(Z)

    elif type_activation == "tanh":
        A, cache_activation = activate_with_tanh(Z)
        pass

    elif type_activation == "leaky-relu":
        A, cache_activation = activate_with_leaky_relu()
        pass

    #  Stack cache values
    # cache_layer_combined = (cache_linear, cache_activation)

    # Combine forward propagation cache. Add batch normalization cache if batch_normalization param. is True
    if batch_normalization:
        cache_layer_combined = (cache_linear, cache_activation, cache_bn)
    else:
        cache_layer_combined = (cache_linear, cache_activation)

    # Return variables
    return A, cache_layer_combined


def initialize_batch_normalization(parameters_model):

    #  Get number of layers
    len_layers = len(parameters_model) // 2

    # Initialize batch norm. parameters
    param_bn = {}
    param_bn["bn_epsilon"] = 1e-5
    param_bn["bn_momentum"] = 0.9

    # Initialization info print
    print("\nInitialized batch normalization parameters.")
    print(
        f'bn_epsilon: {param_bn["bn_epsilon"]}, bn_momentum: {param_bn["bn_momentum"]}'
    )

    # For every layer
    for l in range(1, len_layers + 1):

        # Initialize a gamma parameter as ones vector which has same shape with bias param of the layer
        parameters_model["gamma" + str(l)] = np.ones_like(
            parameters_model["b" + str(l)]
        )

        # Initialize a beta parameter as zeros vector which has same shape with bias param of the layer
        parameters_model["beta" + str(l)] = np.zeros_like(
            parameters_model["b" + str(l)]
        )

        param_bn["ewma_mean" + str(l)] = np.zeros_like(parameters_model["b" + str(l)])
        param_bn["ewma_var" + str(l)] = np.zeros_like(parameters_model["b" + str(l)])

        # Print shape of new parameters
        print(
            f"Created: gamma{l}, beta{l}, Shapes: {parameters_model['gamma'+ str(l)].shape}, {parameters_model['beta'+str(l)].shape}, ewma_mean{l}, emwa_var{l}, Shapes: {param_bn['ewma_mean' + str(l)].shape}, {param_bn['ewma_var' + str(l)].shape}"
        )

    print("Parameter keys:", parameters_model.keys())

    return parameters_model, param_bn


def apply_batch_normalization(Z, gamma, norm_beta, param_bn, train=True, layer=None):

    moving_mean = param_bn["ewma_mean" + str(layer)]
    moving_var = param_bn["ewma_var" + str(layer)]
    norm_epsilon = param_bn["bn_epsilon"]
    norm_momentum = param_bn["bn_momentum"]

    if train:  # Forward Propagation for TRAIN time

        # Compute mean of mini-batch features
        feature_mean = np.mean(Z, axis=1, keepdims=1)
        # print(
        #     f"FEATURE MEAN {feature_mean.shape}---------------------------\n",
        #     feature_mean,
        # )

        # Compute variance of mini-batch features
        feature_var = np.var(Z, axis=1, keepdims=1)
        # print(
        #     f"FEATURE VARIANCE {feature_var.shape}------------------------\n",
        #     feature_var,
        # )

        # Compute epsilon-added standard deviation of mini-batch features
        feature_std = np.sqrt(feature_var + norm_epsilon)
        # print(
        #     f"FEATURE STD {feature_std.shape}-----------------------------\n",
        #     feature_std,
        # )

        # print(f"PURE Z {Z.shape}----------------------------------\n", Z)

        # Compute zero-mean (centered) linear part
        Z_zero_mean = Z - feature_mean
        # print(
        #     f"Z ZERO MEAN {Z_zero_mean.shape}----------------------------------\n",
        #     Z_zero_mean,
        # )

        # Compute standardized form of zero mean linear part
        Z_standardized = Z_zero_mean / feature_std
        # print(
        #     f"Z ZERO MEAN {Z_standardized.shape}----------------------------------\n",
        #     Z_standardized,
        # )

        # print("GAMMA", gamma, "BETA", norm_beta)

        # Scale and shift with batch normalization params gamma and beta
        Z_batch_normed = gamma * Z_standardized + norm_beta
        # print(
        #     f"Z BATCH NORMED {Z_batch_normed.shape}-------------------------------\n",
        #     Z_batch_normed,
        # )

        # Calculate new moving average and moving variance values
        moving_mean = norm_momentum * moving_mean + (1 - norm_momentum) * feature_mean
        moving_var = norm_momentum * moving_var + (1 - norm_momentum) * feature_var

        # Update moving average and moving variance
        param_bn["ewma_mean" + str(layer)] = moving_mean
        param_bn["ewma_var" + str(layer)] = moving_var

        # Stack intermediate variables in cache
        cache_bn = (
            Z_batch_normed,
            Z_standardized,
            Z_zero_mean,
            feature_std,
            gamma,
            norm_beta,
        )

    else:  # Forward Propagation for TEST time

        # Compute zero-mean (centered) linear part with moving averages
        Z_zero_mean = Z - moving_mean

        # Compute standart deviation of features with moving variance
        feature_std = np.sqrt(moving_var + norm_epsilon)

        # Comute standardized form of zero mean linear part
        Z_standardized = Z_zero_mean / feature_std

        # Scale and shift with batch normalization params gamma and beta
        Z_batch_normed = gamma * Z_standardized + norm_beta

        # No need to keep cache for test pass
        cache_bn = None

    # Return bath normalized linear part and batch norm. cache
    return Z_batch_normed, cache_bn


def activate_with_sigmoid(Z):

    # Sigmoid function
    A = 1 / (1 + np.exp(-Z))

    # Save Z as cache
    cache_activation = Z

    # Return variables
    return A, cache_activation


def activate_with_relu(Z):

    # ReLU function
    A = np.maximum(0, Z)

    # Save Z as cache
    cache_activation = Z

    # Return variables
    return A, cache_activation


def activate_with_tanh():
    pass


def activate_with_leaky_relu():
    pass


def forward_propagation(
    X,
    parameters_model,
    act_hidden_layers="relu",
    act_output_layer="sigmoid",
    batch_normalization=False,
    param_bn=None,
    train=True,
):

    # Create empty list for stacking of model parameters
    caches_model = []

    # Assign X as A0 (input activations)
    A = X

    # Get length of layers
    if batch_normalization:
        len_layers = len(parameters_model) // 4
    else:
        len_layers = len(parameters_model) // 2

    for l in range(1, len_layers + 1):

        A_prev_layer = A
        W_layer = parameters_model["W" + str(l)]
        b_layer = parameters_model["b" + str(l)]
        activation_layer = act_hidden_layers

        if batch_normalization:
            gamma_layer = parameters_model["gamma" + str(l)]
            beta_layer = parameters_model["beta" + str(l)]
        else:
            gamma_layer = None
            beta_layer = None

        if l == len_layers:
            activation_layer = act_output_layer

        A, cache_layer = forward_pass_activation_for_one_layer(
            A_prev_layer,
            W_layer,
            b_layer,
            activation_layer,
            batch_normalization,
            gamma_layer,
            beta_layer,
            param_bn,
            train,
            l,
        )
        caches_model.append(cache_layer)

    # Assign AL name to last layer activations
    A_output = A

    # assert A_output.shape == (1, X.shape[1])

    # Return variables
    return A_output, caches_model


def compute_cost(
    A_output,
    Y,
    classification,
    reg_type=None,
    parameters=None,
    lambda_val=None,
    batch_normalization=False,
):

    # Get number of examples
    m = Y.shape[1]

    # Compute the binary cross-entropy cost
    # cost = (-1 / m) * np.sum(Y * np.log(A_output) + (1 - Y) * np.log(1 - A_output))
    if classification == "binary":
        cost = -1 * np.sum(Y * np.log(A_output) + (1 - Y) * np.log(1 - A_output))

    # Adjust shape of cost
    cost = np.squeeze(cost)

    # Regularization Implementation
    cost = regularize_cost(
        cost, reg_type, parameters, lambda_val, m, batch_normalization
    )

    # Return variable
    return cost


def regularize_cost(
    cost, regularization_type, parameters_mdl, lambda_val, m, batch_normalization
):

    if regularization_type is not None:

        # Get number of layers
        if batch_normalization:
            len_layers = len(parameters_mdl) // 4
        else:
            len_layers = len(parameters_mdl) // 2

        # Define regularization accumulator
        reg_accumulator = 0

        # Compute "L2" regularization
        if regularization_type == "l2":

            # For every layer, get sum of weight matrix values of layer
            for l in range(1, len_layers + 1):
                reg_accumulator += np.sum(np.square(parameters_mdl["W" + str(l)]))

            # Collect regularization term
            l2_reg_term = (lambda_val / (2 * m)) * reg_accumulator

            # Modify cost with L2 regularization term
            cost_regularized = cost + l2_reg_term

        #  Compute "L1" regularization
        elif regularization_type == "l1":

            # For every layer, get sum of weight matrix values of layer
            for l in range(1, len_layers + 1):
                reg_accumulator += np.sum(np.abs(parameters_mdl["W" + str(l)]))

            # Collect regularization term
            l1_reg_term = (lambda_val / (2 * m)) * reg_accumulator

            # Modify cost with L1 regularization term
            cost_regularized = cost + l1_reg_term

        else:
            raise ValueError(
                "Undefined regularization type. Please check 'regularization' parameter."
            )

        # Return regularized cost value
        return cost_regularized

    # regularization = None
    else:
        # Return original cost value
        return cost


def backward_pass_linear(dZ_layer, cache_linear):

    # Unpack linear cache
    A_previous, W_layer, b_layer = cache_linear

    # Get number of examples
    m = A_previous.shape[1]

    # Compute gradients of previous layer activations
    dA_previous = np.dot(W_layer.T, dZ_layer)

    # Compute gradient of current layer's weights
    dW_layer = (1 / m) * np.dot(dZ_layer, A_previous.T)

    #  Compute gradient of current layer's bias vector
    db_layer = (1 / m) * np.sum(dZ_layer, axis=1, keepdims=True)

    # Return variables
    return dA_previous, dW_layer, db_layer


def backward_pass_activation_for_one_layer(
    dA_layer, cache_layer, type_activation, batch_normalization
):

    # Control activation type
    if type_activation not in ["sigmoid", "relu"]:
        raise ValueError("Undefined activation type.")

    # Unpack cache variables of current layer
    if batch_normalization:
        cache_linear, cache_activation, cache_bn = cache_layer
        Z_batch_normed = cache_bn[0]
        cache_activation = Z_batch_normed
    else:
        cache_linear, cache_activation = cache_layer

    # APPLY HERE: cache_activation: Z --> Z_bn from cache_bn APPLY on #HERE Tag

    # Control activation type
    if type_activation == "relu":
        # Apply "undo" for ReLU activation
        dZ_layer = deactivate_with_relu(dA_layer, cache_activation)

    elif type_activation == "sigmoid":
        # Apply "undo" for sigmoid activation
        dZ_layer = deactivate_with_sigmoid(dA_layer, cache_activation)

    elif type_activation == "tanh":
        # Apply "undo" for tanh activation
        dZ_layer = deactivate_with_tanh(dA_layer, cache_activation)

    elif type_activation == "leaky-relu":
        # Apply "undo" for leaky-relu activation
        dZ_layer = deactivate_with_leaky_relu(dA_layer, cache_activation)

    # APPLY HERE: replace dZ_layer --> dZ from gradient_for_batch_no.. function
    if batch_normalization:
        dZ_layer, dgamma, dbeta = gradients_for_batch_normalization(dZ_layer, cache_bn)
        # apply here

    # Calculate previous layer's activation using linear part of activation
    dA_previous, dW_layer, db_layer = backward_pass_linear(dZ_layer, cache_linear)

    # BN variables to export
    if batch_normalization:
        param_BN = dgamma, dbeta  # dZ_layer, cache_bn
    else:
        param_BN = None

    # Return the layer's gradient variables
    return dA_previous, dW_layer, db_layer, param_BN


def gradients_for_batch_normalization(dZ_batch_normed, cache):

    Z_batch_normed, Z_standardized, Z_zero_mean, feature_std, gamma, norm_beta = cache

    dbeta = np.sum(dZ_batch_normed, axis=1, keepdims=1)

    dgamma_z_std = dZ_batch_normed

    dgamma = np.sum((dgamma_z_std * Z_standardized), axis=1, keepdims=1)

    dZ_standardized = dgamma_z_std * gamma

    dInv_feature_std = np.sum((dZ_standardized * Z_zero_mean), axis=1, keepdims=1)

    dZ_zero_mean_1 = dZ_standardized / feature_std

    dfeature_std = dInv_feature_std * (-1 / (feature_std ** 2))

    dfeature_var = 0.5 * (1 / feature_std) * dfeature_std

    m = dZ_batch_normed.shape[1]

    dsquare = (np.ones_like(dZ_batch_normed) / m) * dfeature_var

    dZ_zero_mean_2 = 2 * Z_zero_mean * dsquare

    dZ_1 = dZ_zero_mean_1 + dZ_zero_mean_2

    dfeature_mean = -1 * np.sum((dZ_zero_mean_1 + dZ_zero_mean_2), axis=1, keepdims=1)

    dZ_2 = (np.ones_like(dZ_batch_normed) / m) * dfeature_mean

    dZ = dZ_1 + dZ_2

    return dZ, dgamma, dbeta


def deactivate_with_relu(dA, cache_act):

    # Get linear part
    Z = cache_act

    # Calculate gradient of pre-activation part
    dZ = np.array(dA, copy=True)

    # Relu reverse limit
    dZ[Z <= 0] = 0

    return dZ


def deactivate_with_sigmoid(dA, cache_act):

    # Get linear part
    Z = cache_act

    # Sigmoid func
    s = 1 / (1 + np.exp(-Z))

    # Calculate gradient of pre-activation part
    dZ = dA * s * (1 - s)

    return dZ


def deactivate_with_tanh():
    pass


def deactivate_with_leaky_relu():
    pass


def back_propagation(
    A_output,
    Y,
    caches_model,
    m_set=None,
    params_model=None,
    reg_type=None,
    lambda_val=0,
    act_hidden_layers="relu",
    act_output_layer="sigmoid",
    batch_normalization=False,
):  # activ_hid activ_out

    # Create empty dictionary for gradient stacking
    gradients_model = {}

    # Get number of layers
    len_layers = len(caches_model)

    # Reshape labels variable Y to explicit broadcasting
    Y = Y.reshape(A_output.shape)

    # Calculate gradient of output layer (with output activation type)
    dA_output = -(np.divide(Y, A_output) - np.divide((1 - Y), (1 - A_output)))
    temp_dA, temp_dW, temp_db, temp_BN = backward_pass_activation_for_one_layer(
        dA_output, caches_model[len_layers - 1], act_output_layer, batch_normalization,
    )

    gradients_model["dA" + str(len_layers - 1)] = temp_dA  # dA_L-1
    gradients_model["dW" + str(len_layers)] = temp_dW  # dW_L
    gradients_model["db" + str(len_layers)] = temp_db  # db_L

    if batch_normalization:
        dgamma, dbeta = temp_BN
        gradients_model["dgamma" + str(len_layers)] = dgamma
        gradients_model["dbeta" + str(len_layers)] = dbeta

    # Calculate gradient for hidden layers (with hidden activation type)
    for l in reversed(range(1, len_layers)):

        # Calculate gradients of l-th layer
        temp_dA_layer = gradients_model["dA" + str(l)]  # dA_(l)
        cache_layer = caches_model[l - 1]  # cache vals. of l-th layer
        temp_act = act_hidden_layers  # activation type for hidden layers

        (
            temp_dA_prev,
            temp_dW,
            temp_db,
            temp_BN,
        ) = backward_pass_activation_for_one_layer(
            temp_dA_layer, cache_layer, temp_act, batch_normalization
        )

        gradients_model[
            "dA" + str(l - 1)
        ] = temp_dA_prev  # act.gradient of prev.layer (dA_(l-1))
        gradients_model["dW" + str(l)] = temp_dW  # gradients of current layer
        gradients_model["db" + str(l)] = temp_db  # gradients of current layer

        if batch_normalization:
            dgamma, dbeta = temp_BN
            gradients_model["dgamma" + str(l)] = dgamma
            gradients_model["dbeta" + str(l)] = dbeta

    # Print key information
    # print("gradients variables:",gradients_model.keys())

    # Regularize gradients
    gradients_model = regularize_gradients(
        gradients_model, params_model, reg_type, lambda_val, m_set, len_layers
    )

    # Return gradients of model
    return gradients_model


def regularize_gradients(
    gradients_reg, parameters, regularization_type, lambda_val, m, len_layers
):
    # If regularization selected
    if regularization_type is not None:

        # Implementation of "L2" regularization for gradients
        if regularization_type == "l2":
            for l in range(1, len_layers + 1):
                gradients_reg["dW" + str(l)] += (lambda_val / m) * parameters[
                    "W" + str(l)
                ]

        else:
            raise ValueError(
                "Undefined regularization type. Please check 'regularization' parameter."
            )

        return gradients_reg

    else:
        return gradients_reg


def print_final_info(
    A_current_iter, labels_train, cost_current_iter, m_set_train, print_cost, t2, t1
):
    if print_cost:
        print("\n")

    print(f"Final training loss: {cost_current_iter:.6f}")
    accuracy_train = compute_accuracy_binary(A_current_iter, labels_train)

    print(
        f"Set size: {m_set_train}, Number of true predictions: {int(accuracy_train*m_set_train)}"
    )
    print(f"Model training accuracy: {accuracy_train:.4f}")
    print(f"The training took {(t2-t1):.1f} seconds.")


def plot_learning_curve_func(
    costs_model, costs_test, acc_model=None, acc_test=None, epochs=None, split=True
):
    print("** Plotting learning curve... **")

    plt.rcParams["figure.figsize"] = (8, 6)
    plt.style.use("seaborn-dark")

    if split:
        # Create subplot object
        fig, (axis1, axis2) = plt.subplots(
            1, 2, sharey=False, sharex=True, figsize=(16, 6)
        )

        # Configure axis1 values
        axis1.plot(epochs, np.squeeze(costs_model), label="Training Loss")
        axis1.plot(epochs, np.squeeze(costs_test), label="Validation Loss")

        # Configure axis1 names
        axis1.set_ylabel("model loss")
        axis1.set_xlabel("epochs")
        axis1.set_title("Model Loss")
        axis1.legend(loc="upper right")
        axis1.grid()

        # Configure axis2 values
        axis2.plot(epochs, np.squeeze(acc_model), label="Training Accuracy")
        axis2.plot(epochs, np.squeeze(acc_test), label="Validation Accuracy")
        axis2.set_ylabel("model accuracy")
        axis2.set_xlabel("epochs")
        axis2.set_title("Model Accuracy")
        axis2.legend(loc="upper right")
        axis2.tick_params(labelleft=True)
        axis2.grid()

    else:
        # Plot dimension-adjusted model_costs
        plt.plot(np.squeeze(costs_model), label="Training Loss")
        plt.plot(np.squeeze(costs_test), label="Validation Loss")
        plt.plot(np.squeeze(acc_model), label="Training Accuracy")
        plt.plot(np.squeeze(acc_test), label="Validation Accuracy")
        # Configure axis names
        plt.ylabel("model loss / accuracy")
        plt.xlabel("epochs")
        plt.title("Model Training Loss / Accuracy")
        plt.legend(loc="upper right")
        plt.grid()

    # Show plotting
    plt.show()
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    plt.style.use("classic")


def print_initial_info(
    dims_hiddens,
    dims_output,
    type_classification,
    learning_rate,
    method_regularization,
    val_lambda,
    size_mini_batch,
    num_mini_batches,
    method_optimizer,
    beta1_val,
    beta2_val,
):
    print(
        "** The training has started. **\n",
        f"Number of hidden layers: {len(dims_hiddens)}",
        f"Hidden layer dims.:\t {dims_hiddens}",
        f"Output layer dims.:\t {dims_output}",
        f'Classification type:\t "{type_classification}"',
        f"Learning rate (alpha):\t {learning_rate}",
        f'Regularization method:\t "{method_regularization}"',
        f"Reg. param. (lambda):\t {val_lambda}",
        f"Mini batch size:\t {size_mini_batch}",
        f"Number of mini batches:\t {num_mini_batches}",
        f'Optimizer:\t\t "{method_optimizer}"',
        sep="\n",
    )
    if method_optimizer == "gradient-descent":
        pass
    elif method_optimizer == "momentum":
        print(f"Optimization parameters: Beta1: {beta1_val}")
    elif method_optimizer == "rmsprop":
        print(f"Optimization parameters: Beta2: {beta2_val}")
    elif method_optimizer == "adam":
        print(f"Optimization parameters: Beta1: {beta1_val}, Beta2: {beta2_val}")


def get_random_mini_batches(x_set, y_set, size_mini_batch):

    # Get number of training examples
    m_train = x_set.shape[1]

    # Get number of mini batches and round it to floor
    num_mini_batches = int(np.floor(m_train / size_mini_batch))

    # Initialize mini bat
    stack_mini_batches = []

    if num_mini_batches > 1:

        # Order the set randomly
        indexes_random = np.random.permutation(m_train)
        x_set = x_set[:, indexes_random]
        y_set = y_set[:, indexes_random].reshape((1, m_train))

        # For every mini batches
        for t in range(num_mini_batches):

            # For any t-th mini batch except last mini batch
            if t != (num_mini_batches - 1):

                x_mini_batch = x_set[:, t * size_mini_batch : (t + 1) * size_mini_batch]
                y_mini_batch = y_set[:, t * size_mini_batch : (t + 1) * size_mini_batch]

            # For last mini batch (adds the residual examples)
            else:

                x_mini_batch = x_set[:, t * size_mini_batch :]
                y_mini_batch = y_set[:, t * size_mini_batch :]

            # Stack them in a list as a tuple-type element
            current_mini_batch = (x_mini_batch, y_mini_batch)
            stack_mini_batches.append(current_mini_batch)

            # Debug printing
            # print(f"iteration: {t}, lower lim:{t*mini_batch_size}, upper lim: {(t+1)*mini_batch_size}, x_mini shape: {x_mini_batch.shape}, y_mini shape: {y_mini_batch.shape}")

    elif num_mini_batches == 1:

        # Stack x_train and y_train as just one mini batch
        current_mini_batch = (x_set, y_set)
        stack_mini_batches.append(current_mini_batch)

    else:
        # Raise error for invalid input
        raise ValueError(
            "Invalid input. The mini batch size can not be greater than the number of training examples. Please check 'size_mini_batch' variable."
        )

    # Return the stack of mini batches
    return stack_mini_batches


def update_parameters(
    parameters_to_update,
    gradients_iter,
    learning_rate,
    optimizer,
    params_optimizer,
    beta1_val,
    beta2_val,
    t_val,
    epsilon,
    batch_normalization,
):

    # Get number of layers
    if batch_normalization:
        len_layers = len(parameters_to_update) // 4
    else:
        len_layers = len(parameters_to_update) // 2

    if optimizer == "adam":
        v_corrected = {}
        s_corrected = {}

    # Update weights and bias matrix for each layer
    for l in range(1, len_layers + 1):

        # Get the accelerator part of update equations for layer-l
        gradient_dW = gradients_iter["dW" + str(l)]
        gradient_db = gradients_iter["db" + str(l)]

        # If batch normalization is selected, get gradients of gamma and beta
        if batch_normalization:
            gradient_dgamma = gradients_iter["dgamma" + str(l)]
            gradient_dbeta = gradients_iter["dbeta" + str(l)]

        if optimizer == "gradient-descent":

            # Get gradients itself of the layer-l as accelerators
            accelerator_dW = gradient_dW
            accelerator_db = gradient_db

            if batch_normalization:
                accelerator_dgamma = gradient_dgamma
                accelerator_dbeta = gradient_dbeta

        elif optimizer == "momentum":

            # Get momentum-processed gradients of the layer-l as accelerators
            accelerator_dW, accelerator_db = get_accelerator_for_momentum(
                gradient_dW, gradient_db, beta1_val, params_optimizer, layer=l
            )

        elif optimizer == "rmsprop":

            # Get RMSprop-processed gradients of the layer-l as accelerators
            accelerator_dW, accelerator_db = get_accelerator_for_rmsprop(beta2_val)

        elif optimizer == "adam":

            # Get RMSprop-processed gradients of the layer-l as accelerators
            accelerator_dW, accelerator_db = get_accelerator_for_adam(
                gradient_dW,
                gradient_db,
                beta1_val,
                beta2_val,
                t_val,
                epsilon,
                params_optimizer,
                l,
                v_corrected,
                s_corrected,
            )

        # Update variables for l-th layer with gradient descent algorithm
        parameters_to_update["W" + str(l)] -= learning_rate * accelerator_dW
        parameters_to_update["b" + str(l)] -= learning_rate * accelerator_db

        if batch_normalization:
            parameters_to_update["gamma" + str(l)] -= learning_rate * accelerator_dgamma
            parameters_to_update["beta" + str(l)] -= learning_rate * accelerator_dbeta

    # Return updated model parameters
    return parameters_to_update


def initialize_optimizer(optimizer, parameters, batch_normalization):

    if batch_normalization:
        len_layers = len(parameters) // 4
    else:
        len_layers = len(parameters) // 2

    params_optimizer = {}

    if optimizer == "gradient-descent":
        pass

    elif optimizer == "momentum":
        # Initialize momentum parameter: "v"
        params_optimizer = initialize_momentum(parameters, len_layers)

    elif optimizer == "rmsprop":
        pass

    elif optimizer == "adam":
        params_optimizer = initialize_adam(parameters, len_layers)

    # Print keys of optimizer parameter dictionary
    if optimizer != "gradient-descent":
        print(f"Optimizer param keys:\t {', '.join(params_optimizer)}")

    return params_optimizer


def initialize_momentum(parameters, len_layers):

    # Get number of layers
    # len_layers = len(parameters) // 2

    # Initialize optimizer parameters dictionary
    params_optimizer = {}

    # Initialize momentum velocity
    v = {}

    # Create zeros matrices for every weight matrix with same dimensions
    for l in range(1, len_layers + 1):

        # Create and keep it in "v" dictionary
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    # Keep "v" in dictionary of optimizer parameters
    params_optimizer["v"] = v

    # Return the parameters dictionary
    return params_optimizer


def initialize_rmsprop():
    pass


def initialize_adam(parameters, len_layers):

    # Get number of layers
    # len_layers = len(parameters) // 2

    # Initialize optimizer parameters dictionary
    params_optimizer = {}

    # Initialize momentum parameter "velocity"
    v = {}

    # Initialize rmsprop parameter "squared velocity"
    s = {}

    for l in range(1, len_layers + 1):

        # Create momentum (average) tracker variables and keep them in "v" dictionary
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

        # Create squared average tracker variables and keep them in "s" dictionary
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    # Keep "v" and "s" in a dictionary of optimizer parameters
    params_optimizer["v"] = v
    params_optimizer["s"] = s

    print("v keys", v.keys())
    print("s keys", s.keys())

    # Return the parameters dictionary
    return params_optimizer


def get_accelerator_for_momentum(grad_dW, grad_db, beta, params_opt, layer):

    # Get momentum parameter v
    v = params_opt["v"]

    # Update momentum parameters
    v["dW" + str(layer)] = beta * v["dW" + str(layer)] + (1 - beta) * grad_dW
    v["db" + str(layer)] = beta * v["db" + str(layer)] + (1 - beta) * grad_db

    # Keep last parameters
    params_opt["v"] = v

    return v["dW" + str(layer)], v["db" + str(layer)]


def get_accelerator_for_rmsprop():
    pass


def get_accelerator_for_adam(
    grad_dW,
    grad_db,
    beta1,
    beta2,
    t_val,
    epsilon_val,
    params_opt,
    layer,
    v_corrected,
    s_corrected,
):

    # Get adam variables "v"  and "s"
    v = params_opt["v"]
    s = params_opt["s"]

    v["dW" + str(layer)] = beta1 * v["dW" + str(layer)] + (1 - beta1) * grad_dW
    v["db" + str(layer)] = beta1 * v["db" + str(layer)] + (1 - beta1) * grad_db

    v_corrected["dW" + str(layer)] = v["dW" + str(layer)] / (1 - beta1 ** (t_val))
    v_corrected["db" + str(layer)] = v["db" + str(layer)] / (1 - beta1 ** (t_val))

    s["dW" + str(layer)] = beta2 * s["dW" + str(layer)] + (1 - beta2) * np.square(
        grad_dW
    )
    s["db" + str(layer)] = beta2 * s["db" + str(layer)] + (1 - beta2) * np.square(
        grad_db
    )

    s_corrected["dW" + str(layer)] = s["dW" + str(layer)] / (1 - beta2 ** (t_val))
    s_corrected["db" + str(layer)] = s["db" + str(layer)] / (1 - beta2 ** (t_val))

    accelerator_dW_part = v_corrected["dW" + str(layer)] / (
        np.sqrt(s_corrected["dW" + str(layer)]) + epsilon_val
    )
    accelerator_db_part = v_corrected["db" + str(layer)] / (
        np.sqrt(s_corrected["db" + str(layer)]) + epsilon_val
    )

    return accelerator_dW_part, accelerator_db_part


def get_decay_rate(
    learning_rate, epochs, final_ratio, precision="epochs", batches=None, plot=True
):

    if precision == "epochs":
        iterator = epochs

    elif precision == "batches":
        iterator = epochs * batches

    final_learning_rate = learning_rate * final_ratio
    decay_rate = ((learning_rate / final_learning_rate) - 1) / iterator

    print(
        f"Final learning rate: {final_learning_rate}",
        f"Precision: {precision}",
        f"Number of learning rates: {iterator}",
        f"** Decay rate: {decay_rate} **",
        sep="\n",
    )

    if plot:
        lr_by_iters = [
            (learning_rate / (1 + decay_rate * iters)) for iters in range(iterator)
        ]
        plt.plot(lr_by_iters)
        plt.xlabel(precision)
        plt.ylabel("learning rates")
        plt.show()

    return decay_rate


def get_decayed_learning_rate(
    learning_rate0, decay_rate, precision, epoch, batch_index
):
    #  Iteration precision (iterator): current epochs
    if precision == "epochs":
        decay_iter = epoch

    #  Iteration precision (iterator): current epochs * current batch
    elif precision == "batches":
        decay_iter = epoch * batch_index

    #  Compute decayed learning rate for current iteration
    decayed_learning_rate = learning_rate0 / (1 + decay_rate * decay_iter)

    # Return decayed learning rate
    return decayed_learning_rate
