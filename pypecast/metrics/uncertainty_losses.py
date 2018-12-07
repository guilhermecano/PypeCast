
c = 1 #The number of outputs we want to predict
m = 1 #The number of distributions we want to use in the mixture

def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max), 
                       axis=axis, keepdims=True))+x_max


def mean_log_Gaussian_like(y_true, parameters):
    """Mean Log Gaussian Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """
    components = K.reshape(parameters,[-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = K.softmax(K.clip(alpha,1e-8,1.))
    
    exponent = K.log(alpha) - .5 * float(c) * K.log(2 * np.pi) \
    - float(c) * K.log(sigma) \
    - K.sum((K.expand_dims(y_true,2) - mu)**2, axis=1)/(2*(sigma)**2)
    
    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res


def mean_log_LaPlace_like(y_true, parameters):
    """Mean Log Laplace Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """
    components = K.reshape(parameters,[-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = K.softmax(K.clip(alpha,1e-2,1.))
    
    exponent = K.log(alpha) - float(c) * K.log(2 * sigma) \
    - K.sum(K.abs(K.expand_dims(y_true,2) - mu), axis=1)/(sigma)
    
    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res


def scoring_rule_adv(y_true, y_pred):
    """Fast Gradient Sign Method (FSGM) to implement Adversarial Training
    Note: The 'graphADV' pointer is obtained as global variable
    """
    
    # Compute loss 
    #Note: Replace with 'mean_log_Gaussian_like' if you want a Gaussian kernel.
    error = mean_log_LaPlace_like(y_true, y_pred)
    
    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    # Define gradient of loss wrt input
    grad_error = K.gradients(error,graphADV.input) #Minus is on error function
    # Take sign of gradient, Multiply by constant epsilon, Add perturbation to original example to obtain adversarial example
    #Sign add a new dimension we need to obviate
    
    epsilon = 0.08
    
    adversarial_X = K.stop_gradient(graphADV.input + epsilon * K.sign(grad_error)[0])
    
    # Note: If you want to test the variation of adversarial training 
    #  proposed by XX, eliminate the following comment character 
    #  and comment the previous one.
    
    ##adversarial_X = graphADV.input + epsilon * K.sign(grad_error)[0]
    
    adv_output = graphADV(adversarial_X)
    
    #Note: Replace with 'mean_log_Gaussian_like' if you want a Gaussian kernel.
    adv_error = mean_log_LaPlace_like(y_true, adv_output)
    return 0.3 * error + 0.7 * adv_error