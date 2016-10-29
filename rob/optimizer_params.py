# def choose_optimiser(self, optimizer_name, learning_rate=1e-4,
#                      initial_accumulator_value=0.1,  # Used for Adagrad
#                      beta1=0.9,  # Used for Adam
#                      beta2=0.999,  # Used for Adam
#                      epsilon=1e-8,  # Used for Adam, Adadelta
#                      rho=0.95  # Used for Adadelta


class GradientDescentParams:
    def __init__(self, learning_rate=1e-4):
        self.learning_rate=learning_rate
        self.name = 'GD'

    def to_string(self):
        return "Gradient Descent. Learning rate: {}".format(self.learning_rate)


class AdagradParams:
    def __init__(self, learning_rate=1e-4, initial_accumulator_value=0.1):
        self.learning_rate=learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self.name = 'Adagrad'

    def to_string(self):
        return "Adagrid. Learning rate: {}, Initial Accumulator Value: {}"\
            .format(self.learning_rate, self.initial_accumulator_value)


class AdadeltaParams:
    def __init__(self, learning_rate=1e-4, epsilon=1e-8, rho=0.95):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.rho=rho
        self.name = 'Adadelta'

    def to_string(self):
        return "Adadelta. Learning rate: {}, Epsilon: {}, Rho: {}"\
            .format(self.learning_rate, self.epsilon, self.rho)


class AdamParams:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8 ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.name='Adam'

    def to_string(self):
        return "Adam. Learning rate: {}, Epsilon: {}, Beta1: {}, Beta2: {}" \
            .format(self.learning_rate, self.epsilon, self.beta1, self.beta2)