class InvalidArchitectureError(Exception):
    """ Custom Error raised when the architecture of a Network is invalid """

    def __init__(self, **kwargs):
        """ constructor """
        super(InvalidArchitectureError, self).__init__()
        self.message = "The architecture of the Nerwork is invalid. "
        if kwargs:
            self.current_shape = kwargs.get("shape", None)
            self.layer = kwargs.get("layer", None)
        else:
            self.current_shape = None
            self.layer = None

    def __str__(self):
        """ print when raised outside try block """
        if self.layer:
            self.message += "During the addition of the convolution layer {}, the shape of the " \
                            "data becomes {}.".format(self.layer, self.current_shape)
        return self.message
