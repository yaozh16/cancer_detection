# -*- coding: utf-8 -*-
import torch
from torch import nn,autograd
dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

def autograd_test():
    # Create random Tensors for weights.
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    class MyReLU(torch.autograd.Function):
        """
        We can implement our own custom autograd Functions by subclassing
        torch.autograd.Function and implementing the forward and backward passes
        which operate on Tensors.
        """

        @staticmethod
        def forward(self, input):
            """
            In the forward pass we receive a Tensor containing the input and return
            a Tensor containing the output. ctx is a context object that can be used
            to stash information for backward computation. You can cache arbitrary
            objects for use in the backward pass using the ctx.save_for_backward method.
            """
            self.save_for_backward(input)
            return input.clamp(min=0)

        @staticmethod
        def backward(self, grad_output):
            """
            In the backward pass we receive a Tensor containing the gradient of the loss
            with respect to the output, and we need to compute the gradient of the loss
            with respect to the input.
            """
            input, = self.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            return grad_input

    learning_rate = 1e-6
    for t in range(500):
        relu = MyReLU.apply
        # Forward pass: compute predicted y using operations on Tensors; these
        # are exactly the same operations we used to compute the forward pass using
        # Tensors, but we do not need to keep references to intermediate values since
        # we are not implementing the backward pass by hand.
        y_pred = relu(x.mm(w1)).mm(w2)

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the a scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()
        print(t, loss.item())

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call w1.grad and w2.grad will be Tensors holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        # An alternative way is to operate on weight.data and weight.grad.data.
        # Recall that tensor.data gives a tensor that shares the storage with
        # tensor, but doesn't track history.
        # You can also use torch.optim.SGD to achieve this.
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()

def nn_test():

    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function.
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    for t in range(500):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x)

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, y)
        print(t, loss.item())

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

def optim_test():
    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Tensors it should update.
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for t in range(500):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        print(t, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()


def custom_nn_test1():
    class TwoLayerNet(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            """
            In the constructor we instantiate two nn.Linear modules and assign them as
            member variables.
            """
            super(TwoLayerNet, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, H)
            self.linear2 = torch.nn.Linear(H, D_out)

        def forward(self, x):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            h_relu = self.linear1(x).clamp(min=0)
            y_pred = self.linear2(h_relu)
            return y_pred

    # Construct our model by instantiating the class defined above
    model = TwoLayerNet(D_in, H, D_out)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def custom_nn_test2():
    class TwoLayerNet(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            """
            In the constructor we instantiate two nn.Linear modules and assign them as
            member variables.
            """
            super(TwoLayerNet, self).__init__()
            self.linear1_1 = torch.nn.Linear(int(D_in/2), int(H/2))
            self.linear1_2 = torch.nn.Linear(int(D_in/2), int(H/2))
            self.linear2 = torch.nn.Linear(H, D_out)

        def forward(self, x,y):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            assert y==1
            h1 = self.linear1_1(torch.from_numpy(x.numpy()[:,:int(D_in/2)])).clamp(min=0)
            h2 = self.linear1_2(torch.from_numpy(x.numpy()[:,int(D_in/2):])).clamp(min=0)
            combined = torch.cat((h1.view(h1.size(0), -1),
                                  h2.view(h2.size(0), -1)), dim=1)
            y_pred = self.linear2(combined)
            return y_pred

    # Construct our model by instantiating the class defined above
    model = TwoLayerNet(D_in, H, D_out)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(1500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x,1)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def dynamic_test():
    import random
    class DynamicNet(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            """
            In the constructor we construct three nn.Linear instances that we will use
            in the forward pass.
            """
            super(DynamicNet, self).__init__()
            self.input_linear = torch.nn.Linear(D_in, H)
            self.middle_linear = torch.nn.Linear(H, H)
            self.output_linear = torch.nn.Linear(H, D_out)

        def forward(self, x):
            """
            For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
            and reuse the middle_linear Module that many times to compute hidden layer
            representations.

            Since each forward pass builds a dynamic computation graph, we can use normal
            Python control-flow operators like loops or conditional statements when
            defining the forward pass of the model.

            Here we also see that it is perfectly safe to reuse the same Module many
            times when defining a computational graph. This is a big improvement from Lua
            Torch, where each Module could be used only once.
            """
            h_relu = self.input_linear(x).clamp(min=0)
            for _ in range(random.randint(0, 3)):
                h_relu = self.middle_linear(h_relu).clamp(min=0)
            y_pred = self.output_linear(h_relu)
            return y_pred

    # Construct our model by instantiating the class defined above
    model = DynamicNet(D_in, H, D_out)

    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    for t in range(2000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

custom_nn_test2()