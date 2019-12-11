import numpy as np


class Node:
    
    def __init__(self, state):
        self.state = state
        self.next = []
        
    def connect(self, node, operation_type, operation):
        self.next.append((operation_type, operation, node))


def forkSteepnessIncrease(nn, node, steepening_cycles):
    
    # Save parameters
    old_params = nn.getParams()
    
    current_node = node
    
    for step in range(steepening_cycles):
        
        # Increase steepness
        steepening_op = nn.steepen()
        next_node = Node(nn.getState())
        current_node.connect(next_node, "steepening", steepening_op)
        current_node = next_node
        
        # Train the model
        training_op = nn.train(100)
        next_node = Node(nn.getState())
        current_node.connect(next_node, "training", training_op)
        current_node = next_node
        
    nn.setParams(old_params)


def generateSparseningThenSteepeningGrid(main_nn, sparsening_cycles, steepening_cycles):
    
    root_node = Node(main_nn.getState())
    current_node = root_node

    for stage in range(sparsening_cycles):

        # Train the model
        training_op = main_nn.train(100)
        next_node = Node(main_nn.getState())
        current_node.connect(next_node, "training", training_op)
        current_node = next_node

        # Make a fork from the main model and increase steepness
        forkSteepnessIncrease(main_nn, current_node, steepening_cycles)

        # Increase sparseness
        sparsening_op = main_nn.sparsen()
        next_node = Node(main_nn.getState())
        current_node.connect(next_node, "sparsening", sparsening_op)
        current_node = next_node

    # Train the model
    training_op = main_nn.train(100)
    next_node = Node(main_nn.getState())
    current_node.connect(next_node, "training", training_op)
    current_node = next_node

    # Make a fork from the main model and increase steepness
    forkSteepnessIncrease(main_nn, current_node, steepening_cycles)
    
    return root_node
    

def processSparseningThenSteepeningGrid(root_node):
    
    output = []
    current_node = root_node
    
    while len(current_node.next) > 0:
        
        # Process training operation
        op, op_data, next_node = current_node.next[0]
        current_node = next_node
        
        # The current node is the base of the steepening within the current sparsity
        steepening_output = []
        steepening_output.append(current_node.state)
        cached_node = current_node
        
        while len(current_node.next) > 0:
            for op, op_data, next_node in current_node.next:
                if op == "steepening": break
            if op != "steepening": break
                    
            current_node = next_node
            op, op_data, next_node = current_node.next[0]
            current_node = next_node
            steepening_output.append(current_node.state)
            
        current_node = cached_node
        output.append(steepening_output)
        
        for op, op_data, next_node in current_node.next:
            if op == "sparsening": break
        if op != "sparsening": break
        current_node = next_node
        
    return output


def mapMetric(func, grid, *args, **kwargs):
    return np.array([[func(state, *args, **kwargs) for state in sublist] for sublist in grid])