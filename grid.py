class Node:
    
    def __init__(self, state):
        self.state = state
        self.next = []
        
    def connect(self, node, operation_type, operation):
        self.next.append((operation_type, operation, node))


def forkSteepnessIncrease(nn, node):
    
    # Save parameters
    old_params = nn.getParams()
    
    current_node = node
    
    for step in range(8):
        
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


def generateSparseningThenSteepeningGrid(main_nn):
    
    root_node = Node(main_nn.getState())
    current_node = root_node

    for stage in range(30):

        # Train the model
        training_op = main_nn.train(100)
        next_node = Node(main_nn.getState())
        current_node.connect(next_node, "training", training_op)
        current_node = next_node

        # Make a fork from the main model and increase steepness
        forkSteepnessIncrease(main_nn, current_node)

        # Increase sparseness
        sparsening_op = main_nn.sparsen()
        next_node = Node(main_nn.getState())
        current_node.connect(next_node, "sparsening", sparsening_op)
        current_node = next_node

        with open("{}/intermediate-{}.pickle".format(folder_name, stage), "wb") as output_file:
            pickle.dump({"root": root_node}, output_file)

    # Train the model
    training_op = main_nn.train(100)
    next_node = Node(main_nn.getState())
    current_node.connect(next_node, "training", training_op)
    current_node = next_node

    # Make a fork from the main model and increase steepness
    forkSteepnessIncrease(main_nn, current_node)
    
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
                    
            current_node = next_node
            op, op_data, next_node = current_node.next[0]
            current_node = next_node
            steepening_output.append(current_node.state)
            
        current_node = cached_node
        output.append(steepening_output)
        
        for op, op_data, next_node in current_node.next:
            if op == "sparsening": break
        current_node = next_node
        
    return output