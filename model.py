#!/usr/bin/env python

# Daniel Winkelman
# 2019, Duke University

# Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import sys
import os


# Import pre-processed data from pickle files
def importData(categories, subset):
    
    with open("data/{}-{}.pickle".format(categories, subset), "rb") as input_file:
        data = pickle.loads(input_file.read())
        
    return data["in"], data["out"]

###############################################################################
# Observations about sparse weights (and managing multi-layered sparseness):
#  * if a row in a weights matrix is zero, then the corresponding input does
#    not matter
#  * if a column in a weights matrix is zero, the the corresponding output does
#    not matter
#  * an input that does not matter makes the corresponding row in a weights
#    matrix not matter
#  * an output that does not matter makes the corresponding column in a weights
#    matrix not matter

# Define a custom Keras dense layer whose weights can be pruned using a mask
class SparseDense(tf.keras.layers.Dense):
    
    def __init__(self, units, sparsening_rate, **kwargs):
        super(SparseDense, self).__init__(units, **kwargs)
        self.sparsening_rate = sparsening_rate
        self.last_sparse_layer = None
        self.next_sparse_layer = None

    def build(self, input_shape):
        super(SparseDense, self).build(input_shape)
        self.sparsity_matrix_tensor = self.add_weight(
            name="sparsity_matrix",
            shape=self.kernel.shape,
            initializer=tf.keras.initializers.Ones,
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False)
        self.sparsity_matrix = np.ones(self.kernel.shape)
        
    def call(self, inputs):
        # Mask kernel with sparsity matrix and apply activation function
        masked_kernel = self.kernel * self.sparsity_matrix
        output = tf.keras.backend.dot(inputs, masked_kernel)
        return tf.keras.backend.bias_add(output, self.bias, data_format="channels_last")
    
    def removeRedundantWeights(self):
        changed = False
        
        # Zero columns in the previous weights matrix indicate that the
        # parameters in the corresponding row do not matter
        if self.last_sparse_layer is not None:
            for index, col in enumerate(self.last_sparse_layer.sparsity_matrix.T):
                if col.sum() == 0:
                    self.sparsity_matrix[index] = 0
                    changed = True
                    
        # Zero rows in the next weights matrix indicate that the parameters
        # in the corresponding column do not matter
        if self.next_sparse_layer is not None:
            for index, row in enumerate(self.next_sparse_layer.sparsity_matrix):
                if row.sum() == 0:
                    self.sparsity_matrix[:, index] = 0
                    changed = True
                    
        return changed
    
    def makeProportionallySparse(self, p=None):
        kernel = tf.keras.backend.get_value(self.kernel)
        p = self.sparsening_rate if p is None else p
        
        self.removeRedundantWeights()
        
        # Remove some percentage of the remaining weights
        items = []
        for row in range(kernel.shape[0]):
            for col in range(kernel.shape[1]):
                if self.sparsity_matrix[row][col] > 0:
                    items.append((row, col, kernel[row][col]))
                    
        items.sort(key=lambda x: abs(x[2]))
        num_to_remove = int(np.ceil(len(items) * p))
        for row, col, val in items[:num_to_remove]:
            self.sparsity_matrix[row][col] = 0
            
        tf.keras.backend.set_value(self.sparsity_matrix_tensor, self.sparsity_matrix)
        
        return self.getSparsity()
        
    def getSparsity(self):
        return 1 - np.sum(self.sparsity_matrix) / self.sparsity_matrix.size


# Define a custom Keras activation layer with variable steepness
class ParametrizedSigmoid(tf.keras.layers.Activation):
    
    def __init__(self, steepness=1.0):
        super(ParametrizedSigmoid, self).__init__(None)
        self.activation = self.function
        self._steepness = float(steepness)
        self.use_hard_cutoff = False
      
    def setSteepness(self, steepness):
        self._steepness = steepness
        
    def getSteepness(self):
        return self._steepness
        
    def setHardCutoff(self, use_hard_cutoff):
        self.use_hard_cutoff = use_hard_cutoff
        
    def function(self, inputs):
        if self.use_hard_cutoff:
            return tf.keras.backend.cast_to_floatx(tf.keras.backend.greater(inputs, 0))
        else:
            return tf.keras.backend.sigmoid(inputs * self._steepness)
        
    def get_config(self):
        config = super(ParametrizedSigmoid, self).get_config()
        config.update({
            "steepness": self._steepness,
            "use_hard_cutoff": self.use_hard_cutoff
        })
        return config

    
# Define a neural network model
class NeuralNetwork:

    def __init__(self, layer_sizes, output_categories, dropout, batch_size):
        self.dropout = dropout
        self.output_categories = output_categories
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        
        # Parameter to horizontally compress the sigmoid activation
        self.activation_param = 1.0
        
        # Import data
        self.training_in, self.training_out = importData(output_categories, "training")
        self.validation_in, self.validation_out = importData(output_categories, "validation")
        self.testing_in, self.testing_out = importData(output_categories, "testing")
        
        # Create the neural network
        self.makeClassifier()
    
    
    def makeClassifier(self):
        """Construct the model topology"""
        
        self.classifier = tf.keras.Sequential()
        
        self.sparse_layers = []
        self.activation_layers = []
        
        def addSparseLayer(output_dim, sparsening_rate, input_dim=None):
            if input_dim is not None:
                layer = SparseDense(output_dim, sparsening_rate, kernel_initializer="random_normal", input_dim=input_dim)
            else:
                layer = SparseDense(output_dim, sparsening_rate, kernel_initializer="random_normal")
            self.sparse_layers.append(layer)
            self.classifier.add(layer)
            
        def addActivationLayer():
            layer = ParametrizedSigmoid()
            self.activation_layers.append(layer)
            self.classifier.add(layer)
        
        # First hidden layer (with sigmoidal activation)
        addSparseLayer(self.layer_sizes[0], 0.2, self.training_in.shape[1])
        addActivationLayer()
        self.classifier.add(tf.keras.layers.Dropout(self.dropout))
        
        # Second through penultimate hidden layers (with sigmoidal activation)
        for size in self.layer_sizes[1:]:
            addSparseLayer(size, 0.1)
            addActivationLayer()
            self.classifier.add(tf.keras.layers.Dropout(self.dropout))
            
        # Final layer (with softmax activation)
        self.classifier.add(tf.keras.layers.Dense(self.training_out.shape[1], kernel_initializer="random_normal"))
        self.classifier.add(tf.keras.layers.Softmax())
        
        # Link sparse layers so they communicate about redundant weights
        for current_layer, next_layer in zip(self.sparse_layers[:-1], self.sparse_layers[1:]):
            current_layer.__dict__["next_sparse_layer"] = next_layer
            next_layer.__dict__["last_sparse_layer"] = current_layer
        
        self.recompile()
        
        
    def recompile(self):
        # Compile model
        self.classifier.compile(optimizer="adam", loss="binary_crossentropy",
                           metrics=["accuracy"])
        
        
    def train(self, epochs=20):
        """An operation that can be performed on a network;
        The network is trained until a metric stops improving;
        Returns data on the performance of the network as a function of training time
        """
        
        steepness = self.activation_layers[0].getSteepness()
        sparsity = self.sparse_layers[0].getSparsity()
        
        class Logger(tf.keras.callbacks.Callback):
            
            def __init__(self):
                super(Logger, self).__init__()
                self.epoch_number = 1
                
            def on_epoch_begin(self, epoch, logs={}):
                self.start_time = time.time()
            
            def on_epoch_end(self, epoch, logs={}):
                time_diff = time.time() - self.start_time
                stats = ", ".join(["{} = {:.4f}".format(k, v) for k, v in logs.items()])
                print("Epoch {:2d} [{:.2f}] (Sparsity = {:.4f}, Steepness = {}): {}".format(
                        self.epoch_number, time_diff, sparsity, steepness, stats))
                self.epoch_number += 1
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=5, restore_best_weights=False)
        
        output = self.classifier.fit(self.training_in, self.training_out,
                                     validation_data=(self.validation_in, self.validation_out),
                                     batch_size=self.batch_size,
                                     epochs=epochs,
                                     callbacks=[early_stopping, Logger()], verbose=0)
        
        return {"history": output.history, "parameters": output.params}
    
    
    def sparsen(self):
        """An operation that makes sparse layers more sparse"""
        
        output = []
        
        for layer in self.sparse_layers:
            initial_sparsity = layer.getSparsity()
            layer.makeProportionallySparse()
            final_sparsity = layer.getSparsity()
            output.append({
                "initial_sparsity": initial_sparsity,
                "final_sparsity": final_sparsity,
                "function": "makeProportionallySparse",
                "arguments": {"p": layer.sparsening_rate}
            })
            
        for layer in self.sparse_layers:
            layer.removeRedundantWeights()
            
        self.recompile()
            
        return output
    
    
    def steepen(self):
        """An operation that makes sigmoidal activation functions steeper"""
        
        output = []
        
        for layer in self.activation_layers:
            initial_steepness = layer.getSteepness()
            layer.setSteepness(layer.getSteepness() * 2)
            final_steepness = layer.getSteepness()
            output.append({
                "initial_steepness": initial_steepness,
                "final_steepness": final_steepness,
            })
            
        self.recompile()
          
        return output
    
    
    def evaluate(self):
        """Perform a battery of tests on the model"""
        
        def confusionMatrix(predicted, actual):
            """Confusion matrices use the convention that
            each type of prediction is a row and each type
            of actual is a column"""
            
            matrix = np.zeros((predicted.shape[1], actual.shape[1]))
            
            for pred, act in zip(np.argmax(predicted, axis=1), np.argmax(actual, axis=1)):
                matrix[pred][act] += 1
                
            return matrix
        
        output = {}
        sys.stdout = open(os.devnull, 'w')
        
        data_sets = [
            ("training", self.training_in, self.training_out),
            ("validation", self.validation_in, self.validation_out),
            ("testing", self.testing_in, self.testing_out)
        ]
        
        activation_states = [
            ("sigmoidal", False),
            ("hard_cutoff", True)
        ]
        
        for data_name, data_in, data_out in data_sets:
            output[data_name] = {}
            
            for activation_name, activation_state in activation_states:
                
                # Prepare activation functions for hard cutoff or no hard cutoff
                for layer in self.activation_layers:
                    layer.setHardCutoff(activation_state)
                self.recompile()
                    
                # Get metrics for current model
                metrics = self.classifier.evaluate(data_in, data_out, 512)
                
                # Make predictions using the current model
                predictions = self.classifier.predict(data_in, 512)
                
                output[data_name][activation_name] = {
                    "metrics": dict(zip(self.classifier.metrics_names, metrics)),
                    "confusion_matrix": confusionMatrix(predictions, data_out)
                }
                
        # Reset activation functions
        for layer in self.activation_layers:
            layer.setHardCutoff(False)
        self.recompile()
                
        sys.stdout = sys.__stdout__
        return output
    
    
    def getState(self):
        """Get the configuration, parameters, and performance of the model"""
        
        config = self.classifier.get_config()
        config.update({
            "output_categories": self.output_categories,
            "batch_size": self.batch_size
        })
        
        return {
            "eval": self.evaluate(),
            "params": self.getParams(),
            "config": config
        }
    
    
    def getParams(self):
        return {
            "dense": [layer.get_weights() for layer in self.classifier.layers if isinstance(layer, tf.keras.layers.Dense)],
            "activation": [layer.getSteepness() for layer in self.activation_layers]
        }
    
    
    def setParams(self, params):
        dense_layers = [layer for layer in self.classifier.layers if isinstance(layer, tf.keras.layers.Dense)]
        for layer, param in zip(dense_layers, params["dense"]):
            layer.set_weights(param)
            
        for layer, param in zip(self.activation_layers, params["activation"]):
            layer.setSteepness(param)
            layer.setHardCutoff(False)