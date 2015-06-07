package com.joemort;

import javafx.util.Pair;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by Joseph Mortensen on 5/22/2015.
 * ¯\_(ツ)_/¯
 */
public class NeuralNetworkClassifier extends Classifier {
    Network network;
    int layers = 3;
    int iterations = 50;
    double learningFactor = 0.3;

    public NeuralNetworkClassifier(int layers, int iterations, double learningFactor) {
        this.layers = layers;
        this.iterations = iterations;
        this.learningFactor = learningFactor;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {

        ////System.out.println("learning factor: " + learningFactor);
        int inputCount = instances.numAttributes() - 1;

        List<Integer> nodesPerLayer = new ArrayList<>();

        for (int i = 0; i < layers - 1; i++) {
            //nodesPerLayer.add(4); // hardcoded for now TODO: next week
            nodesPerLayer.add(inputCount);
        }

        nodesPerLayer.add(instances.numDistinctValues(instances.classIndex()));

        ////System.out.println("number of nodes per layer");
        for (Integer i : nodesPerLayer) {
            //System.out.println(i);
        }

        //System.out.println();

        network = new Network(inputCount, nodesPerLayer);

        for (Layer l : network.layers) {
            //System.out.println("new layer");
            for (Neuron neuron : l.neurons) {
                //System.out.println("new neuron");
                for (Double w : neuron.weights) {
                    //System.out.println(w);
                }
                //System.out.println();
            }
        }

        ArrayList<Double> errorsPerIteration = new ArrayList<>();
        pimps: for (int j = 0; j < iterations; j++) {
            for (int k = 0; k < instances.numInstances(); k++) {
                Instance instance = instances.instance(k);

                List<Double> input = new ArrayList<>();

                //System.out.println("num attributes: " + instance.numAttributes());
                for (int i = 0; i < instance.numAttributes(); i++) {
                    if (i == instance.classIndex()) {
                        //System.out.println("expected class: " + instance.value(i));
                    } else if (Double.isNaN(instance.value(i))) {
                        input.add(0.0);
                    } else {
                        input.add(instance.value(i));
                    }

                }

                //System.out.println("inputs");
                for (Double in : input) {
                    //System.out.println(in);
                }

                errorsPerIteration.add(network.train(input, instance.value(instance.classIndex()), learningFactor));
                //break pimps;
            }
        }

        //System.out.println();
        //System.out.println("errors per iteration:");
        for (int i = 0; i < errorsPerIteration.size(); i++) {
            //System.out.println(errorsPerIteration.get(i));
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        List<Double> input = new ArrayList<>();
        for (int i = 0; i < instance.numAttributes(); i++) {
            if (Double.isNaN(instance.value(i)) && i != instance.classIndex())
                input.add(0.0);
            else if (i != instance.classIndex())
                input.add(instance.value(i));

        }

        List<Double> outputs = network.getOutputs(input);
        double largeVal = -1;
        int index = -1;
        for (int i = 0; i < outputs.size(); i++) {
            double tmp = outputs.get(i);
            if (tmp > largeVal) {
                largeVal = tmp;
                index = i;
            }
        }

        // TODO: this will not work with regression, only classification
        return index;
    }
}

class Neuron {
    List<Double> weights = new ArrayList<>();
    static Random random = new Random(42);

    public Neuron(int inputCount) {
        // -1/sqrt(inputCount) <= weight <= 1/sqrt(inputCount)
        double oneOver = 1.0 / Math.sqrt(inputCount);
        for (int i = 0; i < inputCount; i++) {
            weights.add(random.nextDouble() * 2.0 * oneOver - oneOver);
        }
    }

    public double produceOutput(List<Double> inputs) {
        if (inputs.size() != weights.size()) {
            throw new UnsupportedOperationException("wrong number of inputs. Expected "
                + weights.size() + " and got " + inputs.size());
        }

        double sum = 0;
        for (int i = 0; i < weights.size(); i++) {
            sum += weights.get(i) * inputs.get(i);
        }

        return (1.0 / (1.0 + Math.exp(-sum)));
    }
}

class Layer {
    List<Neuron> neurons = new ArrayList<>();

    public Layer(int neuronCount, int inputCount) {
        for (int i = 0; i < neuronCount; i++) {
            neurons.add(new Neuron(inputCount));
        }
    }

    public List<Double> produceOutputs(List<Double> inputs) {
        List<Double> outputs = new ArrayList<>();
        for (Neuron neuron : neurons) {
            outputs.add(neuron.produceOutput(inputs));
        }

        return outputs;
    }
}

class Network {
    List<Layer> layers = new ArrayList<>();

    public Network(int inputCount, List<Integer> neuronCountsPerLayer) {
        if (neuronCountsPerLayer.isEmpty()) {
            throw new UnsupportedOperationException("neuronCountsPerLayer is empty");
        }

        // 1 extra for the bias
        layers.add(new Layer(neuronCountsPerLayer.get(0), inputCount + 1));

        for (int i = 1; i < neuronCountsPerLayer.size(); i++) {
            layers.add(new Layer(neuronCountsPerLayer.get(i),
                    neuronCountsPerLayer.get(i - 1) + 1));
        }
    }

    public List<Double> getOutputs(List<Double> inputs) {
        List<Double> outputs = new ArrayList<>(inputs);

        for (Layer layer : layers) {
            // Add bias
            outputs.add(1.0);

            outputs = layer.produceOutputs(outputs);
        }

        return outputs;
    }

    public double train(List<Double> inputs, double classification, double learningValue) {
        ArrayList<List<Double>> allOutputs = new ArrayList<>();
        List<Double> outputs = new ArrayList<>(inputs);
        // feed forward to calculate outputs

        outputs.add(1.0);
        for (Layer layer : layers) {
            outputs = layer.produceOutputs(outputs);
            //System.out.println("outputs:" );
            for (Double d : outputs) {
                //System.out.println(d);
            }
            outputs.add(1.0);
            allOutputs.add(outputs);
            for (Double d : outputs) {
                ////System.out.println("output: " + d);
            }
        }

        ArrayList<ArrayList<Double>> allErrors = new ArrayList<>();
        // work backwards to calculate errors

        // do output nodes
        ArrayList<Double> error = new ArrayList<>();
        List<Double> currentOutputs = allOutputs.get(allOutputs.size() - 1);
        Layer current = layers.get(layers.size() - 1);
        //System.out.println("expected outputs for output nodes:");
        for (int i = 0; i < current.neurons.size(); i++) {

            double expected = (classification == i ? 1 : 0);
            //System.out.println("output node " + i + ": " + expected);
            double errorVal = currentOutputs.get(i) * (1 - currentOutputs.get(i)) * (currentOutputs.get(i) - expected);
            //System.out.println("error for node " + i + ": " + errorVal);
            error.add(errorVal);
        }

        allErrors.add(error);
        //System.out.println();

        // hidden nodes are a different equation
        for (int i = layers.size() - 2; i >= 0; i--) {
            // for each hidden layer
            current = layers.get(i);
            error = new ArrayList<>();
            outputs = allOutputs.get(i);
            //System.out.println("errors per node in (hidden) layer " + i);
            ArrayList<Double> followingError = allErrors.get(0);
            for (int j = 0; j < current.neurons.size(); j++) {
                // for each neuron in current hidden layer
                double sumError = 0;
                //ArrayList<Double> nextError = allErrors.get(0);
                Layer nextLayer = layers.get(i + 1);
                for (int k = 0; k < followingError.size(); k++) {
                    // for each neuron in following layer
                    sumError += followingError.get(k) * nextLayer.neurons.get(k).weights.get(j);
                }

                double errorVal = outputs.get(j) * (1 - outputs.get(j)) * sumError;
                //System.out.println("error in node " + j + ": " + errorVal);
                error.add(errorVal);
            }

            allErrors.add(0, error);
            //output * (1 - output) * foreach node in following layer - that nodes input weight * that nodes error
        }

        // feed forward to update weights based on errors
        inputs.add(-1.0);
        allOutputs.add(0, inputs);
        for (int i = 0; i < layers.size(); i++) {
            // foreach layer
            current = layers.get(i);
            for (int j = 0; j < current.neurons.size(); j++) {
                // foreach neuron in layer
                Neuron neuron = current.neurons.get(j);
                for (int k = 0; k < neuron.weights.size(); k++) {
                    // foreach weight in neuron

                    ////System.out.println("previous value: " + all.get(i).get(k) + "    currentError: " + allErrors.get(i).get(j));
                    double newWeight = neuron.weights.get(k) - allOutputs.get(i).get(k) * allErrors.get(i).get(j) * learningValue;
                    ////System.out.println("Weight change: " + (neuron.weights.get(k) - newWeight));
                    //System.out.println("layer: " + i + " node: " + j + " weight: " + k + "   oldweight: " + neuron.weights.get(k) + " neweight: " + newWeight);
                    neuron.weights.set(k, newWeight);
                }

                current.neurons.set(j, neuron);
            }

            layers.set(i, current);
        }

        // return total error
        double totalError = 0;
        for (List<Double> l : allErrors) {
            for (Double d : l) {
                totalError += Math.abs(d);
            }
        }

        return totalError;
    }
}