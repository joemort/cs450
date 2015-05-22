package com.joemort;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Joseph Mortensen on 5/22/2015.
 * ¯\_(ツ)_/¯
 */
public class NeuralNetworkClassifier {
    // TODO next week
}

class Neuron {
    List<Double> weights = new ArrayList<>();

    public Neuron(int inputCount) {
        for (int i = 0; i < inputCount; i++) {
            weights.add(Math.random() * 2.0 - 1.0);
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

        return (sum <= 0 ? 0 : 1);
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
}