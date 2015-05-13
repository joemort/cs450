package com.joemort;

import javafx.util.Pair;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Created by Joseph Mortensen on 5/11/2015.
 * ¯\_(ツ)_/¯
 */
public class ID3Classifier extends Classifier {

    Node tree;

    private Pair<Boolean, Double> sameClass(List<Instance> instances) {
        int classIndex = instances.get(0).classIndex();
        double tmpValue = Double.NaN;
        for (int i = 0; i < instances.size(); i++) {
            Instance instance = instances.get(i);
            if (Double.isNaN(tmpValue)) {
                // Assign value if we haven't yet
                tmpValue = instance.value(classIndex);
            } else {
                double val = instance.value(classIndex);
                if (!Double.isNaN(val)) {
                    // only check if the data is there
                    if (val != tmpValue) {
                        // we've found multiple data values, not the same
                        return new Pair<>(false, Double.NaN);
                    }
                }
            }
        }

        // only found 1 value for the class
        return new Pair<>(true, tmpValue);
    }

    private Node buildTree(List<Instance> instances, List<Attribute> attributes) {
        if (instances.size() < 1) {
            throw new UnsupportedOperationException("Instances shouldn't be empty");
        }

        Pair<Boolean, Double> classification = sameClass(instances);
        if (classification.getKey()) {
            return new Node(classification.getValue());
        }

        if (attributes.isEmpty()) {
            return new Node(KNearestNeighborsClassifier.getClassification(instances));
        }

        Attribute largestGain = getMaxGain(instances, attributes);
        Node n = new Node(instances, largestGain);
        Map<Instance, Double> vals =  valuesByAttribute(instances, largestGain);
        Map<Double, Integer> summary = summarizeValues(vals);
        ArrayList<Attribute> newList = new ArrayList<>(attributes);
        newList.remove(largestGain);
        for (Double value : summary.keySet()) {
            Node idNode = buildTree(subset(vals, value), newList);
            n.addChild(value, idNode);
        }

        return n;
    }

    private List<Instance> subset(Map<Instance, Double> map, double value) {
        ArrayList<Instance> list = new ArrayList<>();
        for (Instance instance : map.keySet()) {
            if (map.get(instance) == value) {
                list.add(instance);
            }
        }

        return list;
    }

    private Map<Instance, Double> valuesByAttribute(List<Instance> instances, Attribute attribute) {
        HashMap<Instance, Double> map = new HashMap<>();

        for (Instance instance : instances) {
            map.put(instance, instance.value(attribute));
        }

        if (!attribute.isNominal()) {
            for (Instance key : map.keySet()) {
                if (map.get(key) == Double.NaN) {
                } else if (map.get(key) < 0.5) {
                    map.put(key, 0.0); // 'Low'
                } else {
                    map.put(key, 1.0); // 'High'
                }
            }
        }

        return map;
    }

    /**
     Double is the value, integer is the count of that value
     **/
    private Map<Double, Integer> summarizeValues(Map<Instance, Double> input) {
        HashMap<Double, Integer> hashMap = new HashMap<>();

        for (Instance i : input.keySet()) {
            if (!hashMap.containsKey(input.get(i)) ||
                    hashMap.get(i) == null) {
                hashMap.put(input.get(i), 1);
            } else {
                hashMap.put(input.get(i), hashMap.get(i) + 1);
            }
        }

        return hashMap;
    }

    public Attribute getMaxGain(List<Instance> instances, List<Attribute> attributes) {
        Pair<Attribute, Double> maxGain = new Pair<>(null, Double.NEGATIVE_INFINITY);
        double totalEntropy = entropy(instances);
        for (Attribute attribute : attributes) {
            double tmpGain = gain(instances, attribute, totalEntropy);
            if (tmpGain > maxGain.getValue()) {
                maxGain = new Pair<>(attribute, tmpGain);
            }
        }

        return maxGain.getKey();
    }

    private double gain(List<Instance> instances, Attribute attribute, double entropyOfSet) {
        double gain = entropyOfSet;
        Map<Instance, Double> values = valuesByAttribute(instances, attribute);
        HashSet<Double> valueSet = new HashSet<>(values.values());
        for (Double d : valueSet) {
            List<Instance> sub = subset(values, d);
            gain -= sub.size() * 1.0 / instances.size() * entropy(sub);
        }

        return gain;
    }

    private double entropy(List<Instance> instances) {
        double result = 0;
        Map<Double, Integer> summary = summarizeValues(valuesByAttribute(instances, instances.get(0).classAttribute()));
        for (Integer val : summary.values()) {
            double proportion = val * 1.0 / instances.size();
            result -= proportion * Math.log(proportion) / Math.log(2);
        }

        return result;
    }

    public void printTree(Node node, int level, Double value) {
        if (level == 0) {
            System.out.println(node.attribute.name() + " -");
        } else {
            for (int i = 0; i < level; i++)
                System.out.print('\t');

            System.out.print(value);

            if (!node.isLeaf()) {
                System.out.print(" : "
                        + node.getAttribute().name());
            }

            System.out.print(" -");

            if (node.isLeaf()) {
                System.out.println(" " + node.leafValue);
            } else {
                System.out.println();
            }
        }

        for (Node n : node.getChildren()) {
            printTree(n, level + 1, node.get(n));
        }
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        List<Instance> instanceList = new ArrayList<>(instances.numInstances());
        for (int i = 0; i < instances.numInstances(); i++) {
            instanceList.add(instances.instance(i));
        }

        List<Attribute> attributeList = new ArrayList<>(instances.numAttributes());
        for (int i = 0; i < instances.numAttributes(); i++) {
            if (i != instances.classIndex())
                attributeList.add(instances.attribute(i));
        }

        tree = buildTree(instanceList, attributeList);
        printTree(tree, 0, 0.0);
    }

    public double getClassification(Instance instance, Node n) {
        if (n.isLeaf()) {
            return n.leafValue;
        } else {
            Attribute attribute = n.getAttribute();
            if (Double.isNaN(instance.value(attribute))) {
                Map<Double, Integer> classToCount = new HashMap<>();
                for (Node child : n.getChildren()) {
                    Double value = getClassification(instance, child);
                    if (!classToCount.containsKey(value) || classToCount.get(value) == null) {
                        classToCount.put(value, classToCount.get(value) + 1);
                    } else {
                        classToCount.put(value, 1);
                    }
                }

                int maxCount = -1;
                double maxValue = 0;
                for (Double d : classToCount.keySet()) {
                    if (classToCount.get(d) > maxCount) {
                        maxCount = classToCount.get(d);
                        maxValue = d;
                    }
                }

                return maxValue;
            } else {
                Double val = instance.value(n.getAttribute());
                if (val == null || Double.isNaN(val))
                    val = 0.0;

                // convert val to one of the 2 values we have doubles.
                if (!n.getAttribute().isNominal()) {
                    if (val < 0.5) val = 0.0;
                    else val = 1.0;
                }

                Node child = n.get(val);
                if (child == null) {
                    return KNearestNeighborsClassifier.getClassification(n.instances);
                } else {
                    return getClassification(instance, n.get(val));
                }
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return getClassification(instance, tree);
    }
}

class Node {
    List<Instance> instances;
    private Map<Node, Double> children = new HashMap<>();
    private Map<Double, Node> children2 = new HashMap<>();
    Attribute attribute;
    boolean isLeaf = false;
    double leafValue;

    public Node(List<Instance> instances, Attribute attribute) {
        this.instances = instances;
        this.attribute = attribute;
    }

    public Node(double leafValue) {
        isLeaf = true;
        this.leafValue = leafValue;
    }

    public boolean isLeaf() {
        return isLeaf;
    }

    public Attribute getAttribute() {
        return attribute;
    }

    public void addChild(Double value, Node n) {
        children.put(n, value);
        children2.put(value, n);
    }

    public Double get(Node n) {
        return children.get(n);
    }

    public Node get(Double d) {
        return children2.get(d);
    }

    public Set<Node> getChildren() {
        return children.keySet();
    }
}