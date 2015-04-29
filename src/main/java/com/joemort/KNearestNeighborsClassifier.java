package com.joemort;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.Map.Entry;

/**
 * Created by Joseph Mortensen on 4/28/2015.
 * ¯\_(ツ)_/¯
 */
public class KNearestNeighborsClassifier extends Classifier {

    final int k;
    Instances saved;

    public KNearestNeighborsClassifier(int k) {
        this.k = k;
    }

    private static double distance(Instance one, Instance two) {
        double total = 0;
        int totalAttributes = one.numAttributes() - 1;
        for (int i = 0; i < totalAttributes; i++) {
            if (one.classIndex() == i)
                continue;

            double difference = 0;

            if (one.attribute(i).isNumeric()) {
                difference = Math.abs(one.value(i) - two.value(i));
            } else {
                if (!one.stringValue(i).equals(two.stringValue(i))) {
                    difference = 1;
                }
            }

            total += Math.pow(difference, totalAttributes);
        }

        return Math.pow(total, 1.0/totalAttributes);
    }

    public static double getClassification(List<Instance> instances) {
        int index = instances.get(0).classIndex();
        HashMap<Double, Integer> counts = new HashMap<>();
        for (Instance instance : instances) {
            double val = instance.value(index);
            if (!counts.containsKey(val)) {
                counts.put(val, 1);
            } else {
                counts.put(val, counts.get(val) + 1);
            }
        }

        // Use a 'random' pick of whichever one has the largest value,
        // no tie breaking algorithms is implemented.
        int maxCount = 0;
        double maxRValue = 0;
        for (Entry<Double, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                maxRValue = entry.getKey();
            }
        }

        return maxRValue;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        saved = new Instances(instances);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        HashMap<Instance, Double> map = new HashMap<>();
        for (int i = 0; i < saved.numInstances(); i++) {
            Instance tmp = saved.instance(i);
            map.put(tmp, distance(tmp, instance));
        }

        List<Instance> kNearest = new ArrayList<>();
        dance: for (Entry<Instance, Double> inst : entriesSortedByValues(map)) {
            // Always do at least 1
            kNearest.add(inst.getKey());

            if (kNearest.size() >= k) {
                break dance;
            }
        }

        return getClassification(kNearest);
    }

    static <K,V extends Comparable<? super V>> List<Entry<K, V>> entriesSortedByValues(Map<K,V> map) {
        List<Entry<K,V>> sortedEntries = new ArrayList<>(map.entrySet());
        Collections.sort(sortedEntries, new Comparator<Entry<K, V>>() {
                    @Override
                    public int compare(Entry<K, V> e1, Entry<K, V> e2) {
                        return e1.getValue().compareTo(e2.getValue());
                    }
                }
        );

        return sortedEntries;
    }
}
