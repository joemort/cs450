package com.joemort;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

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
        int totalAttributes = one.numAttributes();
        for (int i = 0; i < totalAttributes; i++) {
            if (one.classIndex() == i)
                continue;

            double difference = 0;

            if (one.attribute(i).isNumeric()) {
                difference = Math.abs(one.value(i) - two.value(i));
            } else {
                if (one.stringValue(i).equals(two.stringValue(i))) {
                    difference = 1;
                }
            }

            total += Math.pow(difference, totalAttributes);
        }

        return Math.pow(total, 1.0/totalAttributes);
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        saved = new Instances(instances);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }

    //class
}
