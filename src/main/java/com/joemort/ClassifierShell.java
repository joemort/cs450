package com.joemort;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

import java.util.Random;

/**
 * Created by Joseph Mortensen on 4/25/2015.
 * ¯\_(ツ)_/¯
 */
public class ClassifierShell {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("irisdata.csv");
        Instances dataSetPre = source.getDataSet();
        Standardize stand = new Standardize();
        stand.setInputFormat(dataSetPre);
        Instances dataSet = Filter.useFilter(dataSetPre, stand);

        dataSet.setClassIndex(dataSet.numAttributes() - 1);

        dataSet.randomize(new Random(1));



        int trainSize = (int) Math.round(dataSet.numInstances() * .7);
        int testSize = dataSet.numInstances() - trainSize;
        Instances train = new Instances(dataSet, 0, trainSize);
        Instances test = new Instances(dataSet, trainSize, testSize);

        Classifier classify = new ID3Classifier();
        classify.buildClassifier(train);

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(classify, test);
        System.out.println(eval.toSummaryString("\nResults\n==============================================="
                + "===================", true));


    }
}
