package com.joemort;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.pmml.consumer.NeuralNetwork;
import weka.classifiers.trees.Id3;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToBinary;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Standardize;

import java.util.Random;

/**
 * Created by Joseph Mortensen on 4/25/2015.
 * ¯\_(ツ)_/¯
 */
public class ClassifierShell {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("cardata.csv");
        Instances dataSetPre = source.getDataSet();

        dataSetPre.setClassIndex(dataSetPre.numAttributes() - 1);
        //dataSetPre.setClassIndex(0);

        Standardize stand = new Standardize();
        stand.setInputFormat(dataSetPre);

        Discretize discretize = new Discretize();
        discretize.setInputFormat(dataSetPre);

        NumericToNominal ntb = new NumericToNominal();
        ntb.setInputFormat(dataSetPre);

        Instances dataSet = dataSetPre;

        dataSet = Filter.useFilter(dataSet, stand);


        dataSet.randomize(new Random(9001));

        Classifier classify = new NeuralNetworkClassifier(3, 10000, 0.1);
        Evaluation eval = new Evaluation(dataSet);

        int trainingSize = (int) Math.round(dataSet.numInstances() * .7);
        int testSize = dataSet.numInstances() - trainingSize;

        Instances trainingData = new Instances(dataSet, 0, trainingSize);
        Instances testData = new Instances(dataSet, trainingSize, testSize);

        //Evaluation eval = new Evaluation(trainingData);
        classify.buildClassifier(trainingData);
        eval.evaluateModel(classify, testData);
        /*final int folds = 2;
        for (int n = 0; n < folds; n++) {
            Instances train = dataSet.trainCV(folds, n);
            Instances test = dataSet.testCV(folds, n);

            Classifier clsCopy = Classifier.makeCopy(classify);
            clsCopy.buildClassifier(train);
            eval.evaluateModel(clsCopy, test);
        }*/

        System.out.println(eval.toSummaryString("\n" + 0 + " Fold Cross Validation\n==============================================="
                + "===================", false));


    }
}
