package HomeWork1;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.ThreadLocalRandom;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class MainHW1 {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
		
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) throws Exception 
	{
		//load data for training.
		Instances trainingData = loadData("C:\\Users\\Oz\\eclipse-workspace\\EX1 - Linear regression\\src\\HomeWork1\\wind_training.txt");
		//train classifier
		LinearRegression training_LinearRegression= new LinearRegression();
		training_LinearRegression.buildClassifier(trainingData);
		
		// load data for testing.
		Instances testingData = loadData("C:\\Users\\Oz\\eclipse-workspace\\EX1 - Linear regression\\src\\HomeWork1\\wind_testing.txt");
		training_LinearRegression.calculateSE(testingData);
		// print the error.
		System.out.println(training_LinearRegression.calculateSE(testingData));
	}

}
