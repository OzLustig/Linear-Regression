package HomeWork1;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {
	
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;
	// Defining the required change between the SE's calculation in order to find the alpha.
	private double epsilon = 0.003;
	
	/**
	 * Getter for the weights of the attributes for part 3.e. ie. printing out the weights to a file.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	public double[] getWeights() 
	{
		return m_coefficients;
	}
	
	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception 
	{
		trainingData = new Instances(trainingData);
		m_ClassIndex = trainingData.classIndex();
		//since class attribute is also an attribute we subtract 1
		m_truNumAttributes = trainingData.numAttributes() - 1;
		
		// Initialize the m_coefficients array with zero's.
		m_coefficients = new double[m_truNumAttributes];
		for(int i=0;i<m_truNumAttributes;i++)
		{
			m_coefficients[i] = 0;
		}
		setAlpha(trainingData);
		m_coefficients = gradientDescent(trainingData);
	}
	
	private void setAlpha(Instances data) throws Exception 
	{
		double minimal_calculated_error = 999;
		double min_alpha=1;
		for(int i=-17;i<=2;i++)
		{
			m_alpha = Math.pow(3, i);
			m_coefficients = gradientDescent_alpha_phase(data);
			if(calculateSE(data) < minimal_calculated_error)
			{
				min_alpha = m_alpha;
				minimal_calculated_error = calculateSE(data);
			}
		}
		m_alpha = min_alpha;
	}
	
	/**
	 * An implementation of the gradient descent algorithm iterating 20,000 times in order
	 * to set alpha correctly for the set alpha phase of the gradient descent algorithm.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent_alpha_phase(Instances trainingData)
			throws Exception 
	{
		// Will be used to calculate the partial derivatives with respect to each theta.
		double innerProduct=0;
		// Initialize an array of temporary coefficients place holders to perform simultaneous updates.
		double temp_coefficients[] = new double[m_coefficients.length];
		for(int l=0;l<20000;l++)
		{
			for(int j=0;j<m_truNumAttributes; j++)
			{
				for(int d=0;d<trainingData.numInstances();d++)
				{
					// Each inner products sums  the first coefficient, m_coefficients[0].
					innerProduct = m_coefficients[0];
					for(int k=1;k<m_truNumAttributes; k++)
					{
						// Sum the inner product of each coefficients with the relevant instance coefficient.
						innerProduct+=m_coefficients[k]*trainingData.get(d).value(k-1);
					}
					// Substract the y value of the d'th instance.
					innerProduct-=trainingData.get(d).value(m_ClassIndex);
					if(j>0)
						// Multiply the relevant coefficient of the d'th instance.
						innerProduct *= trainingData.get(d).value(j-1);
				}
				// Simultaneous update the j'th theta's place holder, temp_coefficients[j]
				temp_coefficients[j]= m_coefficients[j] -  ( (innerProduct*m_alpha)/trainingData.numAttributes() );
			}
			for(int j=0;j<m_truNumAttributes; j++)
				// update the coefficients, theta's after each round of the algorithm.
				m_coefficients[j] = temp_coefficients[j];
		}
		return m_coefficients;
	}
	
	
	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData)
			throws Exception 
	{
		// Initialize an array of temporary doubles to perform simultaneous updates.
		double temp_coefficients[] = new double[m_coefficients.length];
		// Initialize the coefficients.
		for(int i=0;i<m_truNumAttributes;i++)
			m_coefficients[i] = 0;
		// Will be used to calculate the partial derivatives accordingly to each attribute.
		double innerProduct=0;
		// initialize the error function for the first iteration.
		double last_squaredError = 100;
		
		// Test if ð??½(ð??ƒ)decreased from the last iteration less than some small value, like 0.003
		while(Math.abs( (last_squaredError - calculateSE(trainingData)) ) > 0.003)
		{
			for(int j=0;j<m_truNumAttributes; j++)
			{
				for(int d=0;d<trainingData.numInstances();d++)
				{
					// Each inner products sums  the first coefficient, m_coefficients[0].
					innerProduct = m_coefficients[0];
					for(int k=1;k<m_truNumAttributes; k++)
						// Sum the inner product of each coefficients with the relevant instance coefficient.
						innerProduct+=m_coefficients[k]*trainingData.get(d).value(k-1);
					// Substract the y value of the d'th instance.
					innerProduct-=trainingData.get(d).value(m_ClassIndex);
					if(j>0)
						// Multiply the relevant coefficient of the d'th instance.
						innerProduct*= trainingData.get(d).value(j-1);
				}
				// Simultaneous update the j'th theta's place holder, temp_coefficients[j]
				temp_coefficients[j]= ( m_coefficients[j] - ( (innerProduct*m_alpha) / (trainingData.numAttributes() ) ) );
			}
			// Calculate the error function for the last round's thetas to compare the error change.
			last_squaredError = calculateSE(trainingData);
			for(int j=0;j<m_truNumAttributes; j++)
				// update the coefficients, theta's after each round of the algorithm.
				m_coefficients[j] = temp_coefficients[j];
		}
		return m_coefficients;
	}
	
	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception 
	{
		double innerProduct = 0;
		for (int i = 1; i < m_truNumAttributes; i++)
			innerProduct += instance.value(i-1)*m_coefficients[i];
		innerProduct += m_coefficients[0];
		return innerProduct;
	}
	
	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param testData
	 * @return
	 * @throws Exception
	 */
	public double calculateSE(Instances data) throws Exception 
	{
		double squared_loss=0;
		for(int j=0;j<data.numInstances(); j++)
			squared_loss+= Math.pow( ( regressionPrediction(data.get(j))  - data.get(j).value(m_ClassIndex) ) , 2);
		squared_loss/= data.numInstances();
		squared_loss/=2;
		if (Double.isNaN(squared_loss)) {
			squared_loss= Double.MAX_VALUE;
			}
		return squared_loss;
	}
    
    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
}
