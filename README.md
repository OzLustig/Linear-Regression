# Machine Learning from Data – IDC

# HW1 – Linear Regression

***** This assignment can be submitted in pairs (and pairs only!).**

In this assignment you will implement a Linear Regression algorithm that uses Gradient Descent.
In order to do so you need to first install WEKA:

1. Download WEKA from this link.
2. Run the .exe file that you downloaded and install WEKA.
Prepare your Eclipse project:
1. Create a project in eclipse called HomeWork1.
2. Create a package called HomeWork1.
3. Move the LinearRegression.java and MainHW1.java that you downloaded from the
Moodle into this package.
4. Add WEKA to the project:
a. Right click on the java library that is used and click Build path -> configure build path.
b. On right hand side click ‘add external jar’.
c. Select the file weka.jar that is in the Weka program directory that was created during
the installation.
Implement the Linear Regression algorithm:
1. In LinearRegression.java fill in the code for the methods:
a. private double[] gradientDescent(Instances trainingData) – were you will implement
the gradient descent algorithm learned in class. The method should return the weights
of a linear regression predictor.
b. public double regressionPrediction(Instance instance) – which takes as an input an
instance of data and uses the coefficients that were calculated already and returns the
linear regression prediction on that instance.
c. public double calculateSE(Instances data) – which takes as an input the whole data
instances and returns the average squared error of that predictor on the data. The
average squared error is the total squared error divided by the number of data
instances.
Choosing alpha:
Create a for loop with a variable i which goes from -17 up to 2. Set alpha equal to 3^i and
run gradient descent with this value with a fixed number of iterations, say 20,000.
Calculate the error on the training data. So now you ran gradient descent with many


```
different alphas. The one which gave you the lowest error, will be good to use. Now you
can run the gradient descent algorithm with a stopping condition.
Stopping condition:
For every 100 iterations check the current error and compare it to the error you calculated
100 iterations ago. If the difference between the errors is very small, say smaller than
0.003, then stop, otherwise continue.
```
2. Now you can use the methods you wrote in order to predict the wind speed in MAL station
    based on several features (date & 11 wind speed from other stations). Included in the folder
    are training and test data files which include information on many wind measurements, for
    each measurement – all explained in the text file (open it with your favorite text editor).
3. In your main method:
    a. Load the wind training data using the loadData(String fileName) provided.
    b. Train your linear regression predictor by running the public void
       buildClassifier(Instances data).
    c. Load the wind testing data using the loadData(String fileName) provided.
    d. Calculate the squared error on the test data.
    e. Create a file hw1.txt
       Display the weights and the error you calculated in the following format:
       "The weights are: <weights>".
       "The error is <error>".
       The weights should be in the same order of the features in the data file.
       Check: which of the features have positive weights and which have negative weights?
       Think: what does the sign of the weights mean? Do the results make sense?
    f. In the file you created, add an explanation of how you decided to stop iterating and
       also how you chose your alpha.

You should hand in a LinearRegression.java, MainHW1.java and hw1.txt files which the grader
will use to test your implementation. All of these files should be placed in a
hw_1_##id1##_##id2##.zip folder with the id of both of the members of the group.

EXAMPLE hw1.txt file:

## The weights are: <weights>

## The error is <error>

## My explanation of how I chose alpha and what my stopping condition is.

*** Submitting in groups on Moodle does not work. Please only submit one zip folder per pair


