import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cdist
import time
import functools
def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

# The start and end dates of U.S. daylight savings time (DST) matches up with the transition points in the training data.
# source: https://en.wikipedia.org/wiki/History_of_time_in_the_United_States
DST_START_DTS = ["3/8/26", "3/9/25", "3/10/24", "3/12/23", "3/13/22", "3/14/21", "3/8/20", "3/10/19", "3/11/18", "3/12/17",
                 "3/13/16", "3/8/15", "3/9/14", "3/10/13", "3/11/12", "3/13/11", "3/14/10", "3/8/09", "3/9/08", "3/11/07",
                 "4/2/06", "4/3/05", "4/4/04", "4/6/03", "4/7/02", "4/1/01", "4/2/00"]

DST_END_DTS = ["11/1/26", "11/2/25", "11/3/24", "11/5/23", "11/6/22", "11/7/21", "11/1/20", "11/3/19", "11/4/18", "11/5/17",
               "11/6/16", "11/1/15", "11/2/14", "11/3/13", "11/4/12", "11/6/11", "11/7/10", "11/1/09", "11/2/08", "11/4/07",
               "10/29/06", "10/30/05", "10/31/04", "10/26/03", "10/27/02", "10/28/01", "10/29/00"]

# Create dictionaries with the days of the year when DST ends and starts for the years 2000-2026.
dstEndDayOfYearDict = {dt.year:dt.dayofyear for dt in pd.to_datetime(DST_END_DTS)}
dstStartDayOfYearDict = {dt.year:dt.dayofyear for dt in pd.to_datetime(DST_START_DTS)}

# Outside of dailight savings, the first daily price sample in data.csv is at 15:05:00.
FIRST_SEC_OUT_OF_DST = 60 * (15 * 60 + 5)
# Within daylight savings, the first daily price sample in data.csv is at 14:05:00.
FIRST_SEC_IN_DST = 60 * (14 * 60 + 5)

def shiftBackArray(arrayIn, fillFirstValue=np.nan):
    """
    Back-shift a given array.

    Args:
        arrayIn (array_like): vector to back-shift
        fillFirstValue (float): value to put in first entry of returned vector

    Returns:
        arrayOut (np.array): back-shifted vector

    """
    arrayOut = np.empty_like(arrayIn)
    arrayOut[0] = fillFirstValue
    arrayOut[1:] = np.where(np.isnan(arrayIn[1:]), np.nan, arrayIn[:-1])
    return arrayOut

def getData(dataFilename, numLagsDict={'x':25, 'y':25}, rowsPerDay=1410):
    """
    Read and process the price data .csv file at the given path.
    Create date and time variables that provide consisitency across the different days
        and daylight savings time periods in the data.
    Create specified number of lagged price difference variables.

    Args:
        dataFilename (str): path to data (.csv)
        numLagsDict (Dict[str, int]): dictionary specifying number of lagged diffences
            to create for (x and y) price variables
        rowsPerDay (int): number of rows (samples) per trading day 

    Returns:
        df (pd.DataFrame): dataframe with processed data

    """
    # Read in .csv file.
    df = pd.read_csv(dataFilename)
    
    # Create date and time variables.
    df['dt'] = np.repeat(np.datetime64('1970-01-01T00:00:00'), len(df)) + df.timestamp.values * np.timedelta64(1, 'ms')
    df['dayOfYear'] = df['dt'].dt.dayofyear
    df['secondInDay'] = 3600 * df['dt'].dt.hour + 60 * df['dt'].dt.minute + df['dt'].dt.second
    # Create a consistent daily time index variable.
    df['timeID'] = np.where(
            # condition for whether a given row falls outside of Daylight Savings Time (DST)
        (df['dayOfYear'] < df['dt'].dt.year.map(dstStartDayOfYearDict)) | (df['dayOfYear'] > df['dt'].dt.year.map(dstEndDayOfYearDict)),
        (df['secondInDay'] - FIRST_SEC_OUT_OF_DST) // 10,
        (df['secondInDay'] - FIRST_SEC_IN_DST) // 10
    )

    # Create variable for the number of steps forward to the yprice row that the returns values are based on. This is 60 when
    # price data is available ten minutes forward; otherwise it's the number of steps to the final row of the current day.
    df['stepsToReturnsPrice'] = np.where(rowsPerDay - 1 - df['timeID'] > 60, 60, rowsPerDay - 1 - df['timeID'])

    # Create variables for lagged differences in xprice and yprice.
    for var, numLags in numLagsDict.items():
        # First compute the difference between each row's current price and the previous price (from 10 seconds before).
        # If there is no previous price available then set variable to zero.
        df[var+'DiffLag0'] = np.where((df['timeID'] > numLags - 1) & (df['stepsToReturnsPrice'] > 0),
                               df[var+'price'] - shiftBackArray(df[var+'price'].values, fillFirstValue=0.), 0.)
        # Next create the lagged differences variables.
        for lag in range(1, numLags):
            df[var+'DiffLag'+str(lag)] = np.where((df['timeID'] > numLags-1) & (df['stepsToReturnsPrice'] > lag),
                                                  shiftBackArray(df[var+'DiffLag'+str(lag-1)], fillFirstValue=0.), 0.)
    return df


def estimateForecastWindow(df, startTrainIndex, trainWindowSize, forecastHorizon,
             covariates, ridgeModel, trainToPredictOffset=60):
    """
    Fit weighted ridge regression model with specified covariates to the specified
        training window rows in df.
    Create training weights based on the distance between values in the training rows
        and the average values across the training window.
    Update the 'prediction' column of df with predictions based on the fitted ridge
        model and the covariate values within the specified prediction range.

    Args:
        df (pd.DataFrame): data to use for training and to update with predictions
        startTrainIndex (int): index at which training window starts
        trainWindowSize (int): number of rows to include in training window
        forecastHorizon (int): number of forward predictions to make
        covariates (List[str]): covariates to include in ridge regression model
        ridgeModel (sklearn.linear_model.Ridge): sklearn ridge regession model
            to fit and predict from
        trainToPredictOffset (int): periods (rows) between the end of the training window 
            and the first prediction

    Returns:
        None

    """
    # Set index specifying the training window (along with startTrainIndex).
    endTrainIndex = startTrainIndex + trainWindowSize - 1
    
    # Set indices corresponding to the prediction horizon.
    startPredictIndex = endTrainIndex + trainToPredictOffset
    endPredictIndex = startPredictIndex + forecastHorizon - 1

    # Compute 'center' of values across training window - the mean of the covariates and returns.
    trainRowsCenter = df.loc[startTrainIndex:endTrainIndex, covariates+['returns']].mean(axis=0).values
    
    # Compute 'scale' vector capturing variability across the training window.
    # This contains the sample standard deviations for each covariate along with the sample standard deviation of the returns
    # column divided by .5 times the number of covariates (putting increased weight on the deviations of the returns values).
    # The ratio of total weight put on the deviations of the covariates to the weight put on the returns deviations is 2:1.
    trainRowsScale = df.loc[startTrainIndex:endTrainIndex, covariates+['returns']].std(axis=0).values
    trainRowsScale[-1] /= (len(covariates) / 2)

    # Use the scipy.spatial.distance.cdist function to compute standardized (weighted) euclidean distances
    # between the covariates & returns values and the trainRowsCenter for each training row.
    distances = cdist(np.expand_dims(trainRowsCenter,0),
                      df.loc[startTrainIndex:endTrainIndex, covariates+['returns']].values,
                      metric='seuclidean', V=trainRowsScale)[0]

    # Compute training weights from these distances using the Gaussian kernel function, phi(x) = exp(-.5 * (x-center)^2 / width)
    # with center equal to the average of the distances, and width equal to 3x the number of covariates.
    # Note this width is equal to the theoretical variance of the computed distances if the covariates and returns were all
    # independent and normally distributed
    trainWeights = np.exp(-np.square(distances - np.mean(distances)) / (6 * len(covariates)))

    # Estimate model coefficients with weighted ridge regression using the sklearn package.
    # Note: the fit is based only on data within the training window
    ridgeModel.fit(df.loc[startTrainIndex:endTrainIndex, covariates].values,
                   df.loc[startTrainIndex:endTrainIndex, 'returns'].values,
                   sample_weight=trainWeights)
    
    # Use the estimated model to make forward predictions.
    # this is an in-place operation, directly updating the 'prediction' column of the input dataframe (df).
    df.loc[startPredictIndex:endPredictIndex, 'prediction'] \
        = ridgeModel.predict(df.loc[startPredictIndex:endPredictIndex, covariates].values)

def modelEstimate(trainingFilename):
    """
    **All model fitting takes place along with prediction in the modelForecast function.**
    This function creates and returns the dictionary 'parameters' containing trainingFilename
    along with data parameters and hyperparameters for the model (see descriptions below).

    Args:
        trainingFilename (str): path to training data. The data will
            be in the same format as the supplied `data.csv` file

    Returns:
        parameters (Dict[str, any]): the parameters needed to pass to modelForecast
            
    """
    parameters = {
            # specified path to training data
        'trainingFilename':trainingFilename,
            # number of rows to include in training window and estimate model with
            # equal to: rowsPerDay * 25 - trainToPredictOffset + 1
            # that is, the training window size is (just short of) 25 days worth of price data
        'trainWindowSize':1410 * 25 - 60 + 1,
            # number of forward predictions for each training window and estimated model
            # equal to: rowsPerDay // 2
            # that is, for each rolling window of data, a half-day worth of forecasts are made
        'forecastHorizon':1410 // 2,
            # hyperparameters specifying how many lagged price differences of the X and Y instruments
            # to include in the regression model
        'numLagsDict':{'x':25, 'y':22},
            # the returns column (variable to predict) is (for the most part) based on prices ten minutes from now,
            # so the number of rows between the last training row and first predicted row needs to be (at least) 60
            # (60 x ten seconds = 10 min)
        'trainToPredictOffset':60,
            # hyperparamter controlling l2 regrularization in ridge regression
        'alpha':1.3,
            # there are 1410 rows per day in data.csv
        'rowsPerDay':1410
    }
    return(parameters)

@timer
def modelForecast(testFilename, parameters):
    """
    Predict returns using a fitted model.

    Args:
        testFilename (str): path to test data. The data will be in
            the same format as the supplied `data.csv` file
        parameters (Dict[str, any]): data parameters and hyperparameters for forecasting, see
            descriptions in the modelEstimate function above.
            
    Returns:
        forecast (np.array): vector of predictions.

    """
    # Read in and process training and test data.
    trainData = getData(parameters['trainingFilename'],
                        numLagsDict=parameters['numLagsDict'],
                        rowsPerDay=parameters['rowsPerDay'])
    testData = getData(testFilename,
                        numLagsDict=parameters['numLagsDict'],
                        rowsPerDay=parameters['rowsPerDay'])
    firstTestTimestamp = testData.loc[0, 'timestamp']

    # Concatenate the last trainWindowSize - trainToPredictOffset - 1 rows of the training data with the test data.
    # The training window will 'roll' across this training set 'tail', using these rows to estimate the model
    # coefficients and make predictions for the first trainWindowSize observations in the test set.
    df = pd.concat([trainData.tail(parameters['trainWindowSize'] + parameters['trainToPredictOffset'] - 1), testData])
    df.index = np.arange(len(df))

    # Intialize the prediction column.
    df['prediction'] = 0.
    # Intialize ridge regression model with specified alpha (hyperparameter controlling l2 regrularization).
    ridgeModel = Ridge(alpha=parameters['alpha'], fit_intercept=False)

    # Perform the rolling window regression by incrementally advancing the training window across the concatenated dataframe.
    # At each step (while there are predictions to make), estimate the model with the trainWindowSize rows in the training window,
    # make forecastHorizon forward predictions and 'roll' the training window forward by forecastHorizon steps.
    startTrainIndex = 0
    while startTrainIndex + parameters['trainWindowSize'] + parameters['trainToPredictOffset'] <= len(df):
        estimateForecastWindow(
            df, startTrainIndex, parameters['trainWindowSize'], parameters['forecastHorizon'],
            covariates=[var+'DiffLag'+str(lag) for var,numLag in parameters['numLagsDict'].items() for lag in range(numLag)],
            ridgeModel=ridgeModel,
            trainToPredictOffset=parameters['trainToPredictOffset']
        )
        startTrainIndex += parameters['forecastHorizon']

    df['squaredError'] = np.square(df['prediction'] - df['returns'])
    print(sum(df['squaredError'] - np.square(df['returns'])))
    print(np.mean(df['squaredError'][df['timestamp'] >= firstTestTimestamp]))
    print(np.mean(np.square(df['returns'])[df['timestamp'] >= firstTestTimestamp]))

    return df['prediction'][df['timestamp'] >= firstTestTimestamp].values
