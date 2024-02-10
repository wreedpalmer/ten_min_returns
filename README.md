The approach I use to predict the returns of instrument Y is a rolling window, weighted (linear) ridge regression model
    that includes, as its covariates, lagged price differences of the X and Y instruments.

The linear model for the returns may be written as follows:<br />
&nbsp;    returns[t] = xBeta[0] * 𝝙xprice[t] + xBeta[1] * 𝝙xprice[t-1] + ... + xBeta[xLags[t]-1] * 𝝙xprice[t-xLags[t]+1]<br />
&nbsp;&nbsp;             + yBeta[0] * 𝝙yprice[t] + yBeta[1] * 𝝙yprice[t-1] + ... + yBeta[yLags[t]-1] * 𝝙yprice[t-yLags[t]+1]

Where 𝝙 is the differencing operator, so that:<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   		𝝙xprice[t] := xprice[t] - xprice[t-1]<br />
	and&nbsp;&nbsp;  	𝝙yprice[t] := yprice[t] - yprice[t-1].

And where xLags[t] and yLags[t] are given by:<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   		xLags[t] := min(numLagsDict['x'], stepsToReturnsPrice[t])<br />
    and&nbsp;&nbsp; 	    yLags[t] := min(numLagsDict['y'], stepsToReturnsPrice[t]).

Note that numLagsDict['x'] and numLagsDict['y'] are specified model hyperparameters and stepsToReturnsPrice[t] is equal to
    60 when there is price data available ten minutes (60 steps forward) from t, and otherwise is equal to the number of
    steps forward to the latest available price (within the given day).

The model parameters xBeta[0],xBeta[0],...,xBeta[numLagsDict['x']-1] and yBeta[0],yBeta[0],...,yBeta[numLagsDict['y']-1]
    are fit by rolling window, weighted ridge regression, and these estimates are used to predict the returns of instrument
    Y at time t. The estimated coefficients vary with t, and depend on the training window size and forecast horizon.

The parameter alpha controlling the l2 regularization in the ridge regression, along with the training window size, the
    forecast horizon (the number of forward predictions to make from each rolling window and estimated model), and
    the numbers of lagged price differences of the X and Y instruments to include in the regression model (numLagsDict) are
    all hyperparameters that need to be specified/tuned in my approach.

I tune these hyperparameters based on train.csv and make the following specifications for my model:<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;		alpha=1.3<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;		numLagsDict['x']=25<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;		numLagsDict['y']=22<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;		trainWindowSize=35191 &nbsp;&nbsp;&nbsp;   (25 days worth of training data)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;		forecastHorizon=705   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   (a half-day worth of forecasts)	

I use a weighted ridge regression to limit the influence of outlier rows within a given training window on the estimated model
    coefficients.

For each rolling window, the training weights used in the weighted ridge regression are based on a scaled Euclidean distance
    between the covariate and returns values in the training rows and the average values across the training window. I
    compute a 'scale' vector that captures variability across the training window and contains the sample standard deviations
    for each covariate along with the sample standard deviation of the returns column divided by .5 times the number of
    covariates in the model. This puts increased weight on the deviations of the returns values.

I compute the training weights from these distances using the Gaussian kernel function:<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;		phi(x) = exp(-.5 * (x-center)^2 / width),<br />
    with center equal to the average of the distances, and width equal to 3x the number of covariates.

My rolling window regression approach is implemented via the modelForecast and estimateForecastWindow functions in solution.py.
    In modelForecast I concatenate the tail of the training data in train.csv with the test data in test.csv, and I use the final
    rows from train.csv to estimate the model coefficients and make predictions for the initial observations in the test set.

I perform the rolling window regression by incrementally advancing the training window across the concatenated dataframe.
    At each step (while there are still predictions to make), I use my estimateForecastWindow function to estimate the model
    based on the data in training window and make forward predictions. Then I 'roll' the training window forward.

Note: Each model fit (along with its associated forward predictions) is based only on data within the current training window and
    does not include any future information / price data from the associated forward prediction rows.
