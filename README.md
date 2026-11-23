# Experiment 6: ARIMA Model for Forecasting

## Aim

To evaluate predictive models using performance metrics.

## Algorithm

1.  **START**
2.  Prepare stationary time series.
3.  Identify ARIMA -- (p, d, q) parameters using ACF/PACF.
4.  Fit the ARIMA model on the data.
5.  Forecast future periods and plot the results.
6.  Evaluate forecast accuracy.
7.  **STOP**



## Program

    import numpy as np
    import matplotlib.pyplot as plt
    
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # Generate stationary time series data
    np.random.seed(42)
    
    data = [0]
    
    for i in range(1, 100):
        value = 0.6 * data[i-1] + np.random.normal(0, 1)
        data.append(value)
    
    # Convert into NumPy array
    data = np.array(data)
    
    # Split training and testing data
    train = data[:80]
    test = data[80:]
    
    # Plot ACF and PACF
    plot_acf(train, lags=20)
    plt.title("Autocorrelation Function (ACF)")
    plt.show()
    
    plot_pacf(train, lags=20)
    plt.title("Partial Autocorrelation Function (PACF)")
    plt.show()
    
    # Fit ARIMA model
    model = ARIMA(train, order=(1, 0, 1))
    model_fit = model.fit()
    
    # Forecast future values
    forecast = model_fit.forecast(steps=len(test))
    
    # Evaluate model
    mae = mean_absolute_error(test, forecast)
    mse = mean_squared_error(test, forecast)
    rmse = np.sqrt(mse)
    
    print("ARIMA Model Evaluation")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    
    # Plot actual and predicted values
    plt.figure(figsize=(10, 5))
    
    plt.plot(range(len(train)), train, label="Training Data")
    plt.plot(range(80, 100), test, label="Actual Values")
    plt.plot(range(80, 100), forecast, label="Forecast Values")
    
    plt.title("ARIMA Time Series Forecasting")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.show()

## Output

     /tmp/ipykernel_508/1894848338.py:27: FutureWarning: adfuller currently returns a plain tuple whose length depends on the store and autolag arguments. In release 0.16 or after July 2027, whichever is later, the default behavior will switch to always returning an ADFullerResult. Set result_object=True to switch now, or result_object=False to keep the current behavior and silence this warning.
      result = adfuller(data)
    ADF Statistic: -11.494310031913571
    P-value: 4.654276370360886e-21
                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                   80
    Model:                 ARIMA(1, 0, 0)   Log Likelihood                -277.211
    Date:                Wed, 18 Nov 2026   AIC                            560.423
    Time:                        13:15:00   BIC                            567.569
    Sample:                             0   HQIC                           563.288
                                     - 80                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         29.2987     38.403      0.763      0.446     -45.970     104.567
    ar.L1          0.9860      0.023     42.329      0.000       0.940       1.032
    sigma2        57.2603      5.221     10.968      0.000      47.028      67.493
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):               223.99
    Prob(Q):                              0.95   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.31   Skew:                            -1.80
    Prob(H) (two-sided):                  0.00   Kurtosis:                        10.36
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    MAE: 4.914708682516378
    MSE: 38.16296859775095
    RMSE: 6.177618359671545



## Result
Thus, the ARIMA time-series forecasting model was successfully implemented using Python. A stationary time series was prepared, ARIMA parameters were selected, and future values were forecasted. Model performance was evaluated using MAE, MSE and RMSE.
