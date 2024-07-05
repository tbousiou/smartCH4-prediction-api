from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from darts.models import RNNModel
from darts import TimeSeries
import numpy as np


class InputData(BaseModel):
    past: List[List[float]]
    future: List[float]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "past": [[8.87, 1, 6.91], [11.02, 1, 10.83], [9.46, 1, 10.89], [11.93, 1, 10.95], [10.89, 1, 10.96], [11.01, 1, 10.98]],
                    "future": [10.60, 1]
                }
            ]
        }
    }


app = FastAPI(
    title="smartCH4 Methane Prediction API",
    description="Predict the methane production of a biogas plant using the smartCH4 prediction model"
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>ssmartCH4 Methane Prediction API</title>
        </head>
        <body>
            <h1>smartCH4 Methane Prediction API</h1>
            <p>Predict the methane production of a biogas plant using the smartCH4 prediction model</p>
            <p>Go to <a href="docs">/docs</a> to see the API documentation.</p>
            <p><a href="https://github.com/tbousiou/smartCH4-prediction-api">GitHub repository</a></p>
        </body>
    </html>
    """


@app.post("/predict/")
def make_prediction(
        input: InputData,
        days_ahead: int = Query(
            6,
            description="The number of days ahead to make the prediction for. The future covariates will be repeated for each day ahead."
        )) -> List[float]:
    """
    Make a prediction using the smartCH4 prediction model.
    """

    # Transform request data into the format your model expects (Daets TimeSeries objects)

    # First transform the input data into numpy arrays
    past = np.array(input.past)
    future = np.array(input.future)

    # Target is the last column of the past data
    target = TimeSeries.from_values(np.array(past[:, -1]))
    # Covariates are all columns except the last one
    past_covs = past[:, :-1]
    # Future covariates are the same for all days ahead
    future_covs = np.tile(future, (days_ahead, 1))
    # Combine past and future covariates
    covariates = TimeSeries.from_values(np.vstack((past_covs, future_covs)))

    # Load the model and scalers from training
    model = RNNModel.load('smartch4_model')

    import joblib
    scaler_target = joblib.load('scaler_target.save')
    scaler_covariate = joblib.load('scaler_covariate.save')

    # Scale real data
    target_scaled = scaler_target.fit_transform(target)
    covariate_scaled = scaler_covariate.transform(covariates)

    # Make predictions with the model
    predictions = model.predict(
        n=days_ahead,  series=target_scaled, future_covariates=covariate_scaled
    )

    # Inverse scale the predictions
    predictions_real = scaler_target.inverse_transform(predictions)

    return predictions_real.values()
