# smartCH4 methane  prediction API
Predict the methane production of a biogas plant using the smartCH4 prediction model

# Installation
Install a python virtual environment and required packages
`python3 -m venv venv`
`source venv/bin/activate`
`pip install darts fastapi`

# Execution
For development
`fastapi dev api.py`

For production
`fastapi run api.py --port xxxx`

For production it's better to create a linux systemd service. See `example.service` file. Of course there other ways for deployment see https://fastapi.tiangolo.com/deployment/
