# import the necessary packages

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from Code import Models as Model


app = dash.Dash(__name__)
# app.css.append_css({'external_url': 'static/custom.css'})
app.config['suppress_callback_exceptions'] = True

# Functions to be passed into html
def about():
    return html.Div(className='content', children=[
        html.P(
            children=[
                """
                According to the Centers for Disease Control and Prevention (CDC), 
                car accidents are one of the leading causes of death in the U.S., 
                causing around thirty five thousands of deaths per year.
                """
                """
                While there is some understanding of the factors that contribute 
                to accident severity, there is a need for further exploration as 
                to how these factors together influence accident severity.
                """
                """
                In our research we intend to use the different factors associated 
                with car accidents to predict the severity of an accident. Thus, 
                We built several Machine Learning and Statistical Baseline Models 
                and compare them to the performance of LSTM in order to predict 
                Traffic Accident Severity and accident risk.
                """
            ]), html.Br(),
        ],)


def reference():
    return html.Div([
        'Reference: ',
        html.A('Paper',
        href='https://www.google.com/)')
        ])


def link_github():
    return html.Div([
        'For a look into this project, please visit the '
        'original repository ',
        html.A('here', href='https://github.com/123porcristina/Capstone'),
        '.'
    ])


def severity():
    return html.Div(className='app-controls-block', children=[
        html.Div(className='app-controls-name', children='Actions'),
        html.Hr(),
        html.Div(className="'app-controls-block'", children=[

            html.Div(
                className='app-controls-name-label',
                children='1st Crash Type:'
            ),
            dcc.Dropdown(
                id="nrows-Crash-Type",
                options=[
                    {'label': 'ANGLE', 'value': 'ANGLE'},
                    {'label': 'FIXED OBJECT', 'value': 'FIXED OBJECT'},
                    {'label': 'HEAD ON', 'value': 'HEAD ON'},
                    {'label': 'OTHER NONCOLLISION', 'value': 'OTHER NONCOLLISION'},
                    {'label': 'OTHER OBJECT', 'value': 'OTHER OBJECT'},
                    {'label': 'OVERTURNED', 'value': 'OVERTURNED'},
                    {'label': 'TURNING', 'value': 'TURNING'},
                ],
                placeholder='Select...',
            ), html.Br(), html.Br(),

            html.Div(
                className='app-controls-name-label',
                children='Crash Hour:'
            ),
            dcc.Dropdown(
                id="nrows-Crash-Hour",
                options=[
                    {'label': '0', 'value': '0'},
                    {'label': '1', 'value': '1'},
                    {'label': '2', 'value': '2'},
                    {'label': '3', 'value': '3'},
                    {'label': '4', 'value': '4'},
                    {'label': '5', 'value': '5'},
                    {'label': '6', 'value': '6'},
                    {'label': '7', 'value': '7'},
                    {'label': '8', 'value': '8'},
                    {'label': '9', 'value': '9'},
                    {'label': '10', 'value': '10'},
                    {'label': '11', 'value': '11'},
                    {'label': '12', 'value': '12'},
                    {'label': '13', 'value': '13'},
                    {'label': '14', 'value': '14'},
                    {'label': '15', 'value': '15'},
                    {'label': '16', 'value': '16'},
                    {'label': '17', 'value': '17'},
                    {'label': '18', 'value': '18'},
                    {'label': '19', 'value': '19'},
                    {'label': '20', 'value': '20'},
                    {'label': '21', 'value': '21'},
                    {'label': '22', 'value': '22'},
                    {'label': '23', 'value': '23'},
                ],
                placeholder='Select...',
            ), html.Br(), html.Br(),

            html.Div(
                className='app-controls-name-label',
                children='Crash Day of Week:'
            ),
            dcc.Dropdown(
                id="nrows-day-week",
                options=[
                    {'label': 'Sunday', 'value': '1'},
                    {'label': 'Monday', 'value': '2'},
                    {'label': 'Tuesday', 'value': '3'},
                    {'label': 'Wednesday', 'value': '4'},
                    {'label': 'Thursday', 'value': '5'},
                    {'label': 'Friday', 'value': '6'},
                    {'label': 'Saturday', 'value': '7'},
                ],
                placeholder='Select...',
            ), html.Br(), html.Br(),

            html.Div(
                className='app-controls-name-label',
                children='Crash Month:'
            ),
            dcc.Dropdown(
                id="nrows-month",
                options=[
                    {'label': 'January', 'value': '1'},
                    {'label': 'February', 'value': '2'},
                    {'label': 'March', 'value': '3'},
                    {'label': 'April', 'value': '4'},
                    {'label': 'May', 'value': '5'},
                    {'label': 'June', 'value': '6'},
                    {'label': 'July', 'value': '7'},
                    {'label': 'August', 'value': '8'},
                    {'label': 'September', 'value': '9'},
                    {'label': 'October', 'value': '10'},
                    {'label': 'November', 'value': '11'},
                    {'label': 'December', 'value': '12'},
                ],
                placeholder='Select...',
            ), html.Br(), html.Br(),

            html.Div(
                className='app-controls-name-label',
                children='Contributory Cause:'
            ),
            dcc.Dropdown(
                id="nrows-contributory",
                options=[
                    {'label': 'ANIMAL', 'value': 'ANIMAL'},
                    {'label': 'DISREGARDING OTHER TRAFFIC SIGNS',
                     'value': 'DISREGARDING OTHER TRAFFIC SIGNS'},
                    {'label': 'DISTRACTION', 'value': 'DISTRACTION'},
                    {'label': 'DRIVING ON WRONG SIDE/WRONG WAY',
                     'value': 'DRIVING ON WRONG SIDE/WRONG WAY'},
                    {'label': 'DRIVING SKILLS/KNOWLEDGE/EXPERIENCE',
                     'value': 'DRIVING SKILLS/KNOWLEDGE/EXPERIENCE'},
                    {'label': 'IMPROPER BACKING', 'value': 'IMPROPER BACKING'},
                    {'label': 'IMPROPER LANE USAGE', 'value': 'IMPROPER LANE USAGE'},
                    {'label': 'OTHER', 'value': 'OTHER'},
                    {'label': 'WEATHER', 'value': 'WEATHER'},
                    {'label': 'VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)',
                     'value': 'VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)'},
                    {
                        'label': 'UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)',
                        'value': 'UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)'},
                    {'label': 'UNABLE TO DETERMINE', 'value': 'UNABLE TO DETERMINE'},
                ],
                placeholder='Select...',
            ), html.Br(), html.Br(),

            html.Div(
                className='app-controls-name-label',
                children='Posted Speed:'
            ),
            dcc.Input(
                id="nrows-speed",
                type="number",
                value=30,
                name="number of rows",
                min=30,
                step=1,
            ), html.Br(), html.Br(),

            html.Div(
                className='app-controls-name-label',
                children='Traffic Control:'
            ),
            dcc.Dropdown(
                id="nrows-traffic",
                options=[
                    {'label': 'None', 'value': 'None'},
                    {'label': 'Traffic Signal', 'value': 'TrafficSignal'},
                    {'label': 'Other Control', 'value': 'OtherControl'},
                ],
                placeholder='Select...',
            ), html.Br(), html.Br(),

            html.Div(
                className='app-controls-name-label',
                children='Weather:'
            ),
            dcc.Dropdown(
                id="nrows-weather",
                options=[
                    {'label': 'Clear', 'value': 'Clear'},
                    {'label': 'Snow', 'value': 'Snow'},
                    {'label': 'Cloudy', 'value': 'Cloudy'},
                    {'label': 'Rain', 'value': 'Rain'},
                    {'label': 'Other', 'value': 'OtherForecast'},
                ],
                placeholder='Select...',
            ), html.Br(), html.Br(),

            html.Div(
                className='app-controls-name-label',
                children='Road Surface:'
            ),
            dcc.Dropdown(
                id="nrows-road",
                options=[
                    {'label': 'Dry', 'value': 'Dry'},
                    {'label': 'Snow', 'value': 'Snow'},
                    {'label': 'Wet', 'value': 'Wet'},
                    {'label': 'Other', 'value': 'OtherCondition'},
                ],

                placeholder='Select...',
            ), html.Br(), html.Br(),

            html.Div(
                className='app-controls-name-label',
                children='Sex:'
            ),
            dcc.Dropdown(
                id="nrows-sex",
                options=[
                    {'label': 'Male-Female', 'value': 'MaleFemale'},
                    {'label': 'Both Male', 'value': 'BothMale'},
                    {'label': 'Both Other', 'value': 'BothOther'},
                    {'label': 'Female Other', 'value': 'Female Other'},
                    {'label': 'Male Other', 'value': 'MaleOther'},
                ],
                placeholder='Select...',
            ), html.Br(), html.Br(),

            html.Div(
                className='app-controls-name-label',
                children='BAC:'
            ),
            dcc.Input(
                id="nrows-bac",
                type="number",
                value=1,
                name="number of rows",
                min=0,
                step=1,
            ), html.Br(), html.Br(),

            html.Div(
                className='app-controls-name-label',
                children='Age:'
            ),
            dcc.Input(
                id="nrows-age",
                type="number",
                value=1,
                name="number of rows",
                min=16,
                step=1,
            ),

            html.Br(),
            html.Br(),
            html.Button('Predict', id='btn-predict-severity', className="control-download", n_clicks=0),#n_clicks_timestamp=0),
            html.Br(),
                ]),
            ])


def risk():
    return html.Div(className='app-controls-block', children=[
        html.Div(className='app-controls-name', children='Actions'),
        html.Hr(),
        html.Div(className="'app-controls-block'", children=[

            html.Div(
                className='app-controls-name-label',
                children='Crash Hour:'
            ),
            dcc.Dropdown(
                id="nrows-Crash-Hour-risk",
                options=[
                    {'label': '0', 'value': '0'},
                    {'label': '1', 'value': '1'},
                    {'label': '2', 'value': '2'},
                    {'label': '3', 'value': '3'},
                    {'label': '4', 'value': '4'},
                    {'label': '5', 'value': '5'},
                    {'label': '6', 'value': '6'},
                    {'label': '7', 'value': '7'},
                    {'label': '8', 'value': '8'},
                    {'label': '9', 'value': '9'},
                    {'label': '10', 'value': '10'},
                    {'label': '11', 'value': '11'},
                    {'label': '12', 'value': '12'},
                    {'label': '13', 'value': '13'},
                    {'label': '14', 'value': '14'},
                    {'label': '15', 'value': '15'},
                    {'label': '16', 'value': '16'},
                    {'label': '17', 'value': '17'},
                    {'label': '18', 'value': '18'},
                    {'label': '19', 'value': '19'},
                    {'label': '20', 'value': '20'},
                    {'label': '21', 'value': '21'},
                    {'label': '22', 'value': '22'},
                    {'label': '23', 'value': '23'},
                ],
                placeholder='Select...',
            ), html.Br(), html.Br(),

            html.Div(
                className='app-controls-name-label',
                children='Crash Day of Week:'
            ),
            dcc.Dropdown(
                id="nrows-day-week-risk",
                options=[
                    {'label': 'Sunday', 'value': '1'},
                    {'label': 'Monday', 'value': '2'},
                    {'label': 'Tuesday', 'value': '3'},
                    {'label': 'Wednesday', 'value': '4'},
                    {'label': 'Thursday', 'value': '5'},
                    {'label': 'Friday', 'value': '6'},
                    {'label': 'Saturday', 'value': '7'},
                ],
                placeholder='Select...',
            ), html.Br(), html.Br(),

            html.Div(
                className='app-controls-name-label',
                children='Weather:'
            ),
            dcc.Dropdown(
                id="nrows-weather-risk",
                options=[
                    {'label': 'Clear', 'value': 'Clear'},
                    {'label': 'Snow', 'value': 'Snow'},
                    {'label': 'Cloudy', 'value': 'Cloudy'},
                    {'label': 'Rain', 'value': 'Rain'},
                    {'label': 'Other', 'value': 'OtherForecast'},
                ],
                placeholder='Select...',
            ),

            html.Br(),
            html.Br(),
            html.Button('Predict', id='btn-predict-risk', className="control-download", n_clicks_timestamp=0),
            html.Br(),
                ]),
            ])


# App Layout
app.layout = html.Div(
    children=[
        # Top Banner severity of accidents
        html.Div(
            className="study-browser-banner row",
            children=[
                html.H2(className="h2-title", children="Severity and Risk of Traffic Accidents in the City of Chicago"),
            ],
        ),
        ##### Tabs #####
        html.Div(id='circos-control-tabs', className='control-tabs', children=[
            dcc.Tabs(id='circos-tabs', value='what-is', children=[
                ##Tab1: about
                dcc.Tab(
                    label='About',
                    value='what-is', className='control-tab',
                    children=html.Div(className='control-tab', children=[
                        html.Div(className='content', children=[

                            html.H4(className='what-is', children="What is this project about?"),
                            about(),
                            reference(),
                            link_github(),
                            html.Br()
                        ]),
                    ], )
                ),
                ##tab 2: EDA
                dcc.Tab(
                    label='Exploratory Analysis',
                    value='data',
                    children=html.Div(className='control-tab', children=[
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children=[
                                html.H4(className="h6-title", children="Exploratory Data Analysis")
                            ]),
                            html.Hr(),
                            html.Div(className="'app-controls-block'", children=[
                                html.Br(),
                                html.Br(),
                                html.Button('Show Exploratory Data Analysis', id='btn-1', className="control-download",
                                            n_clicks_timestamp=0),
                                html.Br(),
                                html.Button('Reset Screen', id='btn-2', n_clicks_timestamp=0),
                                html.Br(),
                            ]),
                        ]),
                        html.Hr(),
                    ])
                ),
                # tab 3: Predict Severity
                dcc.Tab(
                    label='Predict Severity',
                    value='data2',
                    children=html.Div(className='control-tab', children=[
                        severity(),
                        html.Hr(),
                    ])
                ),

                ###tab 4: Predict Risk
                dcc.Tab(
                    label='Predict Risk',
                    value='data3',
                    children=html.Div(className='control-tab', children=[
                        risk(),
                        html.Hr(),
                    ])
                ),
            ])
        ]),
                # Information on the right side - do not change the identation
                html.Div(
                    className="eight columns card-left",
                    children=[
                        html.H5("Results"),
                        html.Div(
                            className="bg-white",
                            children=[
                                html.Div(id='output-video'),
                                html.Div(id='predict-severity'),
                            ],
                        ),
                    ],
                ),
    ]
)


@app.callback(Output('output-video', 'children'),
              [Input('btn-1', 'n_clicks_timestamp'),
               Input('btn-2', 'n_clicks_timestamp')])
def displayClick(btn1, btn2):
    if int(btn1) > int(btn2):
        # Shows EDA in the right side of the screen
        print("button EDA was pressed")
        return [
            html.Iframe(
                id="bla",
                src=app.get_asset_url('EDA.html'),
                height="100%",
                width='100%',
                style={'border-style': 'none'}
            ),
        ]
    elif int(btn2) > int(btn1):
        return html.Div([])
    else:
        return html.Div([])


@app.callback(Output('predict-severity', 'children'),
              [Input('nrows-Crash-Type', 'value'),
               Input('nrows-Crash-Hour', 'value'),
               Input('nrows-day-week', 'value'),
               Input('nrows-month', 'value'),
               Input('nrows-contributory', 'value'),
               Input('nrows-speed', 'value'),
               Input('nrows-traffic', 'value'),
               Input('nrows-weather', 'value'),
               Input('nrows-road', 'value'),
               Input('nrows-sex', 'value'),
               Input('nrows-bac', 'value'),
               Input('nrows-age', 'value'),
               Input('btn-predict-severity', 'n_clicks'),
               ])
def displayPredictSeverity(dropdown1, dropdown2, dropdown3, dropdown4,
                           dropdown5, dropdown6, dropdown7, dropdown8,
                           dropdown9, dropdown10, dropdown11, dropdown12, btn_severity):

    """get data from the screen -tab: predict severity"""
    new_data = {'FIRST_CRASH_TYPE': dropdown1, 'CRASH_HOUR': dropdown2,
                'CRASH_DAY_OF_WEEK': dropdown3, 'CRASH_MONTH': dropdown4,
                'Contributory_Cause_New': dropdown5, 'Posted_Speed_New': dropdown6,
                'Traffic_Control_New': dropdown7, 'Weather_New': dropdown8,
                'Road_Surface_New': dropdown9, 'SEX2': dropdown10,
                'BAC2': dropdown11, 'AGE2': dropdown12}

    """convert to df"""
    df = pd.DataFrame(data=new_data, index=[0])

    if int(btn_severity >= 1):
        print("BOTON predict PRESSED!!!")
        print(df)
        model = Model.ModelAccidents(df)
        prediction = model.predict_newdata_catboost(X_new=df)
        # Convert prediction to text
        msg = str(prediction)
        print("message: " + msg)
        print(prediction)
        return [html.Div(children=[
            html.Div(children=[msg]),
            html.Br(),
            ])
        ]
    else:
        return html.Div([])


if __name__ == '__main__':
    app.run_server(debug=True)