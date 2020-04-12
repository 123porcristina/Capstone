# pip install dash-dangerously-set-inner-html

import datetime
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_dangerously_set_inner_html

import dash_table as dt
import pandas as pd
from pathlib import Path
import pandas as pd
from Code import Preprocessing as preprocessing
from Code import Models as Model

import numpy as np
import pandas as pd
from skimage import io, data, transform
from time import sleep

################################################################################
###
### APP LAYOUT
###
################################################################################
app = dash.Dash(__name__)
app.css.append_css({'external_url': 'static/custom.css'})
app.config['suppress_callback_exceptions'] = True

def about():
    return html.Div(className='instructions-sidebar', children=[
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


app.layout = app.layout = html.Div(
    children=[
        html.Div(
            [
                html.Img(
                    src=app.get_asset_url("dash-logo.png"), className="plotly-logo"
                ),
                html.H1(children="Severity of Traffic Accidents in the City of Chicago"),
                about(),
                html.Div(
                    [
                        html.Button(
                            "BUTTON1",
                            # "LEARN MORE",
                            className="button_instruction",
                            id="learn-more-button",
                        ),
                        html.Button(
                            "BUTTON2", className="demo_button", id="demo"
                            # "UPLOAD DEMO DATA", className="demo_button", id="demo"
                        ),
                    ],
                    className="mobile_buttons",
                ),
                # html.Div(
                #     # Empty child function for the callback
                #     html.Div(id="demo-explanation", children=[])
                # ),
                # html.Div(
                #     [
                #         html.Div(
                #             [
                #                 html.Label("Number of rows"),
                #                 dcc.Input(
                #                     id="nrows-stitch",
                #                     type="number",
                #                     value=1,
                #                     name="number of rows",
                #                     min=1,
                #                     step=1,
                #                 ),
                #             ]
                #         ),
                #         html.Div(
                #             [
                #                 html.Label("Number of columns"),
                #                 dcc.Input(
                #                     id="ncolumns-stitch",
                #                     type="number",
                #                     value=1,
                #                     name="number of columns",
                #                     min=1,
                #                     step=1,
                #                 ),
                #             ]
                #         ),
                #     ],
                #     className="mobile_forms",
                # ),
                # html.Div(
                #     [
                #         html.Label("Downsample factor"),
                #         dcc.RadioItems(
                #             id="downsample",
                #             options=[
                #                 {"label": "1", "value": "1"},
                #                 {"label": "2", "value": "2"},
                #                 {"label": "4", "value": "4"},
                #                 {"label": "8", "value": "8"},
                #             ],
                #             value="2",
                #             labelStyle={"display": "inline-block"},
                #             style={"margin-top": "-15px"},
                #         ),
                #         html.Label("Fraction of overlap (in [0-1] range)"),
                #         dcc.Input(
                #             id="overlap-stitch", type="number", value=0.15, min=0, max=1
                #         ),
                #         html.Br(),
                #         dcc.Checklist(
                #             id="do-blending-stitch",
                #             options=[{"label": "Blending images", "value": 1}],
                #             # value=[1],
                #         ),
                #     ],
                #     className="radio_items",
                # ),
                # html.Label("Measured shifts between images"),
                # html.Div(
                #     [
                #         # dash_table.DataTable(
                #         #     id="table-stitch",
                #         #     columns=columns,
                #         #     editable=True,
                #         #     style_table={
                #         #         "width": "81%",
                #         #         "margin-left": "4.5%",
                #         #         "border-radius": "20px",
                #         #     },
                #         #     style_cell={
                #         #         "text-align": "center",
                #         #         "font-family": "Geneva",
                #         #         "backgroundColor": "#01183A",
                #         #         "color": "#8898B2",
                #         #         "border": "1px solid #8898B2",
                #         #     },
                #         # )
                #     ],
                #     className="shift_table",
                # ),
                html.Br(),
                html.Button(
                    "BUTTON3", id="button-stitch", className="button_submit"
                    # "Run stitching", id="button-stitch", className="button_submit"
                ),
                html.Br(),
            ],
            className="four columns instruction",
        ),
        html.Div(
            [
                dcc.Tabs(
                    id="stitching-tabs",
                    value="canvas-tab",
                    children=[
                        dcc.Tab(label="PREDICT SEVERITY", value="canvas-tab"),
                        dcc.Tab(label="PREDICT RISK", value="result-tab"),
                        dcc.Tab(label="EXPLORATORY DATA ANALYSIS", value="help-tab"),
                    ],
                    className="tabs",
                ),
                html.Div(
                    id="tabs-content-example",
                    className="canvas",
                    style={"text-align": "left", "margin": "auto"},
                ),
                html.Div(className="upload_zone", id="upload-stitch", children=[]),
                html.Div(id="sh_x", hidden=True),
                html.Div(id="stitched-res", hidden=True),
                dcc.Store(id="memory-stitch"),
            ],
            className="eight columns result",
        ),
    ],
    className="row twelve columns",
)

"""CALLBACKS"""
@app.callback(
    Output("tabs-content-example", "children"), [Input("stitching-tabs", "value")]
)
def fill_tab(tab):
    if tab == "canvas-tab":
        return [
            # html.Div(
            #     children=[
            #         html.Div(
            #             html.P(children=["PRIMER TAB"]),
            #             # image_upload_zone("upload-stitch", multiple=True, width="100px")
            #         ),
            #     ],
            #     className="upload_zone",
            #     id="upload",
            # ),

            html.Iframe(
                id="bla",
                src=app.get_asset_url('EDA.html'),
                height="100%",
                width='100%',
                style={'border-style': 'none'}
            ),
        ]
    elif tab == "result-tab":
        return [
            dcc.Loading(
                id="loading-1",
                children=[
                    html.Img(
                        id="stitching-result",
                        # src=array_to_data_url(
                        #     np.zeros((height, width), dtype=np.uint8)
                        # ),
                        # width=canvas_width,
                    ),
                    html.P(children=["RESULT TAB"])
                ],
                # type="circle",
            ),
            html.Div(
                [
                    html.Label("Contrast"),
                    dcc.Slider(
                        id="contrast-stitch", min=0, max=1, step=0.02, value=0.5
                    ),
                    html.P(children=["DIV CONTRAST"])
                ],
                className="result_slider",
            ),
            html.Div(
                [
                    html.Label("Brightness"),
                    dcc.Slider(
                        id="brightness-stitch", min=0, max=1, step=0.02, value=0.5
                    ),
                    html.P(children=["BRIGHTNESS"])
                ],
                className="result_slider",
            ),
        ]
    return [

        ###

        # html.Div(children=[
        #     html.Label("First Crash Type"),
        #     dcc.Dropdown(
        #         id="nrows-stitch",
        #         className="three columns dropdown-box-first",
        #         options=[
        #             {'label': 'ANGLE', 'value': 'ANGLE'},
        #             {'label': 'FIXED OBJECT', 'value': 'FIXED OBJECT'},
        #             {'label': 'HEAD ON', 'value': 'HEAD ON'},
        #             {'label': 'OTHER NONCOLLISION', 'value': 'OTHER NONCOLLISION'},
        #             {'label': 'OTHER OBJECT', 'value': 'OTHER OBJECT'},
        #             {'label': 'OVERTURNED', 'value': 'OVERTURNED'},
        #             {'label': 'TURNING', 'value': 'TURNING'},
        #         ],
        #         value='ANGLE'
        #     ),
        # ]),
        #
        # html.Br(),
        #
        # html.Div(children=[
        #     html.Label("Number units"),
        #     dcc.Input(
        #         id="nrows-stitch",
        #         type="number",
        #         value=1,
        #         name="number of rows",
        #         min=1,
        #         step=1,
        #     ),
        # ]),
        #
        # html.Br(),
        #
        # html.Div(children=[
        #     html.Label("Crash Hour"),
        #     dcc.Dropdown(
        #         id="nrows-stitch",
        #         className="three columns dropdown-box-second",
        #         options=[
        #             {'label': '0', 'value': '0'},
        #             {'label': '1', 'value': '1'},
        #             {'label': '2', 'value': '2'},
        #             {'label': '3', 'value': '3'},
        #             {'label': '4', 'value': '4'},
        #             {'label': '5', 'value': '5'},
        #             {'label': '6', 'value': '6'},
        #             {'label': '7', 'value': '7'},
        #             {'label': '8', 'value': '8'},
        #             {'label': '9', 'value': '9'},
        #             {'label': '10', 'value': '10'},
        #             {'label': '11', 'value': '11'},
        #             {'label': '12', 'value': '12'},
        #             {'label': '13', 'value': '13'},
        #             {'label': '14', 'value': '14'},
        #             {'label': '15', 'value': '15'},
        #             {'label': '16', 'value': '16'},
        #             {'label': '17', 'value': '17'},
        #             {'label': '18', 'value': '18'},
        #             {'label': '19', 'value': '19'},
        #             {'label': '20', 'value': '20'},
        #             {'label': '21', 'value': '21'},
        #             {'label': '22', 'value': '22'},
        #             {'label': '23', 'value': '23'},
        #         ],
        #         value='15'
        #     ),
        # ]),
        #
        # html.Br(),
        #
        # html.Div(children=[
        #     html.Label("Crash Day of Week"),
        #     dcc.Dropdown(
        #         id="nrows-stitch",
        #         className="three columns dropdown-box-third",
        #         options=[
        #             {'label': 'Sunday', 'value': '1'},
        #             {'label': 'Monday', 'value': '2'},
        #             {'label': 'Tuesday', 'value': '3'},
        #             {'label': 'Wednesday', 'value': '4'},
        #             {'label': 'Thursday', 'value': '5'},
        #             {'label': 'Friday', 'value': '6'},
        #             {'label': 'Saturday', 'value': '7'},
        #         ],
        #         value='1'
        #     ),
        # ]),
        #
        # html.Br(),
        #
        # html.Div(children=[
        #     html.Label("Crash Month"),
        #     dcc.Dropdown(
        #         id="nrows-stitch",
        #         className="three columns dropdown-box-first",
        #         options=[
        #             {'label': 'January', 'value': '1'},
        #             {'label': 'February', 'value': '2'},
        #             {'label': 'March', 'value': '3'},
        #             {'label': 'April', 'value': '4'},
        #             {'label': 'May', 'value': '5'},
        #             {'label': 'June', 'value': '6'},
        #             {'label': 'July', 'value': '7'},
        #             {'label': 'August', 'value': '8'},
        #             {'label': 'September', 'value': '9'},
        #             {'label': 'October', 'value': '10'},
        #             {'label': 'November', 'value': '11'},
        #             {'label': 'December', 'value': '12'},
        #         ],
        #         value='1'
        #     ),
        # ]),
        #
        # html.Br(),
        #
        # html.Div(children=[
        #     html.Label("Contributory Cause"),
        #     dcc.Dropdown(
        #         id="nrows-stitch",
        #         className="three columns dropdown-box-first",
        #         options=[
        #             {'label': 'ANIMAL', 'value': 'ANIMAL'},
        #             {'label': 'DISREGARDING OTHER TRAFFIC SIGNS', 'value': 'DISREGARDING OTHER TRAFFIC SIGNS'},
        #             {'label': 'DISTRACTION', 'value': 'DISTRACTION'},
        #             {'label': 'DRIVING ON WRONG SIDE/WRONG WAY', 'value': 'DRIVING ON WRONG SIDE/WRONG WAY'},
        #             {'label': 'DRIVING SKILLS/KNOWLEDGE/EXPERIENCE', 'value': 'DRIVING SKILLS/KNOWLEDGE/EXPERIENCE'},
        #             {'label': 'IMPROPER BACKING', 'value': 'IMPROPER BACKING'},
        #             {'label': 'IMPROPER LANE USAGE', 'value': 'IMPROPER LANE USAGE'},
        #             {'label': 'OTHER','value': 'OTHER'},
        #             {'label': 'WEATHER', 'value': 'WEATHER'},
        #             {'label': 'VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)', 'value': 'VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)'},
        #             {'label': 'UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)',
        #              'value': 'UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)'},
        #             {'label': 'UNABLE TO DETERMINE', 'value': 'UNABLE TO DETERMINE'},
        #         ],
        #         value='WEATHER'
        #     ),
        # ]),
        #
        # html.Br(),
        #
        # html.Div(children=[
        #     html.Label("Posted Speed"),
        #     dcc.Input(
        #         id="nrows-stitch",
        #         type="number",
        #         value=1,
        #         name="number of rows",
        #         min=1,
        #         step=1,
        #     ),
        # ]),
        #
        # html.Br(),
        #
        # html.Div(children=[
        #     html.Label("Traffic Control"),
        #     dcc.Dropdown(
        #         id="nrows-stitch",
        #         className="three columns dropdown-box-first",
        #         options=[
        #             {'label': 'None', 'value': 'None'},
        #             {'label': 'Traffic Signal', 'value': 'TrafficSignal'},
        #             {'label': 'Other Control', 'value': 'OtherControl'},
        #         ],
        #         value='TrafficSignal'
        #     ),
        # ]),
        #
        # html.Br(),
        #
        # html.Div(children=[
        #     html.Label("Weather"),
        #     dcc.Dropdown(
        #         id="nrows-stitch",
        #         className="three columns dropdown-box-first",
        #         options=[
        #             {'label': 'Clear', 'value': 'Clear'},
        #             {'label': 'Snow', 'value': 'Snow'},
        #             {'label': 'Cloudy', 'value': 'Cloudy'},
        #             {'label': 'Rain', 'value': 'Rain'},
        #             {'label': 'Other', 'value': 'OtherForecast'},
        #         ],
        #         value='Clear'
        #     ),
        # ]),
        #
        # html.Br(),
        #
        # html.Div(children=[
        #     html.Label("Road Surface"),
        #     dcc.Dropdown(
        #         id="nrows-stitch",
        #         className="three columns dropdown-box-first",
        #         options=[
        #             {'label': 'Dry', 'value': 'Dry'},
        #             {'label': 'Snow', 'value': 'Snow'},
        #             {'label': 'Wet', 'value': 'Wet'},
        #             {'label': 'Other', 'value': 'OtherCondition'},
        #         ],
        #         value='Clear'
        #     ),
        # ]),
        #
        # html.Br(),
        # html.Div(children=[
        #     html.Label("Sex"),
        #     dcc.Dropdown(
        #         id="nrows-stitch",
        #         className="three columns dropdown-box-first",
        #         options=[
        #             {'label': 'Male-Female', 'value': 'MaleFemale'},
        #             {'label': 'Both Male', 'value': 'BothMale'},
        #             {'label': 'Both Other', 'value': 'BothOther'},
        #             {'label': 'Female Other', 'value': 'Female Other'},
        #             {'label': 'Male Other', 'value': 'MaleOther'},
        #         ],
        #         value='BothMale'
        #     ),
        # ]),
        # html.Br(),
        # html.Div(children=[
        #     html.Label("BAC"),
        #     dcc.Input(
        #         id="nrows-stitch",
        #         type="number",
        #         value=1,
        #         name="number of rows",
        #         min=1,
        #         step=1,
        #     ),
        # ]),
        # html.Br(),
        # html.Div(children=[
        #     html.Label("AGE"),
        #     dcc.Input(
        #         id="nrows-stitch",
        #         type="number",
        #         value=1,
        #         name="number of rows",
        #         min=1,
        #         step=1,
        #     ),
        # ]),
        # *****************
        # html.Div(className='control-tab', children=[
        #     html.Div(className='app-controls-block', children=[
        html.Div(className="three columns padding-top-bot", children=[
            html.Label("First Crash Type *"),
            dcc.Dropdown(
                id="nrows-Crash-Type",
                className='fasta-entry-dropdown',
                # className="three columns dropdown-box-first",
                options=[
                    {'label': 'ANGLE', 'value': 'ANGLE'},
                    {'label': 'FIXED OBJECT', 'value': 'FIXED OBJECT'},
                    {'label': 'HEAD ON', 'value': 'HEAD ON'},
                    {'label': 'OTHER NONCOLLISION', 'value': 'OTHER NONCOLLISION'},
                    {'label': 'OTHER OBJECT', 'value': 'OTHER OBJECT'},
                    {'label': 'OVERTURNED', 'value': 'OVERTURNED'},
                    {'label': 'TURNING', 'value': 'TURNING'},
                ],
                # style={'height': '30px', 'width': '10px'},
                placeholder='Select...',
                # value='ANGLE'
            ),

            html.Label("Number units"),
            dcc.Input(
                id="nrows-Number-units",
                type="number",
                value=1,
                name="number of rows",
                min=0,
                step=1,
                placeholder='Enter a value...',
            ),

            html.Label("Crash Hour"),
            dcc.Dropdown(
                id="nrows-Crash-Hour",
                # className="three columns dropdown-box-second",
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
                # value='15'
                placeholder='Select...',
            ),

            html.Label("Crash Day of Week"),
            dcc.Dropdown(
                id="nrows-day-week",
                # className="three columns dropdown-box-third",
                options=[
                    {'label': 'Sunday', 'value': '1'},
                    {'label': 'Monday', 'value': '2'},
                    {'label': 'Tuesday', 'value': '3'},
                    {'label': 'Wednesday', 'value': '4'},
                    {'label': 'Thursday', 'value': '5'},
                    {'label': 'Friday', 'value': '6'},
                    {'label': 'Saturday', 'value': '7'},
                ],
                # value='1'
                placeholder='Select...',
            ),

            html.Label("Crash Month"),
            dcc.Dropdown(
                id="nrows-month",
                # className="three columns dropdown-box-first",
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
                # value='1'
                placeholder='Select...',
            ),

            html.Label("Contributory Cause"),
            dcc.Dropdown(
                id="nrows-contributory",
                # className="three columns dropdown-box-first",
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
                # value='WEATHER'
                placeholder='Select...',
            ),

            html.Label("Posted Speed"),
            dcc.Input(
                id="nrows-speed",
                type="number",
                value=1,
                name="number of rows",
                min=30,
                step=1,
            ),

            html.Label("Traffic Control"),
            dcc.Dropdown(
                id="nrows-traffic",
                # className="three columns dropdown-box-first",
                options=[
                    {'label': 'None', 'value': 'None'},
                    {'label': 'Traffic Signal', 'value': 'TrafficSignal'},
                    {'label': 'Other Control', 'value': 'OtherControl'},
                ],
                # value='TrafficSignal'
                placeholder='Select...',
            ),

            html.Label("Weather"),
            dcc.Dropdown(
                id="nrows-weather",
                # className="three columns dropdown-box-first",
                options=[
                    {'label': 'Clear', 'value': 'Clear'},
                    {'label': 'Snow', 'value': 'Snow'},
                    {'label': 'Cloudy', 'value': 'Cloudy'},
                    {'label': 'Rain', 'value': 'Rain'},
                    {'label': 'Other', 'value': 'OtherForecast'},
                ],
                # value='Clear'
                placeholder='Select...',
            ),

            html.Label("Road Surface"),
            dcc.Dropdown(
                id="nrows-road",
                # className="three columns dropdown-box-first",
                options=[
                    {'label': 'Dry', 'value': 'Dry'},
                    {'label': 'Snow', 'value': 'Snow'},
                    {'label': 'Wet', 'value': 'Wet'},
                    {'label': 'Other', 'value': 'OtherCondition'},
                ],
                # value='Clear'
                placeholder='Select...',
            ),

            html.Label("Sex"),
            dcc.Dropdown(
                id="nrows-sex",
                # className="three columns dropdown-box-first",
                options=[
                    {'label': 'Male-Female', 'value': 'MaleFemale'},
                    {'label': 'Both Male', 'value': 'BothMale'},
                    {'label': 'Both Other', 'value': 'BothOther'},
                    {'label': 'Female Other', 'value': 'Female Other'},
                    {'label': 'Male Other', 'value': 'MaleOther'},
                ],
                # value='BothMale'
                placeholder='Select...',
            ),

            html.Label("BAC"),
            dcc.Input(
                id="nrows-bac",
                type="number",
                value=1,
                name="number of rows",
                min=0,
                step=1,
            ),

            html.Label("AGE"),
            dcc.Input(
                id="nrows-age",
                type="number",
                value=1,
                name="number of rows",
                min=16,
                step=1,
            ),

        ]),
        # ]),
        # html.Hr(),
        # ])

        ###
    ]




if __name__ == "__main__":
    app.run_server(debug=True)
