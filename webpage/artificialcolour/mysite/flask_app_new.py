from uuid import uuid4
import random
import csv
from flask import Flask, render_template, session, request, redirect, url_for
from utils import generate_palette
from ColourMaths import HEX2LUV

session_lock_vars = [
    'lock1',
    'lock2',
    'lock3',
    'lock4',
    'lock5'
]
session_colours_vars = [
    'col1',
    'col2',
    'col3',
    'col4',
    'col5'
]


def rand_colour():
    return f'#{random.randint(0, 2**24):06X}'


app = Flask(__name__)
app.secret_key = uuid4().hex


@app.route('/', methods=['GET', 'POST'])
def colour_generator():
    hex_inputs = list()
    unlocked_cols = list()
    for sess_col in session_colours_vars:
        if sess_col not in session:
            session[sess_col] = rand_colour()
    if request.method == 'POST':
        for lock, col in zip(session_lock_vars, session_colours_vars):
            if request.form.get(lock) == 'on':
                session[lock] = True
                session[col] = request.form.get(col)
                hex_inputs += [request.form.get(col)]
            else:
                session[lock] = False
                unlocked_cols += [col]
                # session[col] = rand_colour()

        n = 5 - len(hex_inputs)
        gen_out = generate_palette(hex_inputs)[:n]
        for col, hex in zip(unlocked_cols, gen_out):
            session[col] = hex

    return render_template(
        'new_template.html',
        col1=session.get('col1'),
        col2=session.get('col2'),
        col3=session.get('col3'),
        col4=session.get('col4'),
        col5=session.get('col5')
    )

@app.route('/info', methods=['GET', 'POST'])
def load_info():
    if 'col1' not in session:
        return redirect(url_for('colour_generator'))
    else:
        csv_file = open('/home/artificialcolour/mysite/files/user_data.csv', mode='a')
        data = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data.writerow([
            session.get("col1"), str(list(HEX2LUV(session.get("col1")))),
            session.get("col2"), str(list(HEX2LUV(session.get("col2")))),
            session.get("col3"), str(list(HEX2LUV(session.get("col3")))),
            session.get("col4"), str(list(HEX2LUV(session.get("col4")))),
            session.get("col5"), str(list(HEX2LUV(session.get("col5"))))
            ])
        return f'hex {session.get("col1")} {session.get("col2")} {session.get("col3")} {session.get("col4")} {session.get("col5")} thank you <3'
