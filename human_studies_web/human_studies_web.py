from flask import Flask
from flask import render_template, render_template_string, abort

app = Flask(__name__)
from pathlib import Path

import importlib

scenarios = [path.stem for path in (Path(__file__).parent / "scenarios").glob('*.py')]

@app.route('/study/<scenario>/<case>')
def show_user_profile(scenario: str, case: str) -> str:
    if scenario not in scenarios:
        print(f"no scenario {scenario}")
        abort(404)
    module = importlib.import_module(f"scenarios.{scenario}")
    cls = getattr(module, case, None)
    if cls is None:
        print(f"no case scenarios.{scenario}.{case}")
        abort(404)
    return render_template("index.html")
    return cls.render_test_for_user()
    # return render_template_string('hello.html', person=name)
    # show the user profile for that user
    # return f'User {case}'


# ans = [
#     "left",
#     "right,
#     "all good",
#     "all bad"
#     "unknown",
# ]

# class Scenario2(Scenario):
#     def go(self):
#         name = yield ask_name()
#         yield test()
#         yield finish()
