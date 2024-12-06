from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
from flask import Flask
from flask import request, render_template, render_template_string, abort


app = Flask(__name__)
from pathlib import Path

import importlib

scenarios = [
    path.stem for path in (Path(__file__).parent / "scenarios").glob("*.py")
]

@runtime_checkable
class TestCase(Protocol):
    def item_for_user(self, username: str, idx: int) -> Any:
        ...


case_instances: dict[tuple[str, str], TestCase] = {}


@app.route("/")
def list_scenarios() -> str:
    links = [
        (f"study/{scenario}", scenario.capitalize()) for scenario in scenarios
    ]
    return render_template("links_list.html", title="Scenarios", links=links)


@app.route("/study/<scenario>/")
def list_cases(scenario: str) -> str:
    if scenario not in scenarios:
        abort(404)
    module = importlib.import_module(f"scenarios.{scenario}")
    cases = [
        key
        for key, value in module.__dict__.items()
        if not key.startswith("__") and isinstance(value, type) and issubclass(value, TestCase)
    ]
    links = [(case, case.capitalize()) for case in cases]
    return render_template("links_list.html", title=f"Cases for {scenario.capitalize()}", links=links)


def get_case_instance(scenario_name: str, case_name: str) -> TestCase:
    key = scenario_name, case_name
    if key in case_instances:
        return case_instances[key]
    if scenario_name not in scenarios:
        print(f"no scenario {scenario_name}")
        abort(404)
    module = importlib.import_module(f"scenarios.{scenario_name}")
    case_cls = getattr(module, case_name, None)
    if case_cls is None or not issubclass(case_cls, TestCase):
        print(f"no case scenarios.{scenario_name}.{case_name}")
        abort(404)
    isinstance = case_instances[key] = case_cls()
    return isinstance


@app.route("/study/<scenario_name>/<case_name>")
def study_scenario_case(scenario_name: str, case_name: str) -> str:
    case = get_case_instance(scenario_name=scenario_name, case_name=case_name)
    return render_template("index.html", doc=case.__doc__)


@app.route("/study/<scenario_name>/<case_name>/<int:index>")
def study_scenario_case_index(scenario_name: str, case_name: str, index: int) -> str:
    username = request.args['username']
    case = get_case_instance(scenario_name=scenario_name, case_name=case_name)
    return case.item_for_user(username=username, idx=index)

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
