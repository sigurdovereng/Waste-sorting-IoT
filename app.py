from flask import Flask, render_template
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_JSON = os.path.join(BASE_DIR, "dashboard_data.json")

def load_dashboard_data():
    if not os.path.exists(DASHBOARD_JSON):
        return {"latest": {"result": {}}, "history": []}

    with open(DASHBOARD_JSON, "r") as f:
        return json.load(f)

@app.route("/")
def index():
    data = load_dashboard_data()
    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000
    , debug=False)



    