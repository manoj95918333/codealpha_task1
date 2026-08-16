from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("credit_scoring_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get customer details from the form
    age = int(request.form["Age"])
    annual_income = float(request.form["Annual_Income"])
    total_debt = float(request.form["Total_Debt"])
    credit_history_years = float(request.form["Credit_History_Years"])
    payment_history = float(request.form["Payment_History"])
    loan_count = int(request.form["Loan_Count"])
    credit_utilization = float(request.form["Credit_Utilization_Percent"])
    late_payments = int(request.form["Late_Payments"])
    employment_years = float(request.form["Employment_Years"])
    savings = float(request.form["Savings"])
    credit_cards = int(request.form["Number_of_Credit_Cards"])

    # Create customer DataFrame
    new_customer = pd.DataFrame([{
        "Age": age,
        "Annual_Income": annual_income,
        "Total_Debt": total_debt,
        "Credit_History_Years": credit_history_years,
        "Payment_History": payment_history,
        "Loan_Count": loan_count,
        "Credit_Utilization_Percent": credit_utilization,
        "Late_Payments": late_payments,
        "Employment_Years": employment_years,
        "Savings": savings,
        "Number_of_Credit_Cards": credit_cards
    }])

    # Make prediction
    prediction = model.predict(new_customer)

    # Get prediction probability
    probability = model.predict_proba(new_customer)

    # Display result
    if prediction[0] == 1:
        result = "Creditworthy"
        confidence = round(probability[0][1] * 100, 2)
    else:
        result = "Not Creditworthy"
        confidence = round(probability[0][0] * 100, 2)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)