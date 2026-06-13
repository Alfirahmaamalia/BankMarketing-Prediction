from flask import Flask, render_template, request, jsonify
import os
import traceback

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard_view.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_view():
    prediction = None
    prediction_label = None
    error_message = None
    
    if request.method == 'POST':
        try:
            # Get form data
            age = request.form.get('age', type=float)
            job = request.form.get('job')
            marital = request.form.get('marital')
            education = request.form.get('education')
            balance = request.form.get('balance', type=float)
            housing = request.form.get('housing')
            loan = request.form.get('loan')
            duration = request.form.get('duration', type=float)
            campaign = request.form.get('campaign', type=float)
            
            # Use default values for less important fields
            default_credit = 'no'
            contact = 'unknown'
            day = 15
            month = 'may'
            pdays = -1.0
            previous = 0.0
            poutcome = 'unknown'
            
            # Validate inputs
            input_values = [age, job, marital, education, balance, housing, loan, duration, campaign]
                            
            if any(v is None or str(v).strip() == "" for v in input_values):
                error_message = "Semua field harus diisi"
            else:
                # Import prediction function
                try:
                    from modelling import predict
                    
                    # Prepare input data
                    input_data = {
                        'age': age,
                        'job': job,
                        'marital': marital,
                        'education': education,
                        'default': default_credit,
                        'balance': balance,
                        'housing': housing,
                        'loan': loan,
                        'contact': contact,
                        'day': day,
                        'month': month,
                        'duration': duration,
                        'campaign': campaign,
                        'pdays': pdays,
                        'previous': previous,
                        'poutcome': poutcome
                    }
                    
                    # Make prediction
                    result = predict(input_data)
                    
                    if isinstance(result, dict) and "prediction" in result:
                        prediction = result["prediction"]
                        probability = result["probability"]
                        prediction_label = "Berlangganan (Subscribed)" if prediction == "yes" else "Tidak Berlangganan"
                    else:
                        prediction = result
                        probability = None
                        prediction_label = "Berlangganan (Subscribed)" if "yes" in str(result).lower() else "Tidak Berlangganan"
                        
                    # Hardcoded metrics for the Balanced Model
                    metrics = {
                        "accuracy": "0.898042684",
                        "precision": "0.551750380",
                        "f1_score": "0.611298482"
                    }
                        
                except ImportError:
                    error_message = "Model module tidak ditemukan. Pastikan modelling.py sudah ada."
                except Exception as e:
                    error_message = f"Error dalam prediksi: {str(e)}"
                    print(traceback.format_exc())
        
        except Exception as e:
            error_message = f"Error memproses form: {str(e)}"
            print(traceback.format_exc())
    
    return render_template('form_prediction.html', 
                         prediction=prediction, 
                         probability=probability if 'probability' in locals() else None,
                         prediction_label=prediction_label,
                         metrics=metrics if 'metrics' in locals() else None,
                         error_message=error_message)

@app.route('/health')
def health():
    """Health check endpoint untuk deployment"""
    return jsonify({"status": "healthy", "message": "App is running"}), 200

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
