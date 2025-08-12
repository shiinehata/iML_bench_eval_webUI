# app.py
import os
import importlib
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from pathlib import Path
import secrets

# App Configuration
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOADS_DIR'] = 'uploads'
app.config['COMPETITIONS_DIR'] = 'competitions'
app.config['EVALUATORS_DIR'] = 'evaluators'
os.makedirs(app.config['UPLOADS_DIR'], exist_ok=True)

# Custom sort order for competitions
CUSTOM_ORDER = [
    'predict_effective_arguments',
    'toxic_comment_classification',
    'query_domain_classification',
    'predict_the_llms',
    'pet_finder',
    'multi_label_classification',
    'plant_traits_2024',
    'dog_breed_classification',
    'paddy_disease_classification',
    'steel_plate_defect_prediction'
]

def get_competitions():
    """Scans the competitions directory and sorts them according to the custom order."""
    try:
        available_competitions = [d for d in os.listdir(app.config['COMPETITIONS_DIR']) if os.path.isdir(os.path.join(app.config['COMPETITIONS_DIR'], d))]
        
        # Sort available competitions based on CUSTOM_ORDER
        # Competitions not in CUSTOM_ORDER will be placed at the end
        sorted_competitions = sorted(
            available_competitions, 
            key=lambda x: CUSTOM_ORDER.index(x) if x in CUSTOM_ORDER else float('inf')
        )
        return sorted_competitions
    except FileNotFoundError:
        return []

@app.route('/')
def index():
    """Displays the main page with a list of competitions."""
    competitions_list = get_competitions()
    results = session.get('results', {})
    return render_template('index.html', competitions=competitions_list, results=results)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Handles the evaluation for a specific competition."""
    if 'results' not in session:
        session['results'] = {}

    competitions_list = get_competitions()
    
    try:
        competition_name = request.form.get('competition_name')
        if not competition_name:
            raise ValueError("No competition name provided.")
            
        files = request.files.getlist('files[]')
        if not files or files[0].filename == '':
            raise ValueError("Please select one or more files to upload.")
        
        if competition_name not in session['results']:
            session['results'][competition_name] = []

        for file in files:
            filename = secure_filename(file.filename)
            upload_path = Path(app.config['UPLOADS_DIR']) / f"{competition_name}_{filename}"
            
            try:
                file.save(upload_path)
                ground_truth_path = Path(app.config['COMPETITIONS_DIR']) / competition_name / 'test_ground_truth.csv'
                
                evaluator_module = importlib.import_module(f"{app.config['EVALUATORS_DIR']}.{competition_name}")
                
                df_gt, df_pred = evaluator_module.validate_and_read_inputs(ground_truth_path, upload_path)
                scores = evaluator_module.evaluate_predictions(df_gt, df_pred)
                
                session['results'][competition_name].append({"scores": scores, "filename": filename})

            except Exception as e:
                session['results'][competition_name].append({"error": f"An error occurred while processing the file: {str(e)}", "filename": filename})
            
            finally:
                if os.path.exists(upload_path):
                    os.remove(upload_path)

    except Exception as e:
        # This will catch errors like no competition name or no files selected
        # We can't associate it with a specific competition, so maybe add a general error display area
        pass

    session.modified = True
    return redirect(url_for('index'))


@app.route('/clear_results')
def clear_results():
    session.pop('results', None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
