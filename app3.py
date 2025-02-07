from flask import Flask, render_template, request, jsonify
import math

# Necessary imports

# Initiate flask
app = Flask(__name__)







# Function to calculate PCSA
def calculate_pcsa(muscle_volume, pennation_angle, fascicle_length):
    """
    Calculate Physiological Cross-Sectional Area (PCSA)
    Formula: PCSA = (MV * cos(PA)) / FL
    """
    if fascicle_length == 0:  # Prevent division by zero
        return None
    pcsa = (muscle_volume * math.cos(math.radians(pennation_angle))) / fascicle_length
    return round(pcsa, 2)  # Rounded to 2 decimal places

# Function to calculate Muscle Volume
def calculate_muscle_volume(muscle_thickness, height, sex):
    """
    Calculate Muscle Volume (MW) for males and females based on the formula.
    """
    if sex.lower() == 'male':
        muscle_volume = 219.9 * muscle_thickness + 31.3 * ((height - 51.1) / 2.31) - 1758
    else:
        muscle_volume = 219.9 * muscle_thickness + 31.3 * ((height - 70.2) / 1.84) - 1758

    return round(muscle_volume, 2)


@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        muscle_thickness = float(data.get('muscle_thickness', 0))
        pennation_angle = float(data.get('pennation_angle', 0))
        fascicle_length = float(data.get('fascicle_length', 0))
        height = float(data.get('height', 0))
        sex = data.get('sex', 'male')  # Default to male if not provided

        muscle_volume = calculate_muscle_volume(muscle_thickness, height, sex)
        pcsa = calculate_pcsa(muscle_volume, pennation_angle, fascicle_length)

        return jsonify({'muscle_volume': muscle_volume, 'pcsa': pcsa})
    except Exception as e:
        return jsonify({'error': str(e)}), 400




@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
