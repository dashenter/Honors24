from flask import Flask, render_template, request
#import do_calculations_video

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Process form data here
        # You might need to call functions from do_calculations_video.py
        pass

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)