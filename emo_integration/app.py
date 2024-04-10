from flask import Flask, render_template, request, jsonify
import os
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('main.html')
@app.route('/join', methods=['GET','POST'])
def my_form_post():
    target = os.path.join("output")
    if not os.path.isdir(target):
    	os.mkdir(target)
    imagefile = request.files.get("img")
    destination = "/".join([target, "temp.mp3"])
    imagefile.save(destination);
    return jsonify(result=target)
if __name__ == '__main__':
    app.run(debug=True)