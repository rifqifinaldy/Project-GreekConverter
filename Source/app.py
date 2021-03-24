import os
from flask import Flask, render_template,redirect, flash, request
from werkzeug.utils import secure_filename
from Core2 import prosesAll, delete

Web_Folder = os.path.join('static', 'img')
Thresh_folder = os.path.join('static','Threshold')
Gray_folder = os.path.join('static', 'Grayscale')
Lpf_folder = os.path.join('static','lpf')
Char_folder = os.path.join('static','char_seg')
ext = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['Web_Folder'] = Web_Folder
app.config['Gray_folder'] = Gray_folder
app.config['Thresh_folder']= Thresh_folder
app.config['Lpf_folder']=Lpf_folder
app.config['Char_folder']= Char_folder
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 
app.config["CACHE_TYPE"] = "null"

def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ext

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/kamus")
def kamus():
    return render_template("dictionary.html")

@app.route("/dataset")
def dataset():
    return render_template("dataset.html")

@app.route("/upload")
def upload():
    clear = delete()
    return render_template("upload.html", clear = clear)


@app.route('/upload2', methods=['POST'])
def upload2():
    full_filename = os.path.join(app.config['Web_Folder'], 'Capture.PNG')

    if request.method == 'POST':
        if 'files[]' not in request.files:
                flash('No file part')
                return redirect(request.url)
        files = request.files.getlist('files[]')
        for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['Web_Folder'], 'Capture.PNG'))
                    
        flash('File(s) successfully uploaded')
    return render_template('upload2.html',user_image = full_filename)

@app.route("/Translate", methods=['GET', 'POST'])
def Translate():

    full_filename = os.path.join(app.config['Web_Folder'], 'Capture.PNG')
       
    translate = prosesAll(full_filename)
    return render_template('translate.html', translate=translate,user_image = full_filename)

@app.route("/Preprocess", methods=['GET','POST'])
def Preprocess():
    
    full_filename = os.path.join(app.config['Web_Folder'], 'Capture.PNG')
    full_filename1 = os.path.join(app.config['Gray_folder'], 'Grayscale.PNG')
    full_filename2 = os.path.join(app.config['Thresh_folder'], 'Threshold.PNG')
    full_filename3 = os.path.join(app.config['Lpf_folder'], 'LPF.PNG')
    translate = prosesAll(full_filename)
    
    
    return render_template('preprocess.html',translate=translate,
                           gray_image = full_filename1, 
                           thresh_image = full_filename2, 
                           Lpf_image = full_filename3, 
                           user_image = full_filename )

@app.route("/Normal", methods=['GET','POST'])
def Normal():
    
    norms = os.listdir('static/normal')
    norms = ['normal/' +file for file in norms]
    
    return render_template('normal.html',  norms = norms)


@app.route("/Segmentasi", methods=['GET','POST'])
def segmentasi():
    char_segmens = os.listdir('static/char_seg')
    char_segmens = ['char_seg/' + file for file in char_segmens]
    
    line_segmens = os.listdir('static/line_seg')
    line_segmens = ['line_seg/' + file for file in line_segmens]
    
    words_segmens = os.listdir('static/words_seg')
    words_segmens = ['words_seg/' + file for file in words_segmens]
    
    return render_template('segmen.html',char_segmens=char_segmens, 
                           words_segmens = words_segmens, 
                           line_segmens = line_segmens)
        
        
@app.route("/clear", methods =['GET','POST'])
def clear():
    clear = delete()
    return render_template('upload.html', clear = clear)
    
if __name__=="__main__":
    app.run(port=5000, debug=True)

        
    