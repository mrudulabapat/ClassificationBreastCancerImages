from flask import Flask

UPLOAD_FOLDER = 'C://Users/Mrudula Bapat/PycharmProjects/cnnmodelv1/uploads'

app = Flask(__name__)
app.secret_key = "a7d4a959e9715edea8752781093b264a11c4c525832a59ae"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.jinja_env.filters['zip'] = zip

