from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

models = {
    'random_forest': pickle.load(open('./diabetes_model.pkl', 'rb')),
    'decision_tree': pickle.load(open('./tree_model.pkl', 'rb')),
    'knn': pickle.load(open('./knn_model.pkl', 'rb')),
    'svm': pickle.load(open('./svm_model.pkl', 'rb')),
}


@app.route('/')
def home():
    return render_template('/home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    int_features = [x for x in request.form.values() if x.isdigit()]
    model_name = [x for x in request.form.values() if not x.isdigit()][0]
    final_features = [np.array(int_features)]
    model = models[model_name]
    prediction = model.predict(final_features)

    output = prediction[0]

    if output == 0:
        return render_template('/home.html', prediction_text='Diabetes : No')
    else:
        return render_template('/home.html', prediction_text='Diabetes : Yes')


if __name__ == "__main__":
    app.run(debug=True)
