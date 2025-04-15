import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings

# تجاهل التحذيرات
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# إعداد Flask
app = Flask(__name__)
CORS(app)

# تحديد المسار الأساسي للملفات
base_path = os.path.dirname(__file__)

# تحميل البيانات
data = pd.read_csv(os.path.join(base_path, "final_dataset.csv"))
data = data.apply(pd.to_numeric, errors='coerce')

# تحويل Gender إلى 0 و1
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data.dropna(subset=['Age', 'Height', 'Weight', 'Gender', 'BMI', 'Exercise Recommendation Plan'], inplace=True)

# تحديد فئات BMI
bins = [-np.inf, 18.5, 24.9, 29.9, np.inf]
labels = ['Underweight', 'Normal weight', 'Overweight', 'Obese']
data['BMIcase'] = pd.cut(data['BMI'], bins=bins, labels=labels)

# تجهيز الـ Scaler للـ features
X = data[['Age', 'Height', 'Weight', 'Gender']].values
scaler_X = StandardScaler()
scaler_X.fit(X)

# تجهيز الـ Scaler للـ BMI (مش هنحتاجه دلوقتي)
y_bmi = data['BMI'].values
scaler_bmi = StandardScaler()
scaler_bmi.fit(y_bmi.reshape(-1, 1))

# تجهيز y_exercise_plan
y_exercise_plan = data['Exercise Recommendation Plan'].values
y_exercise_plan_adjusted = y_exercise_plan - 4
label_encoder_exercise_plan = OneHotEncoder(sparse_output=False)
label_encoder_exercise_plan.fit(y_exercise_plan_adjusted.reshape(-1, 1))

# تجهيز BMIcase
y_bmicas = data['BMIcase'].values
label_encoder_bmicas = OneHotEncoder(sparse_output=False)
label_encoder_bmicas.fit(y_bmicas.reshape(-1, 1))

# تحميل الموديلات
model_bmi = tf.keras.models.load_model(os.path.join(base_path, "model_bmi.keras"))
model_exercise_plan = tf.keras.models.load_model(os.path.join(base_path, "model_exercise_plan.keras"))
model_bmicas = tf.keras.models.load_model(os.path.join(base_path, "model_bmicas.keras"))
print("✅ تم تحميل النماذج بنجاح!")

# دالة التوصيات
def get_nutrition_and_exercise_recommendations(bmi_case):
    recommendations = {'nutrition': '', 'exercise': ''}
    if bmi_case == 'Underweight':
        recommendations['nutrition'] = 'Increase caloric intake with nutritious foods.'
        recommendations['exercise'] = 'Focus on strength training to build muscle mass.'
    elif bmi_case == 'Normal weight':
        recommendations['nutrition'] = 'Maintain a balanced diet with adequate nutrients.'
        recommendations['exercise'] = 'Continue regular exercise, mixing cardio and strength training.'
    elif bmi_case == 'Overweight':
        recommendations['nutrition'] = 'Reduce caloric intake and focus on healthy foods.'
        recommendations['exercise'] = 'Incorporate regular cardio exercises and strength training.'
    elif bmi_case == 'Obese':
        recommendations['nutrition'] = 'Consult a nutritionist for a tailored meal plan.'
        recommendations['exercise'] = 'Start with low-impact exercises and gradually increase intensity.'
    return recommendations

# Endpoint للتنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        age = float(data["age"])
        height = float(data["height"])
        weight = float(data["weight"])
        gender = 1 if data["gender"].lower() == "male" else 0

        calculated_bmi = weight / (height ** 2)

        input_data = np.array([[age, height, weight, gender]])
        input_data_scaled = scaler_X.transform(input_data)
        input_data_rnn = np.expand_dims(input_data_scaled, axis=1)

        # التنبؤ بـ BMIcase
        try:
            bmicas_prediction = model_bmicas.predict(input_data_rnn, verbose=0)
            predicted_bmicas = label_encoder_bmicas.inverse_transform(bmicas_prediction)[0][0]
        except Exception as e:
            return jsonify({"error": f"Failed to predict BMIcase: {str(e)}"}), 500

        # التنبؤ بخطة التمارين
        try:
            exercise_plan_prediction = model_exercise_plan.predict(input_data_rnn, verbose=0)
            predicted_exercise_plan_adjusted = label_encoder_exercise_plan.inverse_transform(exercise_plan_prediction)
            predicted_exercise_plan = int(predicted_exercise_plan_adjusted[0][0]) + 4
            predicted_exercise_plan = max(4, min(predicted_exercise_plan, 7))
        except Exception as e:
            return jsonify({"error": f"Failed to predict Exercise Plan: {str(e)}"}), 500

        recommendations = get_nutrition_and_exercise_recommendations(predicted_bmicas)

        response = {
            "Calculated BMI (based on height and weight)": round(float(calculated_bmi), 2),
            "Predicted BMIcase": predicted_bmicas,
            "Exercise Plan": predicted_exercise_plan,
            "Nutrition Recommendation": recommendations['nutrition'],
            "Exercise Recommendation": recommendations['exercise']
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Vercel handler
def handler(environ, start_response):
    return app(environ, start_response)

# تشغيل محلي
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
