from flask import Flask, render_template, request
from inference_sdk import InferenceHTTPClient
import base64

app = Flask(__name__)

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="bnxGkARcQkoHG64jxvtf"
)

# Treatment dictionary
treatment_plan = {
    "Gingivitis": "Maintain good oral hygiene, brush twice a day, use antiseptic mouthwash, and get professional dental cleanings regularly.",
    "Hipodonsia": "Treatment options include dental implants, bridges, or orthodontic solutions to close gaps or replace missing teeth.",
    "Kalkulus": "Dental scaling and root planing by a dentist are essential to remove tartar buildup.",
    "Kanker": "Oral cancer treatment may include surgery, radiation therapy, or chemotherapy depending on severity. Early diagnosis is crucial.",
    "Karies": "Remove decayed parts and restore with dental fillings, crowns, or root canal if severe. Prevent with fluoride and diet control.",
    "Perubahan-Warna": "Professional teeth cleaning, whitening treatments, or veneers can help. Avoid stain-causing foods and drinks.",
    "Sariawan": "Apply topical gels, rinse with salt water or antiseptic mouthwash. Avoid spicy foods and manage stress.",
    "Warna": "If discoloration is natural, no treatment needed. If pathological, consult for cleaning or aesthetic correction."
}

@app.route("/", methods=["GET", "POST"])
def index():
    message = []
    prediction = None
    image_data = None
    treatment = None
    result_img = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_data = base64.b64encode(file.read()).decode("utf-8")
            result = CLIENT.infer(image_data, model_id="mouthdity-classification/1")

            if result and "predicted_classes" in result and result["predicted_classes"]:
                prediction = result["predicted_classes"][0]
                treatment = treatment_plan.get(prediction, "Please consult a dental professional for further advice.")
                result_img = result.get("image", "")

    return render_template("index.html", result_img=result_img, prediction=prediction, image_data=image_data, treatment=treatment)



if __name__ == "__main__":
    app.run(debug=True)
