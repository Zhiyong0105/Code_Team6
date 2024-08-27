from fastapi import FastAPI, File, UploadFile
from deepface import DeepFace
from PIL import Image
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):

    image = Image.open(io.BytesIO(await file.read()))
    img_array = np.array(image)

    analysis = DeepFace.analyze(img_array, actions=['age', 'gender'])

    people_count = len(analysis) if isinstance(analysis, list) else 1

    result = {
        "total_people": people_count,
        "details": []
    }

    for i, person in enumerate(analysis):
        gender_data = person['gender']
        woman_prob = gender_data['Woman']
        man_prob = gender_data['Man']

        gender = 'Woman' if woman_prob > man_prob else 'Man'

        result["details"].append({
            "person": i + 1,
            "gender": gender,
            "probability_woman": woman_prob,
            "probability_man": man_prob
        })

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
