from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import traceback

# تحميل النماذج والملفات
binary_model = joblib.load("binary_allergen_model.pkl")
multi_label_model = joblib.load("multilabel_allergen_model.pkl")
encoder = joblib.load("preprocessor.pkl")
allergen_list = joblib.load("allergen_list.pkl")

# تعريف الأعمدة
categorical_cols = ['Food Product', 'Main Ingredient', 'Sweetener', 'Fat/Oil', 'Seasoning']
numerical_cols = ['Is_Allergen']  # في حالة الاستخدام لاحقًا

# إنشاء تطبيق FastAPI
app = FastAPI(title="Allergen Detection API")

# تعريف نموذج البيانات
class FoodInput(BaseModel):
    Food_Product: str
    Main_Ingredient: str
    Sweetener: str
    Fat_Oil: str
    Seasoning: str

# الدالة المسؤولة عن المعالجة والتنبؤ
def process_and_predict(input_data: FoodInput):
    try:
        # تحويل البيانات إلى DataFrame
        input_dict = {
            'Food Product': input_data.Food_Product,
            'Main Ingredient': input_data.Main_Ingredient,
            'Sweetener': input_data.Sweetener,
            'Fat/Oil': input_data.Fat_Oil,
            'Seasoning': input_data.Seasoning
        }
        df = pd.DataFrame([input_dict])

        # تطبيق OneHotEncoder
        cat_encoded = encoder.transform(df[categorical_cols]).toarray()

        # التنبؤ بوجود حساسية
        has_allergen = binary_model.predict(cat_encoded)[0]
        if has_allergen == 0:
            return {
                "contains_allergens": False,
                "detected_allergens": [],
                "message": "No allergens detected."
            }

        # التنبؤ بأنواع مسببات الحساسية
        multi_preds = multi_label_model.predict(cat_encoded)[0]
        detected = [a for a, present in zip(allergen_list, multi_preds) if present]

        return {
            "contains_allergens": True,
            "detected_allergens": detected,
            "message": f"Allergens detected: {', '.join(detected)}"
        }

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# نقطة النهاية للتنبؤ
@app.post("/predict")
def predict_allergens(input_data: FoodInput):
    result = process_and_predict(input_data)
    if "error" in result:
        return JSONResponse(status_code=500, content=result)
    return result
