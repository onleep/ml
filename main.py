from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
from io import BytesIO
import pandas as pd
import uvicorn
import pickle
import uvicorn

with open('model_preprocessors.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data['model']
    encoder = data['encoder']
    scaler = data['scaler']

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def convert_mileage(mileage, fuel_type):
    fuel_density = {
        'Diesel': 0.832,
        'Petrol': 0.74,
        'LPG': 0.51,
        'CNG': 0.2
    }
    if 'kmpl' in mileage: return float(mileage.replace(' kmpl', ''))
    elif 'km/kg' in mileage:
        km_per_kg = float(mileage.replace(' km/kg', ''))
        density = fuel_density.get(fuel_type)
        if density is None: return None
        return km_per_kg * density
    return None


def preprocessing(car_data) -> pd.DataFrame:
    car_data['name'] = car_data['name'].str.split().str[0]

    car_data['mileage'] = car_data.apply(
        lambda x: convert_mileage(x['mileage'], x['fuel'])
        if pd.notnull(x['mileage']) and not isinstance(x['mileage'], float) else x['mileage'], axis=1)

    car_data['engine'] = car_data.apply(
        lambda x: int(x['engine'].replace(' CC', '')) if pd.notnull(
            x['engine']) and not isinstance(x['engine'], int) else x['engine'],
        axis=1)

    car_data['max_power'] = car_data['max_power'].replace(' bhp', None)
    car_data['max_power'] = car_data.apply(
        lambda x: float(x['max_power'].replace(' bhp', '')) if pd.notnull(
            x['max_power']) and not isinstance(x['max_power'], float) else x['max_power'],
        axis=1)
    return car_data


@app.post('/predict_item')
def predict_item(item: Item) -> float:
    car_data = pd.DataFrame([item.model_dump()])
    car_data = preprocessing(car_data)

    object_features = ['seats'] + [col for col in car_data.columns 
                                   if car_data[col].dtype == 'object' and col != 'torque']
    car_data_encoded = encoder.transform(car_data[object_features])
    car_data_encoded = pd.DataFrame(car_data_encoded, columns=encoder.get_feature_names_out(object_features))

    number_features = [col for col in car_data.columns
                       if car_data[col].dtype in ['int64', 'float64'] and col != 'selling_price']
    car_data_scaled = scaler.transform(car_data[number_features])
    car_data_scaled = pd.DataFrame(
        car_data_scaled, columns=scaler.get_feature_names_out(number_features))
    
    car_data_2 = pd.concat([car_data_scaled, car_data_encoded], axis=1)
    predicted_price = model.predict(car_data_2)
    return abs(predicted_price[0])


@app.post('/predict_items')
async def predict_items(request: Request) -> StreamingResponse:
    content = await request.body()
    df = pd.read_csv(BytesIO(content))
    try:
        df_records = df.to_dict(orient='records')
        for i in range(len(df_records)):
            try: df_records[i]['predict_price'] = predict_item(Item(**df_records[i]))
            except: continue
        result_df = pd.DataFrame(df_records)
        csv_buffer = BytesIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return StreamingResponse(csv_buffer, media_type='text/csv', headers={'Content-Disposition': 'attachment; filename=result.csv'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Ошибка обработки данных: {str(e)}')

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
