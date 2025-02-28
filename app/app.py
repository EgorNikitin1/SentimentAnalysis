from fastapi import FastAPI

from ml.ml import polarity_scores_roberta

app = FastAPI()

# GET запрос с параметром пути
@app.get("/process/{input_string}")
async def process_input(input_string: str):
    result = polarity_scores_roberta(input_string)
    return result


# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)