from fastapi import FastAPI, UploadFile, File
import uvicorn
import pandas as pd
import io

app = FastAPI()

@app.post("/api/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    # Reading the contents of the file into memory 
    content = await file.read()
    
    try:
        # Using pandas for reading CSV from RAM 
        # io.BytesIO let pandas read bytes like it's saved files 
        df = pd.read_csv(io.BytesIO(content))
        
        # Get basic data of the table 
        columns = df.columns.tolist()
        row_count = len(df)
        
        return {
            "filename": file.filename,
            "status": "success",
            "rows": row_count,
            "columns": columns,
            "message": "CSV successfully loaded!"
        }
    except Exception as e:
        return {
            "filename": file.filename,
            "status": "error",
            "message": f"Could not process CSV. Error: {str(e)}"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
