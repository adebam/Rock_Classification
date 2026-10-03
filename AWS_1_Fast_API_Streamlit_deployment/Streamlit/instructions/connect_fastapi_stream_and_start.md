PORT summary

| Application | Port |
|---|---|
| JupyterLab (current jupyter) | `8888` |
| FastAPI | `8000` |
| Streamlit | `8501` |


Make sure that your enviroment and you are in the streamlit folder

**STEP 1**
In fast_api_app.py, set the bottom block to:
if __name__ == "__main__":
    uvicorn.run(
        app="fast_api_app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )

**STEP 2**
Expose FASTAPI Address in Helper Functions
**Base URL:**  
`http://127.0.0.1:8000`
*Example:*  
```python
api_url = "[http://127.0.0.1:8000/api/v1/rock_classification/single-image](http://127.0.0.1:8000/api/v1/rock_classification/single-image)"
```

**STEP 3**
Keep your FastAPI startup command as:

```bash
nohup python -m uvicorn fast_api_app:app \
    --host 127.0.0.1 \
    --port 8000 \
    > fastapi.log 2>&1 &
```
If FastAPI is already running, stop the existing process using your usual method before starting its replacement.


**STEP 4**
Keep your Streamlit startup command as:
```bash
nohup python -m streamlit run streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    > streamlit.log 2>&1 &
```

**STEP 5**
Verify FastAPI from your EC2 terminal
```bash
curl -fsS http://127.0.0.1:8000/
```
Your backend should return:
"Hello! I am up!!!"

**STEP 6**
Open Streamlit in your browser:
http://YOUR_EC2_PUBLIC_IP:8501
http://3.217.190.12:8501
