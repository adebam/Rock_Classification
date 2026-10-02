FROM python:3.12-slim

WORKDIR /app
# copy fast_ap_api to the working dir app
COPY Streamlit/fast_api_app.py /app/ 
 # requirement to the working dir app
COPY Streamlit/requirements.txt /app/ 

COPY Streamlit/data_model.py /app/
COPY Streamlit/*.py /app/
COPY Streamlit/*.json /app/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 8501
CMD ["uvicorn", "fast_api_app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# instructions
# because you are starting a using yaml file and starting 2 apps use this

#step 1
#docker compose up --build


# push to docker hub, but Amazon ECR is better if u wanna deploy on ec2
# for this i am not going to deploy to aws because of the security issues. I might deploy it, but other people won't be able to deploy it