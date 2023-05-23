FROM python:3.10.6
 
WORKDIR /code
 
COPY ./requirements.txt /code/requirements.txt
 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
 
COPY . /code

RUN cd /code/GroundingDINO && pip install -e .

COPY .aws /root/.aws
 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]