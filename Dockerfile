# ใช้ฐานข้อมูลของ Python ที่มี Django ไว้แล้วเป็นภาพรวม (base image)
FROM python:3.9

# ตั้งค่าตัวแปรสภาพแวดล้อม
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# ติดตั้งคำสั่งรัน Django และคำสั่งสำหรับการทำงานใน Docker
RUN pip install --upgrade pip
COPY requirements.txt /code/
RUN pip install -r /code/requirements.txt
COPY . /code/
WORKDIR /code/

# เปิดพอร์ต 8000 เพื่อให้ Django เข้าถึง
EXPOSE 8000

# คำสั่งเริ่มต้นการรัน Django
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
