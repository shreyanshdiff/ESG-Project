FROM python:3.10

WORKDIR  /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501 5000

RUN echo '#!/bin/bash\n\
python apps/app.py & \n\
streamlit run apps/streamlit_app.py\n\
' > start.sh && chmod +x start.sh

ENV GROQ_API_KEY = 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

CMD ["./start.sh"]

