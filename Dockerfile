FROM ubuntu:jammy as build

COPY requirements.txt /requirements.txt

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \ 
    python3-pip git x11-apps \
    && pip install -r requirements.txt

COPY main.py /main.py

RUN chmod +x /main.py

# Final stage build
FROM scratch

COPY --from=build / /

CMD ["./main.py"]