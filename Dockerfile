FROM tensorflow/tensorflow:2.7.0-gpu
WORKDIR /usr/local/src
VOLUME [ "/usr/local/src" ]
USER node
RUN groupadd -g 1810 py \
&& useradd -m -u 1820 -g py py
COPY --chown=py:py "requirements.txt" "/usr/local/src/requirements.txt"
RUN pip install -r requirements.txt
CMD [ "python", "main.py" ]