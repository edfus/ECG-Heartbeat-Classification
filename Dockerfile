FROM tensorflow/tensorflow:2.7.0-gpu
WORKDIR /usr/local/src
VOLUME [ "/usr/local/src" ]

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN groupadd -g 1810 py \
&& useradd -m -u 1820 -g py py
USER py

COPY --chown=py:py "requirements.txt" "/usr/local/src/requirements.txt"
RUN pip3 install -r requirements.txt

CMD [ "python", "main.py" ]