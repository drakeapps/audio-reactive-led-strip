FROM balenalib/raspberrypi3:buster

RUN apt-get update && apt-get install -y \
	python3 python3-pip python3-numpy python3-scipy python3-pyaudio build-essential

# fix the numpy build
# RUN pip3 install 'numpy==1.16' --force-reinstall

COPY build/asound.conf /etc/asound.conf

RUN sed -i "s|defaults.ctl.card 0|defaults.ctl.card 1|g" /usr/share/alsa/alsa.conf
RUN sed -i "s|defaults.pcm.card 0|defaults.pcm.card 1|g" /usr/share/alsa/alsa.conf

COPY python /code/python

WORKDIR /code/python

RUN pip3 install -r requirements.txt --force-reinstall

CMD python3 visualization.py
