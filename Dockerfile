FROM balenalib/raspberrypi3:buster

RUN apt-get update && apt-get install -y \
	python3 python3-pip python3-dev python3-setuptools build-essential git python3-pyaudio python3-numpy python3-scipy libatlas-base-dev

# fix the numpy build
# RUN pip3 install 'numpy==1.16' --force-reinstall

COPY build/asound.conf /etc/asound.conf

RUN sed -i "s|defaults.ctl.card 0|defaults.ctl.card 1|g" /usr/share/alsa/alsa.conf
RUN sed -i "s|defaults.pcm.card 0|defaults.pcm.card 1|g" /usr/share/alsa/alsa.conf

COPY requirements.txt /code/python/requirements.txt

WORKDIR /code/python

RUN pip3 install -r requirements.txt --force-reinstall

COPY AudioReactiveLEDStrip /code/python

WORKDIR /code/python


CMD python3 visualization.py
