
#!/bin/bash

## open Launch in VSCode

## From https://circuitdigest.com/microcontroller-projects/raspberry-pi-based-emotion-recognition-using-opencv-tensorflow-and-keras

## Need to open docker desktop first, then:

```
docker run -it -p 1883:1883 eclipse-mosquitto:2.0 mosquitto -c /mosquitto-no-auth.conf
```

## This must be in a separate terminal and kept running

## activate virtual environment
source .venv/bin/activate

## go to directory with code
# cd fall-detect

## Meraki

python falling-data.py


## Webcam
## run emotion1.py for the webcam