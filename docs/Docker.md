Docker
======================================

On Linux, the Dockerfile can be used to run the GUI of Spirit.
In general, it can be used to run the core library, e.g. using Python.

The process can be

```
sudo docker build -t spirit .
```

then

```
xhost +
sudo docker run -ti --rm --device=/dev/dri:/dev/dri -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix spirit
```

which uses the users X11 sessions (assuming that uid and gid of the host are 1000).