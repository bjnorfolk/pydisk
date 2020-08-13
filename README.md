# pydisk

pydisk is a python package that contains all the useful code I've had to develop for my PhD in protoplanetary disks. This is by no means extensive and is aimed mostly at masters and new PhD students. I can make no promises on the timely manner that I commit new code or the quality of my code.

pydisk offers (or will offer) the following functionalities:

 - plot disk continuum maps using image fits
 - and much more ...


## Installation:

```
git clone https://github.com/bjnorfolk/pydisk.git
cd pydisk
python3 setup.py install
```

If you don't have the `sudo` rights, use `python3 setup.py install --user`.

To install in developer mode: (i.e. using symlinks to point directly
at this directory, so that code changes here are immediately available
without needing to repeat the above step):

```
 python3 setup.py develop
```
