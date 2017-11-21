# powerai-collision-near-miss

# *** WORK-IN-PROGRESS ***

* In the current version, we used a .json file or an array to represent the crosspath.

* There are only two labels in this scenario from the API, mostly "car" and some "people". I have specially treated "people" and all others are "car".

* for prerequisite, there is no special requirement except ffmpeg and opencv 2.x. I am developing on ubuntu 16.04 and python 2.7, pip install opencv-python works.

* For development, what operating system are you using? I would suggest Ubuntu 14.04 or 16.04, Windows should also work, but I haven't tested it yet.

## Example Usage:

```commandline
python near_miss_demo_code.py process_stream --video_src rtsp://vssod.dot.ga.gov:80/hi/atl-cam-086.stream --crossway_file json/online_cam_86.json 
```

## Demo Environment

PowerAI Vision is available on Super Vessel (IBM Research Cloud):
- https://ptopenlab.com/cloudlabconsole/#/ (registration button is on the right top of the page)
- After registration, AI Vision is accessible at .  https://ny1.ptopenlab.com/AIVision/index.html

or

- We provide technology previews for our customers to download on Power servers https://developer.ibm.com/linuxonpower/deep-learning-powerai/technology-previews/powerai-vision/ and them out for 90 days.
