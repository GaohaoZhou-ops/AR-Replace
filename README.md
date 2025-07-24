# AR Replace

The function of this repository is to replace image template A in the video stream with image template B.

---
# Dependencies

```bash
$ conda create -n ar_replace python=3.8
$ conda activate ar_replace
```

install `opencv-python` and `numpy`

```bash
(ar_replace) $ pip install opencv-python numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---
# How to Use

## Step1. Prepare images
You need to prepare at least 2 images, one for template A and the other for the replacement template B. In this example the two files are `template_1.png` and `replace_template.png`.

The directory structure of the files is as follows:
```bash
(ar_replace) $ tree
.
├── ReadMe.md
├── main.py
└── resources
    ├── replace_template.png
    ├── template_1.png
    └── template_2.png

2 directories, 5 files
```

If you want to use other files, you can also find the line `template_a, template_b_composite, kp_a, des_a = load_templates('./resources/template_1.png', './resources/template_b.png')` in main.py to specify your file path

## Step2. Run scripts
```bash
(ar_replace) $ python main.py
```

![single_detect](./resources/single_detect.gif)

