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

There are two scripts provided here to achieve different functions. If you only want to replace one template, use the `single_detect.py` script; if you want to replace multiple templates, use the `multi_detect.py` script.

At the begin, you need to prepaer images, and the directory structure of the files is as follows:

```bash
(ar_replace) $ tree
.
├── ReadMe.md
├── single_detect.py
├── multi_detect.py
└── resources
    ├── replace_template_1.png
    ├── replace_template_2.png
    ├── template_1.png
    └── template_2.png

2 directories, 5 files
```


## Single Detect
Single template detection and replacement.

### Step1. Prepare images
You need to prepare at least 2 images, one for template A and the other for the replacement template B. In this example the two files are `template_1.png` and `replace_template.png`.

If you want to use other files, you can also find those part in `single_detect.py` to specify your file path

```python
template_file = "./resources/template_1.png"
replace_template_file = "./resources/replace_template_1.png"
```

then run the script:
```bash
(ar_replace) $ python single_detect.py
```

![single_detect](./resources/single_detect.gif)


----
## Multi Detect
Multiple template detection and replacement

```bash
(ar_replace) $ python multi_detect.py
```

You can also modify the detection template and parameters in the source code:

```python
# Global Params
DEFAULT_PARAMS = {
    "MIN_MATCH_COUNT": 10,
    "LOWE_RATIO_THRESHOLD": 0.7,
    "RANSAC_REPROJ_THRESHOLD": 4.0,
    "MIN_ANGLE_THRESHOLD": 35.0,
    "MAX_ANGLE_THRESHOLD": 130.0
}

...

def main():
    ...
    TEMPLATE_PAIRS = {
        "book_cover": {
            "template_path": "./resources/template_1.png",
            "replace_path": "./resources/replace_template_1.png",
            # 此模板未使用自定义参数，将全部采用 DEFAULT_PARAMS 的值
        },
        "card": {
            "template_path": "./resources/template_2.png",
            "replace_path": "./resources/replace_template_2.png",
            "params": {
                # 这个模板比较小，特征少，我们放宽一些条件
                "MIN_MATCH_COUNT": 10,          # 降低最小匹配数要求
                "LOWE_RATIO_THRESHOLD": 0.70,   # 允许质量稍差一些的匹配
                "MIN_ANGLE_THRESHOLD": 25.0,    # 允许更大的透视形变
                "MAX_ANGLE_THRESHOLD": 130,
            }
        },
    }
    ...
```

![multi_detect](./resources/multi_detect.gif)