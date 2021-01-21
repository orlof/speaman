# +
import os
import math
import time

import numpy
import cv2
import pytesseract
import mss
import regex
import pynput

import craft_text_detector as craft

from torch import cuda
from IPython.display import HTML, Image


def get_screen_resolution(measurement="px"):
    """
    Tries to detect the screen resolution from the system.
    @param measurement: The measurement to describe the screen resolution in. Can be either 'px', 'inch' or 'mm'. 
    @return: (screen_width,screen_height) where screen_width and screen_height are int types according to measurement.
    """
    mm_per_inch = 25.4
    px_per_inch = 72.0  # most common
    
    try:  # Platforms supported by GTK3, Fx Linux/BSD
        from gi.repository import Gdk 
        screen = Gdk.Screen.get_default()
        if measurement == "px":
            width = screen.get_width()
            height = screen.get_height()
        elif measurement == "inch":
            width = screen.get_width_mm()/mm_per_inch
            height = screen.get_height_mm()/mm_per_inch
        elif measurement == "mm":
            width = screen.get_width_mm()
            height = screen.get_height_mm()
        else:
            raise NotImplementedError("Handling %s is not implemented." % measurement)
        return (width, height)
    except:
        try: #Probably the most OS independent way
            if PYTHON_V3: 
                import tkinter 
            else:
                import Tkinter as tkinter
            root = tkinter.Tk()
            if measurement=="px":
                width = root.winfo_screenwidth()
                height = root.winfo_screenheight()
            elif measurement=="inch":
                width = root.winfo_screenmmwidth()/mm_per_inch
                height = root.winfo_screenmmheight()/mm_per_inch
            elif measurement=="mm":
                width = root.winfo_screenmmwidth()
                height = root.winfo_screenmmheight()
            else:
                raise NotImplementedError("Handling %s is not implemented." % measurement)
            return (width,height)
        except:
            try: #Windows only
                from win32api import GetSystemMetrics 
                width_px = GetSystemMetrics (0)
                height_px = GetSystemMetrics (1)
                if measurement=="px":
                    return (width_px,height_px)
                elif measurement=="inch":
                    return (width_px/px_per_inch,height_px/px_per_inch)
                elif measurement=="mm":
                    return (width_px/mm_per_inch,height_px/mm_per_inch)
                else:
                    raise NotImplementedError("Handling %s is not implemented." % measurement)
            except:
                try: # Windows only
                    import ctypes
                    user32 = ctypes.windll.user32
                    width_px = user32.GetSystemMetrics(0)
                    height_px = user32.GetSystemMetrics(1)
                    if measurement=="px":
                        return (width_px,height_px)
                    elif measurement=="inch":
                        return (width_px/px_per_inch,height_px/px_per_inch)
                    elif measurement=="mm":
                        return (width_px/mm_per_inch,height_px/mm_per_inch)
                    else:
                        raise NotImplementedError("Handling %s is not implemented." % measurement)
                except:
                    try: # Mac OS X only
                        import AppKit 
                        for screen in AppKit.NSScreen.screens():
                            width_px = screen.frame().size.width
                            height_px = screen.frame().size.height
                            if measurement=="px":
                                return (width_px,height_px)
                            elif measurement=="inch":
                                return (width_px/px_per_inch,height_px/px_per_inch)
                            elif measurement=="mm":
                                return (width_px/mm_per_inch,height_px/mm_per_inch)
                            else:
                                raise NotImplementedError("Handling %s is not implemented." % measurement)
                    except: 
                        try: # Linux/Unix
                            import Xlib.display
                            resolution = Xlib.display.Display().screen().root.get_geometry()
                            width_px = resolution.width
                            height_px = resolution.height
                            if measurement=="px":
                                return (width_px,height_px)
                            elif measurement=="inch":
                                return (width_px/px_per_inch,height_px/px_per_inch)
                            elif measurement=="mm":
                                return (width_px/mm_per_inch,height_px/mm_per_inch)
                            else:
                                raise NotImplementedError("Handling %s is not implemented." % measurement)
                        except:
                            try: # Linux/Unix
                                if not self.is_in_path("xrandr"):
                                    raise ImportError("Cannot read the output of xrandr, if any.")
                                else:
                                    args = ["xrandr", "-q", "-d", ":0"]
                                    proc = subprocess.Popen(args,stdout=subprocess.PIPE)
                                    for line in iter(proc.stdout.readline,''):
                                        if isinstance(line, bytes):
                                            line = line.decode("utf-8")
                                        if "Screen" in line:
                                            width_px = int(line.split()[7])
                                            height_px = int(line.split()[9][:-1])
                                            if measurement=="px":
                                                return (width_px,height_px)
                                            elif measurement=="inch":
                                                return (width_px/px_per_inch,height_px/px_per_inch)
                                            elif measurement=="mm":
                                                return (width_px/mm_per_inch,height_px/mm_per_inch)
                                            else:
                                                raise NotImplementedError("Handling %s is not implemented." % measurement)
                            except:
                                # Failover
                                screensize = 1366, 768
                                sys.stderr.write("WARNING: Failed to detect screen size. Falling back to %sx%s" % screensize)
                                if measurement=="px":
                                    return screensize
                                elif measurement=="inch":
                                    return (screensize[0]/px_per_inch,screensize[1]/px_per_inch)
                                elif measurement=="mm":
                                    return (screensize[0]/mm_per_inch,screensize[1]/mm_per_inch)
                                else:
                                    raise NotImplementedError("Handling %s is not implemented." % measurement)


def notebook_image(temp, file, comment=None):
    if comment is None:
        comment = file

    with open(os.path.join(temp, file), 'rb') as f:
        image = f.read()

    display(create_gallery([(image, comment)]))

def notebook_image_marked(image, *marks, comment=""):
    image = image.copy()
    for center in marks:
        image = cv2.circle(image, center, 20, (0, 0, 255), 3)
    is_success, im_buf_arr = cv2.imencode(".png", image)
    byte_im = im_buf_arr.tobytes()

    display(create_gallery([(byte_im, comment)]))

def _src_from_data(data):
    """Base64 encodes image bytes for inclusion in an HTML img element"""
    img_obj = Image(data=data)
    for bundle in img_obj._repr_mimebundle_():
        for mimetype, b64value in bundle.items():
            if mimetype.startswith('image/'):
                return f'data:{mimetype};base64,{b64value}'


def create_gallery(images, row_height='auto'):
    figures = []
    for image, caption in images:
        src = _src_from_data(image)

        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="{src}" style="height: {row_height}">
              {caption}
            </figure>
        ''')
    return HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    ''')
# +
class spaeman:
    def __init__(self, debug=False):
        self.debug = debug

        self.bgr = None
        self.rgb = None
        self.grey = None

        self.dx = 1.0
        self.dy = 1.0

        self.text_areas = None
        self.text_lines = None

        self.image_path = 'screen_shot.png'
        self.temp_dir = "temp/"
        self.num_searches = 0

        self.refine_net = None
        self.craft_net = None

        self.cuda = cuda.is_available()
        self.mouse = pynput.mouse.Controller()

    def freeze(self):
        self.screenshot()
        self.detect_text()
        self.recognize_text()
        
    def screenshot(self, monitor=1):
        with mss.mss() as sct:
            monitor = sct.monitors[monitor]
            bgra = numpy.array(sct.grab(monitor))

        self.bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        self.rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)

        width, height = get_screen_resolution()
        self.dx = width / self.rgb.shape[1]
        self.dy = height / self.rgb.shape[0]

        if self.debug:
            f = os.path.join(self.temp_dir, self.image_path)
            cv2.imwrite(f, self.bgr)
            notebook_image(self.temp_dir, self.image_path)

    def detect_text(self):
        if not self.refine_net:
            self.refine_net = craft.load_refinenet_model(cuda=self.cuda)
            self.craft_net = craft.load_craftnet_model(cuda=self.cuda)

        # perform prediction
        self.text_areas = craft.get_prediction(
            image=self.rgb,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=self.cuda,
            long_size=1280
        )

        if self.debug:
            # export heatmap, detection points, box visualization
            craft.export_extra_results(
                image_path=self.image_path,
                image=self.rgb,
                regions=self.text_areas["boxes"],
                heatmaps=self.text_areas["heatmaps"],
                output_dir=self.temp_dir
            )

            file = os.path.splitext(self.image_path)[0]
            notebook_image(self.temp_dir, "%s_text_detection.png" % (file))
            notebook_image(self.temp_dir, "%s_text_score_heatmap.png" % (file))
            notebook_image(self.temp_dir, "%s_link_score_heatmap.png" % (file))

    def recognize_text(self):
        self.text_lines, images = [], []
        for (x0, y0), (x1, y1), (x2, y2), (x3, y3) in self.text_areas['boxes']:
            center = round((x0 + x1) / 2), round((y0 + y2) / 2)
            region = self.bgr[round(y0):round(y2), round(x0):round(x2)]
            d = pytesseract.image_to_data(region, output_type=pytesseract.Output.DICT)

            words = []
            for text, conf, x, y, w, h in zip(d['text'], map(float, d['conf']), d['left'], d['top'], d['width'], d['height']):
                if conf > 0 and text:
                    words.append({
                        "text": text.strip(),
                        "conf": conf,
                        "center": (round(x0 + x + w / 2), round(y0 + y + h / 2))
                    })

            if words:
                line = {
                    "center": center,
                    "top": round(y0), "bottom": round(y2),
                    "left": round(x0), "right": round(x1),
                    "text": " ".join([word["text"] for word in words]),
                    "words": words,
                    "conf": sum([word["conf"] for word in words]) // len(words)
                }
                self.text_lines.append(line)

                if self.debug:
                    is_success, im_buf_arr = cv2.imencode(".png", region)
                    byte_im = im_buf_arr.tobytes()
                    images.append((byte_im, line["text"]))

        if self.debug:
            display(create_gallery(images))

    def _store_search_image(self, image, comment=None):        
        filename = f"{self.num_searches:03}_search.png"
        self.num_searches += 1
        f = os.path.join(self.temp_dir, filename)
        cv2.imwrite(f, image)
        notebook_image(self.temp_dir, filename, comment=comment)
    
    def search_text(self, pattern, distance=0):
        pattern = r'(?e)(%s){e<=%d}' % (pattern, distance)
        matches = [match for match in self.text_lines if regex.search(pattern, match["text"], flags=regex.IGNORECASE)]

        if self.debug:
            image = self.bgr.copy()
            for d in matches:
                image = cv2.rectangle(image, (d["left"], d["top"]), (d["right"], d["bottom"]), (0, 0, 255), 3)
                image = cv2.circle(image, d["center"], 20, (0, 0, 255), 3)
            self._store_search_image(image, pattern)

        return matches

    def search_word(self, pattern, distance=0):
        pattern = r'(?e)(%s){e<=%d}' % (pattern, distance)
        matches = [match for sublist in [line["words"] for line in cm.text_lines] for match in sublist if regex.search(pattern, match["text"], flags=regex.IGNORECASE)]

        if self.debug:
            image = self.bgr.copy()
            for d in matches:
                image = cv2.circle(image, d["center"], 20, (0, 0, 255), 3)
            self._store_search_image(image, pattern)

        return matches

    def search_image(self, filename, threshold=0.8, best_match=False):
        if self.grey is None:
            self.gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(self.gray, template, cv2.TM_CCOEFF_NORMED)
        
        matches = []
        if best_match:
            _, val, _, loc = cv2.minMaxLoc(res)
            if val >= threshold:
                matches.append({
                    "left": loc[0], "right": loc[0] + w,
                    "top": loc[1], "bottom": loc[1] + h,
                    "center": (loc[0] + w // 2, loc[1] + h // 2)
                })
        else:
            loc = numpy.where(res >= threshold)

            for pt in zip(*loc[::-1]):
                cx, cy = pt[0] + w // 2, pt[1] + h // 2
                if not any([d["left"] < cx < d["right"] and d["top"] < cy < d["bottom"] for d in matches]):
                    matches.append({
                        "left": pt[0], "right": pt[0] + w,
                        "top": pt[1], "bottom": pt[1] + h,
                        "center": (cx, cy)
                    })
        
        if self.debug:
            bgr = self.bgr.copy()
            for d in matches:
                cv2.rectangle(bgr, (d["left"], d["top"]), (d["right"], d["bottom"]), (0,0,255), 3)

            self._store_search_image(bgr, filename)

        return matches

    def _button(self, name):
        return {
            "left": pynput.mouse.Button.left, 
            "right": pynput.mouse.Button.right, 
            "middle": pynput.mouse.Button.middle
        }[name.lower()]
    
    def mouse_position(self, x, y):
        if self.debug:
            notebook_image_marked(self.bgr, (x, y))
        self.mouse.position = (x * self.dx, y * self.dy)
    
    def mouse_press(self, button="left"):        
        self.mouse.press(_button(button))
    
    def mouse_release(self, button="left"):
        self.mouse.release(_button(button))
    
    def mouse_click(self, button="left", count="1"):
        self.mouse.click(_button(button), int(count))
    
    def mouse_scroll(self, dx, dy):
        self.mouse.scroll(dx, dy)

    def mouse_click_position(self, x, y, button="left", count="1"):
        if self.debug:
            notebook_image_marked(self.bgr, (x, y))
        self.mouse_position(x, y)
        self.mouse_click(button, count)
        
    def click_text(self, text):
        result = self.search(text)
        if result:
            self.click(*result[0]["center"])
        else:
            notebook.notebook_print("%s no match")

    def unload_models(self):
        if self.refine_net:
            # unload models from gpu
            craft.empty_cuda_cache()
            self.refine_net = None
            self.craft_net = None


cm = spaeman(debug=False)
cm.freeze()

# -

cm.debug = True
cm.search_word("spaeman")

cm.debug = True
cm.search_image("button.png", threshold=.7, best_match=False)


