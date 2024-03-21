import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin

url = "http://127.0.0.1:7860"

payload = {
    "prompt": "best quality, ultra high res, (photorealistic:1.4), RAW photo, (upper waist:1.4,wide shot:1.2), (bokeh:1.7),(Window light only:1.6), (in bright coffee shop:1.5),(bright:1.5), sit, upscale restaurant, candlelit dinner, romantic, elegance, fine dining, ambiance, soft lighting, intimate, 1japanese girl, (solo:1.95), (smile:1.2,close mouth:0.7), (dark brown eyes), natural skin, (brown medium hair,bangs),standing, (leaning forward), (white thin sleeveless shirt:1.5),(dark color skirt:1.2)",
    "negative_prompt": "(extra nipples), (Supernumerary nipple),double navel,dark mole, painting,sketches,(worst quality:2),(low quality:2),(normal quality:2),(monochrome),(grayscale), missing fingers ,skin spots ,acnes,skin blemishes,loli",
    "steps": 25,
    "sampler_index": "DPM++ 2M Karras",
    "width": 512,
    "height": 786,
}

response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

r = response.json()
for i in r['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

    png_payload = {
        "image": "data:image/png;base64," + i
    }
    response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", response2.json().get("info"))
    image.save('output/output.png', pnginfo=pnginfo)
