import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin
import cv2

url = "http://127.0.0.1:7860"

img_path = "input/input.png"
image = Image.open(img_path)

# バイナリデータからテキスト変換
with io.BytesIO() as img_bytes:
    image.save(img_bytes, format="PNG")
    img_b64 = base64.b64encode(img_bytes.getvalue()).decode()

png_data = {}
png_data["image"] = [img_b64]

model = "Counterfeit-V3.0.safetensors [db6cd0a62d]"
prompt = "(masterpiece best quality ultra-detailed illustration) a girl silver hair"
neg_prompt = "nsfw, (worst quality, low quality:1.4), (jpeg artifacts:1.4), (depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare:1.0)"
steps = 25

payload = {
    "init_images": png_data["image"],
    "sd_model_checkpoint": model,
    "prompt": prompt,
    "negative_prompt": neg_prompt,
    "sampler_index": "DPM++ 2M Karras",
    "width": 512,
    "height": 786,
    "steps": steps,
}


img = cv2.imread("input/input.png")
retval, bytes = cv2.imencode(".png", img)
encoded_image = base64.b64encode(bytes).decode("utf-8")

# controlnet
payload["alwayson_scripts"] = {
    "controlnet": {
        "args": [
            {
                # "input_image": encoded_image, # 画像を使う場合使用
                "module": "openpose_face",
                "model": "control_v11p_sd15_openpose [cab727d4]",
                "pixel_perfect": True,
                "control_mode": 1,
            }
        ]
    }
}

response = requests.post(url=f"{url}/sdapi/v1/img2img", json=payload)

r = response.json()
print(r)

# controlnetの結果画像あり
for num, i in enumerate(r["images"]):
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))

    png_payload = {"image": "data:image/png;base64," + i}
    response2 = requests.post(url=f"{url}/sdapi/v1/png-info", json=png_payload)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", response2.json().get("info"))
    image.save(f"output/{num}output_cn_i2i.png", pnginfo=pnginfo)
    image.show()

# controlnetの結果画像なし
# image = Image.open(io.BytesIO(base64.b64decode(r['images'][0].split(",",1)[0])))
# image.save('output_cn_i2i.png')
# image.show()
