from PIL import Image
from urllib.request import urlopen
from waifuset import WaifuScorer

waifu_scorer = WaifuScorer.from_pretrained("Eugeoter/waifu-scorer-v4-beta")
img_url = "https://cdn.donmai.us/sample/92/13/__lucy_cyberpunk_and_1_more_drawn_by_bigroll__sample-92136a201e8ff2a1f9773537535bb2ff.jpg"
image = Image.open(urlopen(img_url))
score = waifu_scorer(image)
print(score)
