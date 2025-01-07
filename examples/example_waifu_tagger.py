from PIL import Image
from urllib.request import urlopen
from waifuset import WaifuTagger

waifu_tagger = WaifuTagger.from_pretrained("SmilingWolf/wd-eva02-large-tagger-v3")
img_url = "https://cdn.donmai.us/sample/92/13/__lucy_cyberpunk_and_1_more_drawn_by_bigroll__sample-92136a201e8ff2a1f9773537535bb2ff.jpg"
image = Image.open(urlopen(img_url))
tags = waifu_tagger(image)
print(tags)
