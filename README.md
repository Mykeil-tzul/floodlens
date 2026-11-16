# FloodLens

[Open the historical flood replay](https://mykeil-tzul.github.io/floodlens/)

FloodLens lets you inspect what a model sees in Sentinel-1 radar during a flood. Open a tile from the January 2018 France flood activation and compare the radar, predicted flood area, reference annotation, and a simple radar threshold. Sort by misses or false alarms to find difficult examples.

## Training and evaluation

I sampled 15 paired radar and mask chips per activation from [MMFlood](https://zenodo.org/records/6534637). The model trains on three activations and evaluates on 15 sampled chips from a separate activation, [EMSR265](https://mapping.emergency.copernicus.eu/activations/EMSR265/). None of that activation’s tiles enter training or threshold selection. The sample is a small experiment and does not estimate performance across all floods.

The model reads Sentinel-1 VV and VH intensity, converts it to a clipped log scale, and predicts flood masks with a small convolutional network. I choose both methods’ cutoffs using training pixels. On the sampled France activation, the CNN reached 21.9% IoU, 31.7% precision, and 41.5% recall. The radar threshold reached 34.8% IoU, 46.7% precision, and 57.6% recall. The first unweighted model predicted no flood pixels at its default cutoff. I inspected that result during development, then used class weighting and selected the final cutoff on training data. The France activation never enters fitting, but since I inspected an early result, these scores are exploratory rather than a blind final benchmark. Both use 128×128 pixels per chip for this demo. Read precision, recall, and IoU alongside each replay because aggregate results can hide failures on individual tiles.

The static site serves precomputed outputs. This is a historical research demo. The maps are unsuitable for routing, emergency response, or decisions about current flooding.

## Reproduce

```sh
python -m pip install -r requirements.txt
python fetch_mmflood.py data
python train.py data
python -m unittest discover -s tests
python -m http.server 8000 --directory site
```

The downloader fetches byte ranges from Zenodo’s archive and checks each extracted file against its ZIP checksum. Radar and label GeoTIFFs are fetched separately and are not bundled in the repo.

## Credits

Radar and reference masks are adapted from [MMFlood by Fabio Montello, Edoardo Arnaudo, and Claudio Rossi](https://zenodo.org/records/6534637), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preview images were resized, contrast-scaled, and colorized. Event identifiers and source mapping come from Copernicus Emergency Management Service. Background map tiles are by [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
