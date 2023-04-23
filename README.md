
```console
python train.py -f bsnet4000.pth
python predict.py -b bsnet4000.pth -e CP_epoch1445.pth
python predict_sample.py -b bsnet4000.pth -e CP_epoch1445.pth -o underwater/org_11.png -d underwater/distorted_11.png
```
