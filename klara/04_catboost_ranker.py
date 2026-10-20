import argparse
import pandas as pd
from src.training import ROOT,fit
p=argparse.ArgumentParser(); p.add_argument('--model',default='ranker_base'); p.add_argument('--iterations',type=int,default=700); a=p.parse_args()
fit(a.model,pd.read_csv(ROOT.parent/'train.csv'),pd.read_csv(ROOT.parent/'validation.csv'),iterations=a.iterations)
