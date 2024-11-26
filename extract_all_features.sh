#! /usr/bin/bash

python3 feature_extraction.py -d datasets/KDD-TSAD/ --output-path features/features_KDD-TSAD.csv --n-jobs 20
python3 feature_extraction.py -d datasets/MGAB/ --output-path features/features_MGAB.csv --n-jobs 20
python3 feature_extraction.py -d datasets/NAB/ --output-path features/features_NAB.csv --n-jobs 20
python3 feature_extraction.py -d datasets/NASA-MSL/ --output-path features/features_NASA-MSL.csv --n-jobs 20
python3 feature_extraction.py -d datasets/NASA-SMAP/ --output-path features/features_NASA-SMAP.csv --n-jobs 20
python3 feature_extraction.py -d datasets/NormA/ --output-path features/features_NormA.csv --n-jobs 20

