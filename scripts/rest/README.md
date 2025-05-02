# IndividualParcellation - Resting-state analysis
RestLocalization project using Functional_Fusion & HierarchBayesParcel

## Scripts

Individual parcellations of MDTB subjects using rest, rest+HCP-Vs (Connectivity fingerprints learned from HCP data) and task:
```scripts/rest/fit_rest.py``` 

Evaluating rest, rest+HCP-Vs and task parcellations:
```scripts/rest/evaluation.py``` 

Comparing connectivity fingerprints of cerebellar-only networks with cortico-cerebellar networks:
```scripts/rest/cerebellar_networks.py``` 


## Notebooks

Flatmap plots of individual parcellations of MDTB subjects using rest, rest+HCP-Vs and task:
```notebooks/rest/variance.ipynb```

Plots of fit diagnostics:
```notebooks/rest/fit_diagnostics.ipynb``` 

Evaluation plots of DCBC and cosine error evaluation of rest, rest+HCP-Vs and task parcellations:
```notebooks/rest/evaluation.ipynb``` 

Exploration of the resting-state networks derived from the HCP groupICA:
```notebooks/rest/networks.ipynb``` 

