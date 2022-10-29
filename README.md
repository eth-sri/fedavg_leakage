# Data Leakage in Federated Averaging <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>


The code accompanying our TMLR 10/2022 paper: [**Data Leakage in Federated Averaging**](https://openreview.net/forum?id=e7A0B99zJf).

## Requirements

Install [Anaconda](https://conda.io/) and execute the following command:
```
conda env create --name fedavg --file fedavg/fedavg.yml
conda activate fedavg
```

Further, you need to download FEMNIST with the following commands:

```
cd data/
git clone https://github.com/TalwalkarLab/leaf.git
cd leaf/data/femnist
./preprocess.sh -s niid --sf 0.05 -k 0 -t sample
cd ../../../
mv leaf/data/femnist ./
rm -rf leaf
cd ../
```

### Running experiments

We provide scripts to reproduce all of our tables in the scripts folder. For example you can run FEMNIST experiments in Table 1 as follows:
```
cd fedavg
scripts/reconstruct_femnist_table1.sh
``` 

Optionally, to visualize the results better, you can run the experiments using [Neptune](https://neptune.ai/). To do so add the arguments NEPTUNE_API_TOKEN and NEPTUNE_PROJECT_NAME at the end of your command, as follows:
```
cd fedavg
scripts/reconstruct_femnist_table1.sh NEPTUNE_API_TOKEN NEPTUNE_PROJECT_NAME
```

To train the defended networks used to produce Table 6, execute the following:
```
cd fedavg
scripts/train_femnist.sh
```

## Citation

```
@inproceedings{
    dimitrov2022data,
    title={Data Leakage in Federated Averaging},
    author={Dimitrov, Dimitar I and Balunovi{\'c}, Mislav and Konstantinov, Nikola and Vechev, Martin},
    booktitle={Transactions on Machine Learning Research},
    year={2022},
    url={https://openreview.net/forum?id=e7A0B99zJf},
}
```
