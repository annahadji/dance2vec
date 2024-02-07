# dance2vec

## Running the model

## Analysing the data

```
(.venv) [17:31:50] 🚀 roslin-2022-analysis $ python build_dataset.py --help
usage: build_dataset.py [-h] [--build_errors_ds] [--data_path DATA_PATH] [--file_out FILE_OUT] [--nbins NBINS] [--antenna_len_str {default,mid_only,full_only}]

Build antennal positioning dataset.

optional arguments:
  -h, --help            show this help message and exit
  --build_errors_ds     Build error dataset instead of antennae dataset.
  --data_path DATA_PATH
                        Name of raw data folder. Defaults to 'Cropped-Anna Hadjitofi-2022-11-01'. For building errors dataset, this should be the path to the folder containing the experiments.
  --file_out FILE_OUT   Resulting name (and path) to save dataset(s), without extension. Defaults to current date.
  --nbins NBINS         Number of bins to use for binning angles to dancer. Defaults to 180.
  --antenna_len_str {default,mid_only,full_only}
                        Calculate antenna angle using mid length / bend or full length. The default uses base to tip if available and midpoint as fallback.
```

## Built With
