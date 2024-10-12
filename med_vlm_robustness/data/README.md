# Preprocessing the Data

## SLAKE
To download the SLAKE dataset, go to [this](https://www.med-vqa.com/slake/) website and choose your preferred download option. In the end, the structure of the downloaded dataset should be 

```
SLAKE
└── imgs
└── KG
└── mask.txt
└── train.json
└── validate.json
└── test.json
```

## OVQA
The OVQA dataset can be downloaded [here](http://47.94.174.82/). Note that you have to conduct a CITI data privacy course and be credentialed in order to access the dataset. After downloading the dataset, you should bring it into the following structure: 

```
OVQA
└── img
└── train.json
└── validate.json
└── test.json
```

After downloaded everything and set it up in the structure as described above, run
```
python ovqa_clean.py -p <path/to/OVQA>
```
This will create a ``train.json``, ``validate.json``, and ``test.json``with the corresponding questions in the OVQA folder. The original files will be saved as ``train_old.json``, ``validate_old.json``, and ``test_old.json``

## MIMIC-CXR-VQA

In order to access the MIMIC-CXR-VQA dataset, you have to be a credentialed used on [Physionet](https://physionet.org).  
The exact information how to download the dataset and construct the questions can be found [here](https://github.com/baeseongsu/mimic-cxr-vqa). After following these steps, your folder structure should look like this:

```
MIMIC
└── mimic-cxr-vqa
    └── dataset_builder
    └── mimiccxrvqa
        └── dataset
            ├── ans2idx.json
            ├── _train_part1.json
            ├── _train_part2.json
            ├── _valid.json
            ├── _test.json
            ├── train.json (available post-script execution)
            ├── valid.json (available post-script execution)
            └── test.json  (available post-script execution)
    └── physionet.org
        └── files
            ├── chest-imagenome
                └── 1.0.0
                    ├── gold_dataset
                    ├── semantics
                    ├── silver_dataset
                    └── utils
            ├── mimic-cxr-jpg
                └── 2.0.0
                    ├── files
                    └── mimic-cxr-2.0.0-metadata.csv
            └── mimic-iv
                └── 2.2
                    └── hosp
                        ├── files (!Download separately)
                        └── patients.csv
```
Here,  the admission.csv file from the mimic-iv dataset needs to be downloaded separately and put into ```physionet.org/files/mimiv-iv/2.2/hosp```. You can find this file [here](https://physionet.org/content/mimiciv/2.2/hosp/admissions.csv.gz).  
Also, make sure to run
```
bash download_images.sh
```
from [here](https://github.com/baeseongsu/mimic-cxr-vqa?tab=readme-ov-file#downloading-mimic-cxr-jpg-images) to download the images from the MIMIC-CXR dataset that are relevant for the asked questions.  

After downloaded everything and set it up in the structure as described above, run
```
python mimic_cxr_vqa_data.py -p <path/to/MIMIC>
```
This will create a ``train.json``, ``validate.json``, and ``test.json``with the corresponding questions in the MIMIC folder.