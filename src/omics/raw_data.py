import os
import zipfile
import shutil
import urllib.request as request
from contextlib import closing

output_folder = f'data/omics/raw_data/'

def download_file(url):
    # collect file
    with closing(request.urlopen(url)) as r:
        with open(f'{output_folder}{url.split("/")[-1]}', 'wb') as f:
            shutil.copyfileobj(r, f)

    if url.split('.')[-1] == 'zip':
        filename = url.split("/")[-1]
        with zipfile.ZipFile(f'{output_folder}{filename}', 'r') as zip_ref:
            zip_ref.extractall(f'{output_folder}{filename.split(".")[-2]}/')
    
if __name__ == "__main__":


    """ 
    Dataset details from 
    ===================

    File Name: 
    Description: 

    """

    datasets = {
        'Collins_MND_CSF': "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5592736/bin/NIHMS898604-supplement-Table_S-1.xlsx",
        'Bereman_ALS_CSF': 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6218542/bin/41598_2018_34642_MOESM1_ESM.xlsx',

        'Bai_AD_CSF': 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7318843/bin/NIHMS1569542-supplement-4.xlsx', #sheet C
        'Bader_AD_CSF': 'https://www.embopress.org/action/downloadSupplement?doi=10.15252%2Fmsb.20199356&file=msb199356-sup-0005-DatasetEV3.xlsx',
        'Bader_AD_samplemap': 'https://www.embopress.org/action/downloadSupplement?doi=10.15252%2Fmsb.20199356&file=msb199356-sup-0004-DatasetEV2.xlsx',

        "DAlessandro_COVID_serum": 'https://pubs.acs.org/doi/suppl/10.1021/acs.jproteome.0c00365/suppl_file/pr0c00365_si_002.xlsx',
        "Di_COVID_serum": 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41392-020-00333-1/MediaObjects/41392_2020_333_MOESM2_ESM.xls',

    }

    for dataset, url in datasets.items():
        # download_file(f'{url}') # some of these files require institutional access to download :(
        filename = url.split('/')[-1]
        os.rename(f'{output_folder}{filename}', f'{output_folder}{dataset}.{filename.split(".")[-1]}')

  
