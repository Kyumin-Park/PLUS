import argparse
import pandas
import requests
import tqdm
from xml.etree import ElementTree

def collect_smiles(n_sample=5000):
    molecule_df = pandas.read_csv('./data/molecule_dictionary_data.tsv', sep='\t', header=0, index_col='molregno')
    protein_df = pandas.read_csv('./data/target_sequence_data.tsv', sep='\t', header=0, index_col='tid')
    active_df = pandas.read_csv('./data/active_data.tsv', sep='\t', header=0)
    inactive_df = pandas.read_csv('./data/inactive_data.tsv', sep='\t', header=0)

    active_sample = active_df.sample(n_sample)
    inactive_sample = inactive_df.sample(n_sample)

    result = []
    for label, sample in enumerate([inactive_sample, active_sample]):
        for idx, row in tqdm.tqdm(sample.T.iteritems(), total=len(sample), desc=f'Build {["inactive", "active"][label]}'):
            tid = row['tid']
            molregno = row['molregno']
            protein = protein_df.loc[tid]['sequence']
            chem_id = molecule_df.loc[molregno]['chembl_id']

            xml_data = requests.get(f'https://www.ebi.ac.uk/chembl/api/data/molecule/{chem_id}')
            if xml_data.status_code != 200:
                continue
            xml_text = xml_data.text
            root = ElementTree.fromstring(xml_text)
            smiles = root.find('molecule_structures').findtext('canonical_smiles')
            if smiles is None:
                continue
            result.append(f'{label}\t{protein}\t{smiles}\n')

    with open('./data/preprocessed_data.tsv', 'w', encoding='utf-8') as f:
        f.writelines(result)
    print(len(result))

def collect_ecfp(n_sample=5000):
    protein_df = pandas.read_csv('./data/target_sequence_data.tsv', sep='\t', header=0, index_col='tid')
    active_df = pandas.read_csv('./data/active_data.tsv', sep='\t', header=0)
    inactive_df = pandas.read_csv('./data/inactive_data.tsv', sep='\t', header=0)
    ecfp_df = pandas.read_csv('./data/ecfp_data.tsv', sep='\t', header=0, index_col='molregno')

    active_sample = active_df.sample(n_sample)
    inactive_sample = inactive_df.sample(n_sample)

    result = []
    for label, sample in enumerate([inactive_sample, active_sample]):
        for idx, row in tqdm.tqdm(sample.T.iteritems(), total=len(sample), desc=f'Build {["inactive", "active"][label]}'):
            tid = row['tid']
            molregno = row['molregno']
            protein = protein_df.loc[tid]['sequence']

            ecfp = ecfp_df.loc[molregno]['ecfp']
            result.append(f'{label}\t{protein}\t{ecfp}\n')

    with open('./data/preprocessed_data_ecfp.tsv', 'w', encoding='utf-8') as f:
        f.writelines(result)
    print(len(result))

def get_smiles_vocab():
    data = pandas.read_csv('./data/preprocessed_data.tsv', sep='\t', header=None)
    molecules = data[2]
    vocab = ''
    for mole in molecules:
        for e in mole:
            if e not in vocab:
                vocab += e
    return bytes(vocab, encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='smiles', type=str, choices=['smiles', 'ecfp'],
                        help='Collection mode')
    parser.add_argument('--n_data', default=5000, type=int, help='Number of data per label')
    cfg = parser.parse_args()

    if cfg.mode == 'smiles':
        collect_smiles(cfg.n_data)
    else:
        collect_ecfp(cfg.n_data)
