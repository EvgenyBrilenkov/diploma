#предыдущий файл - 1_data_preparation.ipynb

import json
import pubchempy
from STOUT import translate_forward
from tqdm.auto import tqdm

with open('unique_mol_list.json', 'r') as file:
    data = json.load(file)

my_list = data[1::] #т.к. первое значение - пустая строка

my_dict = {}

waiting = len(my_list)
pbar = tqdm(total=waiting) #для отслеживания прогресса

while len(my_list) != 0:
    try:
        my_dict[my_list[0]] = pubchempy.get_compounds(my_list[0], 'smiles')[0].iupac_name #в случае наличия названия молекулы в базе данных PubChem название берется оттуда по ИЮПАК
    except:
        my_dict[my_list[0]] = translate_forward(my_list[0]) #если молекулы нет в PubChem, название присваивается обученной нейронной сетью по ИЮПАК
    my_list.pop(0) #удаляю каждую обработанную молекулу для экономия времени и памяти
    pbar.update(1)
    if len(my_dict) == 50000: #сохраняю каждые 50000 молекул во избежание вылета
        with open('ended_dict.json', 'r') as file:
            existing_data = json.load(file)
        existing_data.update(my_dict)
        with open('ended_dict.json', 'w') as file:
            json.dump(existing_data, file)
        my_dict = {}

pbar.close()

#финальное сохранение датасета
with open('ended_dict.json', 'r') as file:
    existing_data = json.load(file)
existing_data.update(my_dict)
with open('ended_dict.json', 'w') as file:
    json.dump(existing_data, file)

#продолжение в 3_data_from_SMILES_to_text.ipynb