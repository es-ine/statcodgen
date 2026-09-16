![](https://www.ine.es/menus/_b/img/logoINESocial.png)

# codauto v.1.0.0

[TOC]

# Work in Progress
This project is currently under active development. 

## Disclaimer

This repository and its contents are provided for informational and technical purposes only.  
Use of this code is at your own risk.

The Instituto Nacional de Estadística (Spanish Statistical Office) provides this software **"as is"**, without warranty of any kind, either expressed or implied.

By using this repository, you acknowledge that:
- You are responsible for reviewing and testing the code before using it in any production or critical environment.
- The Instituto Nacional de Estadística is **not liable** for any direct, indirect, or consequential damages arising from the use of this software.
 

## Description
The purpose of this library is to assist official statisticians in classifying variables according to standards, 
such as economic activities using [CNAE](https://ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736177032&menu=ultiDatos&idp=1254735976614), 
the standard classification for economic activities in Spanish official statistics (the Spanish version of [NACE](https://ec.europa.eu/eurostat/web/nace)).

This library is designed to train, evaluate, and use any model for any standard.

It has two modes for prediction: codification and assistance. Codification returns a code, while assistance returns a list of the most likely codes. It can also be used to classify a large list of data.

## Requirements
**C ++ compiler** for install fasttext if you are working on windows.

**fasttext v0.9.2** .

**python v3.9.18**.

You can use *entorno_codpython_20240613.yml* to easily install **codauto** in a standardized conda environment.

## Usage
Here you will find a brief explanation of how to install and use codauto using Windows.
### Installation
#### Generate Python env with Conda (optional)
If you work in windows and have installed Microsoft Visual Studio tools, follow this instructions to create an environment using *entorno_codpython_20240613.yml*  in Conda:

1. **Set the path to C++ compiler**
You must find the directory where cl compiler is located. Then, in the Conda Prompt, type:
````conda
set PATH=%PATH%;{compiler path}
````
where {compiler path} is the path to your cl directory.

2. **Change the environment name (optional)**
The default name is *entorno_codpython_20240613*. To change it, open *entorno_codpython_20240613.yml*  and modify the name in the NAME field.

3. **Create the environment**
In the Conda Prompt, navigate to the directory where *entorno_codpython_20240613.yml* is located using *cd* and type:
````conda
conda env create -f entorno_codpython_20240613.yml
````

#### Install codauto 
Type this for install codauto v.1.0.0 lib from github repository in python.
```conda
pip install git+https://github.com/es-ine/statcodgen.git@v1.0.0
```
### Generate a Structured child for your classification.
This library provides an abstract class for processing any standard structure, *Structured*. 
For any standard, you must code a *Structured* child that implements its own method *get_level(code)*. 
This method's input must be a code as a string, and its output must be an integer indicating the hierarchical level of the code, with 0 indicating the highest level.

An example for CNAE-2025.
```python
from codauto import Structured

class StructuredCNAE(Structured):
    """
    Structured hierarchy for CNAE (Clasificación Nacional de Actividades Económicas).

    Levels are determined by the length of the classification code
    (after removing dots).
    """

    def get_level(self, code):
        """
        Compute the hierarchical level for a CNAE code.

        Parameters
        ----------
        code : str
            CNAE classification code.

        Returns
        -------
        int
            Hierarchical level based on code length minus one.
        """
        code = code.replace('.', '')
        return len(code)-1

structurecnae25 = StructuredCNAE(
    structure_df=structure_df,
    names_l=['section', 'division', 'group', 'class']
)
```
Parameters for any *Structured*:
- *structure_df* (pandas.DataFrame): Must contain two columns:
    - Column 0 (str): Hierarchical sorted codes (possibly with dot notation).
    - Column 1 (str): Category titles corresponding to each code.
- *names_l* (None|tuple(str), Optional): Names of the different hierarchy levels. If it is None, it will be generated automatically.

### Generate a Codifier child for your model.
This library provides an abstract class that can be used to implement any model that provides confidence values for each possible code in the standard's lower levels, *Codifier*. 
For any strategy the child class must include at least 4 methods:
- *train(**kgrams)*: This is an abstract method designed to receive a dictionary of parameters, which allows it to be easily configured for training models with different NLP libraries available in Python (fastText, BERT, etc.). Use self.model.
- *load(name)*: This is an abstract method intended to implement a loading procedure for a previously trained model with the name {name} in self.model.
- *save(name)*: This is an abstract method intended to implement a saving procedure for a previously trained model with the name {name} from self.model.
- *get_preds_for_batch(sample)*:Abstract method designed to generate predictions on a batch of input samples. It receives as parameters a list of texts (samples) to be classified, and the method must return a list of tuples in which each element corresponds to each element of _samples_ and contains, first, a list of all possible lower-level codes sorted by confidence, second, those confidence levels. 

For example, if the classification standard includes two hierarchical levels—two sections (A, B), each with two groups (A1, A2) and (B1, B2), respectively—and two samples are processed, the method must return a structure like the following:
```python
[(["A1", "A2", "B1", "B2"], [0.55, 0.30, 0.1, 0.05]), (["B1", "A1", "B2", "A2"], [0.5, 0.4, 0.07, 0.03])]
```
The main method that conect the child with *Codifier* is *get_preds_for_batch(samples)*. 
The other methods can either be fully implemented in the child class or simply defined with pass if no logic is needed.

#### Codifier parameters
- *structure_instance*(Structured child): Structured child for the standard
- *train_df* (pandas.DataFrame): Must contain at least two columns:
    - Column 0 (str): Codes (possibly with dot notation).
    - Column 1 (str): Text description.
- *test_df* (pandas.DataFrame): Must contain three columns:
    - Column 0 (str): Ground truth last level code.
    - Column 1 (str): Text descriptions.
    - Column 2 (str): The source of the description, it can be empty but must be.
- *root_path* (str): Directory path where models will be save and load. If the directory does not exit it will be created.
- *corres_df* (pandas.DataFrame|None): Optional must contain three columns, defaults value None:
    - Column 0 (str): The previous classification code.
    - Column 1 (str): The new classification code.
    - Column 2 (int): The hierarchical level of the correspondence.
- *min_length_texts* (int): Minimum number of characters for text descriptions. Defaults to 3

#### Codifier main methods

##### Evaluate
The evaluate method evaluates the performance of the model using the test_df provided. It calculates various metrics and optionally includes a precision-recall curve.

Parameters:
- *source* (str, optional): Source of the test data to evaluate. If 'all' (default), uses the entire test set. Otherwise, filters the test set by the specified source value.
- *get_curve* (bool, optional): Whether to plot a precision-recall curve. Defaults value True
- *simplify* (bool, optional): If True, return basic metrics; if False,return detailed metrics and a class metric report. Defaults value True
- *version* (str, optional): Version of the CNAE model to evaluate. Defaults value CNAEmodel

Returns:
- *metrics* (dict): A dictionary of evaluation metrics per level. Includes accuracy and F1 scores (micro, macro, weighted).
- *report* (dict, optional):A dictionary of classification reports per label and level. Only returned if `simplify` is False.

##### Predict
The predict method can provide different hierarchical level codes and also return a list of the most likely codes.

Parameters:
- *desc_l* (tuple(str)): List of textual descriptions to classify.
- *mode* (str): Prediction mode. Must be either:
    - "codification" : return only top-1 prediction.
    - "assistance"   : return up to 15 predictions above threshold.
- *hierarchical_level* (int): The hierarchical classification level (1-based index).
- *threshold* (float): Minimum confidence required for a prediction to be considered valid.
- *original_code_l*  (tuple(str|None)|None, optional):List of previously known codes for each sample (for recoding assistance). If provided, corres_df must be defined. 1:1 correspondence will be directly recoded. Defaults None
- *identifier_l* (tuple(any), optional):List of identifiers corresponding to each input sample (e.g., for traceability). If its None will be made automatically. Defaults None

Returns:
The output stored in prediction is a sorted list of dictionaries (dict), where each dict from each dictionary corresponds to an description and is in the same list position as the description in the description_l list. The dictionaries are structured as follows:
```python
# prediction output for a CNAE classifier model
[{
'description': 'compra y venta de vehículos de motor',
'original_code': 'G',
'label': ('G',),
'confidence': (100,),
'hierarchical_level': 1,
'identifier': 'A01',
'title': ('Comercio al por mayor y al por menor; reparación de vehículos de motor y motocicletas', )
}]
```
- 'description': input description.
- 'original_code': original code used for recoding (if any).
- 'label': tuple of predicted class codes.
- 'confidence': tuple of confidence scores (integers, 0-100).
- 'hierarchical_level': classification level used.
- 'identifier': identifier from input or generated.- 'title': tuple of category titles corresponding to the labels.
- 'title': tuple of category titles corresponding to the labels.

#### Example
```python
import random
import pandas as pd

from codauto import Structured
from codauto import Codifier

# First, create the sample structure
example_structure_dict = {
    'code': ['A', 'A1', 'A2', 'B', 'B1', 'B2'],
    'title': ['Sección A', 'Grupo A1', 'Grupo A2', 'Sección B', 'Grupo B1', 'Grupo B2']
}
example_structure_df = pd.DataFrame(example_structure_dict)


# Second, create a child of Structured for this classification
class StructuredEXAMPLE(Structured):
    def get_level(self, code):
        return len(code) - 1

# Create an instance of StructuredEXAMPLE
structured_example = StructuredEXAMPLE(
    example_structure_df,
    ['section', 'group']
)

# Now we create a Codifier's son with a method that assigns random labels and confidence scores.
class CodifierRandom(Codifier):
    def save(self, name):
        pass

    def load(self, name):
        pass

    def train(self, name):
        pass

    def get_pred_for_batch(self, samples):
        """
        Parameters
        ----------
        samples : list of str
            Text samples to predict.

        Returns
        -------
        list, tuple
            list of tuples with sorted labels and confidences.
        """
        # Get last level labels
        last_level_labels = list(self.structure.reversed_hierarchy.keys())
        preds = []
        for sample in samples:
            random_vals = [random.random() for _ in last_level_labels]
            total = sum(random_vals)
            probs = [v / total for v in random_vals]
            label_probs = zip(last_level_labels, probs)
            sorted_pred = sorted(label_probs, key=lambda x: x[1], reverse=True)
            label_l = [label for label, conf in sorted_pred]
            conf_l = [conf for label, conf in sorted_pred]
            preds.append((label_l, conf_l))
        return preds

# To create an instance of CodifierRandom, a train_df, a test_df, an instance of StructuredEXAMPLE,
# and a path that will serve as the root directory for the desired classification are needed
train_dict = {
    'label': ['A1', 'A1', 'A2', 'A2', 'B1', 'B2', 'bad_code']*100,
    'probe': ['pueba']*700
}
test_df = {
    'label': ['A1', 'A1', 'B1', 'B2', 'bad_code']*100,
    'probe': ['pueba']*500,
    'source': ['sour1', 'sour2', 'sour1', 'sour2', 'sour3']*100
}
train_df = pd.DataFrame(train_dict)
test_df = pd.DataFrame(test_df)

codifier_example = CodifierRandom(
    structure_instance=structured_example,  # It must be an instance of the subclass of Structured for the classification
    train_df=train_df,  # It must be a pandas DataFrame containing strings, with two columns: 0 for label, 1 for description
    test_df=test_df,  # It must be a pandas DataFrame containing strings, with three columns: 0 for label, 1 for description, 2 for source
    root_path=r'',  # The base working directory will be created automatically if it does not exist.
    corres_df=None,   # It must be a pandas DataFrame containing strings, with three columns: 0 for source code,
    # 1 for code of interest in the classification, 2 for the hierarchy of the code of interest in the classification
    min_lenght_texts=3  # int indicating the minimum length that the descriptions in train_df and test_df must have
)

codifier_example.evaluate(version='example')
# Predictions can be made on a specific level
desc_l = ['prueba']*5
mode = 'codification'
hierarchical_level = 2
threshold = 0

codifier_example.predict(
    desc_l,
    mode,
    hierarchical_level,
    threshold
)
# Assistance can also be requested
desc_l = ['prueba']*5
mode = 'assistance'
hierarchical_level = 1
threshold = 0

codifier_example.predict(
    desc_l,
    mode,
    hierarchical_level,
    threshold
)
```

## Prompt generation
This library also has two functions for generating prompts to produce synthetic data with LLMs via two different approaches using explanatory notes.

#### Fill_prompt_synt_data
It fills a synthetic prompt template by replacing predefined placeholders with a given title and notes. The purpose of this function is to generate prompts for requesting that LLMs directly generate samples of coded descriptions.
```python
from codauto.prompt_maker import fill_prompt_synt_data

fill_prompt_synt_data(title, includes, prompt)
```
Parameters:
- *title* (str): The title to insert into the prompt.
- *includes* (str): The includes explanatory notes to insert into the prompt.
- *prompt* (str): Prompt template

#### Fill_prompt_aug_data
It fills a synthetic prompt template by replacing predefined placeholders with a given title and notes. 
The purpose of this function is to generate prompts for requesting that LLMs create dictionaries of synonyms from the keywords in the title, and then generate samples of coded descriptions by replacing the words in the title.
```python
from codauto.prompt_maker import fill_aug_synt_data

fill_prompt_aug_data(title, includes, language, prompt)
```
Parameters:
- *title* (str): The input title to extract keywords from.
- *includes* (str): The includes explanatory notes to insert into the prompt.
- *language* (str, optional): The language to use for stopword filtering. Defaults value 'spanish'.
- *prompt* (str): Prompt template

## References
Install [Visual studio C++ compiler.](https://github.com/bycloudai/InstallVSBuildToolsWindows)

[Fasttext library.](https://fasttext.cc/docs/en/support.html)

[Create conda env.](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

What are [.yml files.](https://en.wikipedia.org/wiki/YAML)

## Support
Write an email to any author asking for help.

You can also contact via nomenclaturas@ine.es.

## Contributing
This is a private project, but any ideas are welcome. You can email nomenclaturas@ine.es to contribute.

## Authors
Statistics Spain (git.metodologia@ine.es) 

## Acknowledgment
Sebastián Gallego Herrera

Andrés Jurado Prieto

Adrián Pérez Bote

Carlos Sáez Calvo

Jorge Fernández Calatrava