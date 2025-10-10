# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Copyright (C) [2025] Instituto Nacional de Estadística
#
# Este archivo forma parte del proyecto statcodgen.
#
# Licenciado bajo la Licencia Pública de la Unión Europea (EUPL) v.1.2.
# Puede obtener una copia de la licencia en la raiz de este proyecto o en:
# https://eupl.eu/1.2/es/
#
# A menos que se indique lo contrario, este software se distribuye
# "TAL CUAL", SIN GARANTÍAS NI CONDICIONES DE NINGÚN TIPO.
# Consulte la licencia para conocer los términos específicos.
# ------------------------------------------------------------------------------
# Copyright (C) [2025] National Institute of Statistics
#
# This file is part of the statcodgen project.
#
# Licensed under the European Union Public License (EUPL) v.1.2.
# You can obtain a copy of the license at the root of this project or at:
# https://eupl.eu/1.2/es/
#
# Unless otherwise indicated, this software is distributed
# "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the license for specific terms.
# ------------------------------------------------------------------------------
"""
Created on Thu Jul 31 13:59:16 2025

@author: git.metodologia@ine.es
"""
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

from codauto.prompt_text import prompt_synt_data
from codauto.prompt_text import prompt_aug_data


def fill_prompt_synt_data(title, includes, prompt=prompt_synt_data):
    """
    Fill a synthetic prompt template with the provided title and included
    content.

    This function replaces the placeholders '<title>' and '<includes>' in the
    prompt template with the actual `title` and `includes` values supplied.

    Parameters:
    title (str): The title to insert into the prompt.
    includes (str): Additional content or context to insert into the prompt.
    prompt (str, optional): The synthetic prompt template to use. Defaults to
    global `prompt_synt_data`.

    Returns:
    str: The prompt text with the placeholders replaced by the provided values.
    """
    data_text = prompt.replace(
        '<title>', title
    ).replace(
        '<includes>', includes
    )
    return data_text


def fill_prompt_aug_data(
        title, includes, language='spanish', prompt=prompt_aug_data
):
    """
    Fill a prompt template with keywords extracted from a title and additional
    content.

    This function tokenizes the input `title`, removes stopwords and
    punctuation according to the specified `language`, and replaces the
    placeholders '<title>' and '<includes>' in the prompt template with the
    filtered keywords and the additional content.

    Parameters:
        title (str): The input title from which to extract keywords.
        includes (str): Additional content or context to insert into the prompt
        language (str, optional): Language for stopword removal (default is
        'spanish').
        prompt (str, optional): The prompt template to fill (default is a
        global variable `prompt_aug_data`).

    Returns:
        str: The filled-in prompt with keywords and additional content.
    """
    stop_words = set(stopwords.words(language))
    stop_words.update(punctuation)
    words = word_tokenize(title)
    words_list = set([word for word in words if word not in stop_words])
    data_text = prompt.replace(
        '<title>', str(words_list)
    ).replace(
        '<includes>', includes
    )
    return data_text
