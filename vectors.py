from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer


@dataclass()
class IMDBConfig:
    

def flatten(lsls: List[List], groups=None, stop=None):
    res = []

    for i, item in enumerate(lsls):
        if groups:
            item = " ".join(item) + " " + groups[i]
            res.append(item)
        else:
            res.append(" ".join(item))

        if stop and (i >= stop):
            break

    return res


class VectorSimilarities:
    """creates grouped df from combined and has methods for index retreival of titles"""

    def __init__(
        self,
        combined: pd.DataFrame,
        params=dict(ngram_range=(1, 3)),
    ):
        self.params = params
        self.combined_grouped = (
            combined.groupby(params["groupby_vars"])
            .agg(
                {
                    self.params["title"]: "last",
                    self.params["name_id"]: list,
                    self.params["role"]: list,
                    self.params["name"]: list,
                }
            )
            .reset_index()
        )

        self.vectorizer = CountVectorizer(**self.params)
        self.matrix = None

    def search(self, pattern, start, end):
        return self.combined_grouped[
            self.combined_grouped.originalTitle.str.contains(pattern, case=False)
            & (self.combined_grouped.startYear >= start)
            & (self.combined_grouped.startYear <= end)
        ]

    @property
    def matrix(self):
        return self.__matrix

    @matrix.setter
    def matrix(self, value):
        lsls = self.combined_grouped["nconst"].tolist()
        flat = flatten(lsls)

        self.__matrix = self.vectorizer.fit_transform(flat)

    def get_item_similarity(self, i, n, df=True):
        print(self.combined_grouped.iloc[i])
        sims = cosine_similarity(self.matrix[i], self.matrix)
        index = np.argsort(sims[i])
        item = {
            i: {
                "index": index[::-1][0:n],
                "scores": sims[i][index][::-1][0:n],
                "titles": self.combined_grouped[self.params["title"]].iloc[
                    index[::-1][0:n]
                ],
            }
        }

        if not df:
            return item

        result = self.combined_grouped.iloc[item[i]["titles"]]
        return result[result.genres != "Music"]
