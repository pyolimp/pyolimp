from __future__ import annotations
from typing import Annotated, Literal
from pydantic import Field
from pathlib import Path
from .base import StrictModel


class DatasetConfig(StrictModel):
    pass


class SCA2023(DatasetConfig):
    name: Literal["SCA2023"]

    subsets: set[
        Literal[
            "Images",
            "Images/Icons",
            "Images/Real_images/Animals",
            "Images/Real_images",
            "Images/Real_images/Faces",
            "Images/Real_images/Natural",
            "Images/Real_images/Urban",
            "Images/Texts",
            "PSFs",
            "PSFs/Broad",
            "PSFs/Medium",
            "PSFs/Narrow",
        ]
    ]

    def load(self):
        from ...dataset.sca_2023 import SCA2023Dataset

        return SCA2023Dataset(self.subsets)


class Olimp(DatasetConfig):
    name: Literal["Olimp"]

    subsets: set[
        Literal[
            "*",  # load all
            "abstracts and textures",
            "abstracts and textures/abstract art",
            "abstracts and textures/backgrounds and patterns",
            "abstracts and textures/colorful abstracts",
            "abstracts and textures/geometric shapes",
            "abstracts and textures/neon abstracts",
            "abstracts and textures/textures",
            "animals",
            "animals/birds",
            "animals/farm animals",
            "animals/insects and spiders",
            "animals/marine life",
            "animals/pets",
            "animals/wild animals",
            "art and culture",
            "art and culture/cartoon and comics",
            "art and culture/crafts and handicrafts",
            "art and culture/dance and theater performances",
            "art and culture/music concerts and instruments",
            "art and culture/painting and frescoes",
            "art and culture/sculpture and bas-reliefs",
            "food and drinks",
            "food and drinks/desserts and bakery",
            "food and drinks/dishes",
            "food and drinks/drinks",
            "food and drinks/food products on store shelves",
            "food and drinks/fruits and vegetables",
            "food and drinks/street food",
            "interiors",
            "interiors/gyms and pools",
            "interiors/living spaces",
            "interiors/museums and galleries",
            "interiors/offices",
            "interiors/restaurants and cafes",
            "interiors/shopping centers and stores",
            "nature",
            "nature/beaches",
            "nature/deserts",
            "nature/fields and meadows",
            "nature/forest",
            "nature/mountains",
            "nature/water bodies",
            "objects and items",
            "objects and items/books and stationery",
            "objects and items/clothing and accessories",
            "objects and items/electronics and gadgets",
            "objects and items/furniture and decor",
            "objects and items/tools and equipment",
            "objects and items/toys and games",
            "portraits and people",
            "portraits and people/athletes and dancers",
            "portraits and people/crowds and demonstrations",
            "portraits and people/group photos",
            "portraits and people/individual portraits",
            "portraits and people/models on runway",
            "portraits and people/workers in their workplaces",
            "sports and active leisure",
            "sports and active leisure/cycling and rollerblading",
            "sports and active leisure/extreme sports",
            "sports and active leisure/individual sports",
            "sports and active leisure/martial arts",
            "sports and active leisure/team sports",
            "sports and active leisure/tourism and hikes",
            "text and pictogram",
            "text and pictogram/billboard text",
            "text and pictogram/blueprints",
            "text and pictogram/caricatures and pencil drawing",
            "text and pictogram/text documents",
            "text and pictogram/traffic signs",
            "urban scenes",
            "urban scenes/architecture",
            "urban scenes/city at night",
            "urban scenes/graffiti and street art",
            "urban scenes/parks and squares",
            "urban scenes/streets and avenues",
            "urban scenes/transport",
        ]
    ]

    def load(self):
        from ...dataset.olimp import OlimpDataset

        return OlimpDataset(self.subsets)


class Directory(DatasetConfig):
    name: Literal["Directory"]
    path: Path
    matches: list[str] = ["*.jpg", "*.jpeg", "*.png"]

    def load(self):
        from ...dataset.directory import DirectoryDataset

        return DirectoryDataset(self.path, self.matches)


Dataset = Annotated[
    SCA2023 | Olimp | Directory, Field(..., discriminator="name")
]