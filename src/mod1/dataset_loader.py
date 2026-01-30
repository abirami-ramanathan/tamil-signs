"""
Module 1: TLFS23 Dataset Loading and Label Mapping

This module is responsible for loading the TLFS23 dataset (Tamil Language Fingerspelling dataset)
which contains 254,147 images across 247 Tamil alphabet classes. It handles dataset initialization,
class mapping, and organizes the data structure for subsequent processing stages.

Author: Tamil Sign Language Recognition Team
Date: January 2026
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
import cv2
from tqdm import tqdm


class TamilCharacterMapping:
    """
    Manages the mapping between numerical labels (0-246) and Tamil alphabet characters.
    """
    
    def __init__(self):
        """Initialize the Tamil character mappings based on TLFS23 dataset."""
        # Character mappings from ReadMe.txt
        self.folder_to_character = {
            1: {'tamil': 'அ', 'pronunciation': 'a', 'type': 'vowel'},
            2: {'tamil': 'ஆ', 'pronunciation': 'ā', 'type': 'vowel'},
            3: {'tamil': 'இ', 'pronunciation': 'i', 'type': 'vowel'},
            4: {'tamil': 'ஈ', 'pronunciation': 'ī', 'type': 'vowel'},
            5: {'tamil': 'உ', 'pronunciation': 'u', 'type': 'vowel'},
            6: {'tamil': 'ஊ', 'pronunciation': 'ū', 'type': 'vowel'},
            7: {'tamil': 'எ', 'pronunciation': 'e', 'type': 'vowel'},
            8: {'tamil': 'ஏ', 'pronunciation': 'ē', 'type': 'vowel'},
            9: {'tamil': 'ஐ', 'pronunciation': 'ai', 'type': 'vowel'},
            10: {'tamil': 'ஒ', 'pronunciation': 'o', 'type': 'vowel'},
            11: {'tamil': 'ஓ', 'pronunciation': 'ō', 'type': 'vowel'},
            12: {'tamil': 'ஔ', 'pronunciation': 'au', 'type': 'vowel'},
            13: {'tamil': 'ஃ', 'pronunciation': 'ak', 'type': 'vowel'},
            14: {'tamil': 'க்', 'pronunciation': 'k', 'type': 'consonant'},
            15: {'tamil': 'ங்', 'pronunciation': 'ṅ', 'type': 'consonant'},
            16: {'tamil': 'ச்', 'pronunciation': 'c', 'type': 'consonant'},
            17: {'tamil': 'ஞ்', 'pronunciation': 'ñ', 'type': 'consonant'},
            18: {'tamil': 'ட்', 'pronunciation': 'ṭ', 'type': 'consonant'},
            19: {'tamil': 'ண்', 'pronunciation': 'ṇ', 'type': 'consonant'},
            20: {'tamil': 'த்', 'pronunciation': 't', 'type': 'consonant'},
            21: {'tamil': 'ந்', 'pronunciation': 'n', 'type': 'consonant'},
            22: {'tamil': 'ப்', 'pronunciation': 'p', 'type': 'consonant'},
            23: {'tamil': 'ம்', 'pronunciation': 'm', 'type': 'consonant'},
            24: {'tamil': 'ய்', 'pronunciation': 'y', 'type': 'consonant'},
            25: {'tamil': 'ர்', 'pronunciation': 'r', 'type': 'consonant'},
            26: {'tamil': 'ல்', 'pronunciation': 'l', 'type': 'consonant'},
            27: {'tamil': 'வ்', 'pronunciation': 'v', 'type': 'consonant'},
            28: {'tamil': 'ழ்', 'pronunciation': 'lzh', 'type': 'consonant'},
            29: {'tamil': 'ள்', 'pronunciation': 'll', 'type': 'consonant'},
            30: {'tamil': 'ற்', 'pronunciation': 'ṟ', 'type': 'consonant'},
            31: {'tamil': 'ன்', 'pronunciation': 'ṉ', 'type': 'consonant'},
            32: {'tamil': 'க', 'pronunciation': 'Ka', 'type': 'compound'},
            33: {'tamil': 'கா', 'pronunciation': 'Kā', 'type': 'compound'},
            34: {'tamil': 'கி', 'pronunciation': 'Ki', 'type': 'compound'},
            35: {'tamil': 'கீ', 'pronunciation': 'Kī', 'type': 'compound'},
            36: {'tamil': 'கு', 'pronunciation': 'Ku', 'type': 'compound'},
            37: {'tamil': 'கூ', 'pronunciation': 'Kū', 'type': 'compound'},
            38: {'tamil': 'கெ', 'pronunciation': 'Ke', 'type': 'compound'},
            39: {'tamil': 'கே', 'pronunciation': 'Kē', 'type': 'compound'},
            40: {'tamil': 'கை', 'pronunciation': 'Kai', 'type': 'compound'},
            41: {'tamil': 'கொ', 'pronunciation': 'Ko', 'type': 'compound'},
            42: {'tamil': 'கோ', 'pronunciation': 'Kō', 'type': 'compound'},
            43: {'tamil': 'கௌ', 'pronunciation': 'Kau', 'type': 'compound'},
            44: {'tamil': 'ங', 'pronunciation': 'Nga', 'type': 'compound'},
            45: {'tamil': 'ஙா', 'pronunciation': 'Ngā', 'type': 'compound'},
            46: {'tamil': 'ஙி', 'pronunciation': 'Ngi', 'type': 'compound'},
            47: {'tamil': 'ஙீ', 'pronunciation': 'Ngī', 'type': 'compound'},
            48: {'tamil': 'ஙு', 'pronunciation': 'Ngu', 'type': 'compound'},
            49: {'tamil': 'ஙூ', 'pronunciation': 'Ngū', 'type': 'compound'},
            50: {'tamil': 'ஙெ', 'pronunciation': 'Nge', 'type': 'compound'},
            51: {'tamil': 'ஙே', 'pronunciation': 'Ngē', 'type': 'compound'},
            52: {'tamil': 'ஙை', 'pronunciation': 'Ngai', 'type': 'compound'},
            53: {'tamil': 'ஙொ', 'pronunciation': 'Ngo', 'type': 'compound'},
            54: {'tamil': 'ஙோ', 'pronunciation': 'Ngō', 'type': 'compound'},
            55: {'tamil': 'ஙௌ', 'pronunciation': 'Ngau', 'type': 'compound'},
            56: {'tamil': 'ச', 'pronunciation': 'Sa', 'type': 'compound'},
            57: {'tamil': 'சா', 'pronunciation': 'Sā', 'type': 'compound'},
            58: {'tamil': 'சி', 'pronunciation': 'Si', 'type': 'compound'},
            59: {'tamil': 'சீ', 'pronunciation': 'Sī', 'type': 'compound'},
            60: {'tamil': 'சு', 'pronunciation': 'Su', 'type': 'compound'},
            61: {'tamil': 'சூ', 'pronunciation': 'Sū', 'type': 'compound'},
            62: {'tamil': 'செ', 'pronunciation': 'Se', 'type': 'compound'},
            63: {'tamil': 'சே', 'pronunciation': 'Sē', 'type': 'compound'},
            64: {'tamil': 'சை', 'pronunciation': 'Sai', 'type': 'compound'},
            65: {'tamil': 'சொ', 'pronunciation': 'So', 'type': 'compound'},
            66: {'tamil': 'சோ', 'pronunciation': 'Sō', 'type': 'compound'},
            67: {'tamil': 'சௌ', 'pronunciation': 'Sau', 'type': 'compound'},
            68: {'tamil': 'ஞ', 'pronunciation': 'Ña', 'type': 'compound'},
            69: {'tamil': 'ஞா', 'pronunciation': 'Ñā', 'type': 'compound'},
            70: {'tamil': 'ஞி', 'pronunciation': 'Ñi', 'type': 'compound'},
            71: {'tamil': 'ஞீ', 'pronunciation': 'Ñī', 'type': 'compound'},
            72: {'tamil': 'ஞு', 'pronunciation': 'Ñu', 'type': 'compound'},
            73: {'tamil': 'ஞூ', 'pronunciation': 'Ñū', 'type': 'compound'},
            74: {'tamil': 'ஞெ', 'pronunciation': 'Ñe', 'type': 'compound'},
            75: {'tamil': 'ஞே', 'pronunciation': 'Ñē', 'type': 'compound'},
            76: {'tamil': 'ஞை', 'pronunciation': 'Ñai', 'type': 'compound'},
            77: {'tamil': 'ஞொ', 'pronunciation': 'Ño', 'type': 'compound'},
            78: {'tamil': 'ஞோ', 'pronunciation': 'Ñō', 'type': 'compound'},
            79: {'tamil': 'ஞௌ', 'pronunciation': 'Ñau', 'type': 'compound'},
            80: {'tamil': 'ட', 'pronunciation': 'Ṭa', 'type': 'compound'},
            81: {'tamil': 'டா', 'pronunciation': 'Ṭā', 'type': 'compound'},
            82: {'tamil': 'டி', 'pronunciation': 'Ṭi', 'type': 'compound'},
            83: {'tamil': 'டீ', 'pronunciation': 'Ṭī', 'type': 'compound'},
            84: {'tamil': 'டு', 'pronunciation': 'Ṭu', 'type': 'compound'},
            85: {'tamil': 'டூ', 'pronunciation': 'Ṭū', 'type': 'compound'},
            86: {'tamil': 'டெ', 'pronunciation': 'Ṭe', 'type': 'compound'},
            87: {'tamil': 'டே', 'pronunciation': 'Ṭē', 'type': 'compound'},
            88: {'tamil': 'டை', 'pronunciation': 'Ṭai', 'type': 'compound'},
            89: {'tamil': 'டொ', 'pronunciation': 'Ṭo', 'type': 'compound'},
            90: {'tamil': 'டோ', 'pronunciation': 'Ṭō', 'type': 'compound'},
            91: {'tamil': 'டௌ', 'pronunciation': 'Ṭau', 'type': 'compound'},
            92: {'tamil': 'ண', 'pronunciation': 'Ṇa', 'type': 'compound'},
            93: {'tamil': 'ணா', 'pronunciation': 'Ṇā', 'type': 'compound'},
            94: {'tamil': 'ணி', 'pronunciation': 'Ṇi', 'type': 'compound'},
            95: {'tamil': 'ணீ', 'pronunciation': 'Ṇī', 'type': 'compound'},
            96: {'tamil': 'ணு', 'pronunciation': 'Ṇu', 'type': 'compound'},
            97: {'tamil': 'ணூ', 'pronunciation': 'Ṇū', 'type': 'compound'},
            98: {'tamil': 'ணெ', 'pronunciation': 'Ṇe', 'type': 'compound'},
            99: {'tamil': 'ணே', 'pronunciation': 'Ṇē', 'type': 'compound'},
            100: {'tamil': 'ணை', 'pronunciation': 'Ṇai', 'type': 'compound'},
            101: {'tamil': 'ணொ', 'pronunciation': 'Ṇo', 'type': 'compound'},
            102: {'tamil': 'ணோ', 'pronunciation': 'Ṇō', 'type': 'compound'},
            103: {'tamil': 'ணௌ', 'pronunciation': 'Ṇau', 'type': 'compound'},
            104: {'tamil': 'த', 'pronunciation': 'Ta', 'type': 'compound'},
            105: {'tamil': 'தா', 'pronunciation': 'Tā', 'type': 'compound'},
            106: {'tamil': 'தி', 'pronunciation': 'Ti', 'type': 'compound'},
            107: {'tamil': 'தீ', 'pronunciation': 'Tī', 'type': 'compound'},
            108: {'tamil': 'து', 'pronunciation': 'Tu', 'type': 'compound'},
            109: {'tamil': 'தூ', 'pronunciation': 'Tū', 'type': 'compound'},
            110: {'tamil': 'தெ', 'pronunciation': 'Te', 'type': 'compound'},
            111: {'tamil': 'தே', 'pronunciation': 'Tē', 'type': 'compound'},
            112: {'tamil': 'தை', 'pronunciation': 'Tai', 'type': 'compound'},
            113: {'tamil': 'தொ', 'pronunciation': 'To', 'type': 'compound'},
            114: {'tamil': 'தோ', 'pronunciation': 'Tō', 'type': 'compound'},
            115: {'tamil': 'தௌ', 'pronunciation': 'Tau', 'type': 'compound'},
            116: {'tamil': 'ந', 'pronunciation': 'Na', 'type': 'compound'},
            117: {'tamil': 'நா', 'pronunciation': 'Nā', 'type': 'compound'},
            118: {'tamil': 'நி', 'pronunciation': 'Ni', 'type': 'compound'},
            119: {'tamil': 'நீ', 'pronunciation': 'Nī', 'type': 'compound'},
            120: {'tamil': 'நு', 'pronunciation': 'Nu', 'type': 'compound'},
            121: {'tamil': 'நூ', 'pronunciation': 'Nū', 'type': 'compound'},
            122: {'tamil': 'நெ', 'pronunciation': 'Ne', 'type': 'compound'},
            123: {'tamil': 'நே', 'pronunciation': 'Nē', 'type': 'compound'},
            124: {'tamil': 'நை', 'pronunciation': 'Nai', 'type': 'compound'},
            125: {'tamil': 'நொ', 'pronunciation': 'No', 'type': 'compound'},
            126: {'tamil': 'நோ', 'pronunciation': 'Nō', 'type': 'compound'},
            127: {'tamil': 'நௌ', 'pronunciation': 'Nau', 'type': 'compound'},
            128: {'tamil': 'ப', 'pronunciation': 'Pa', 'type': 'compound'},
            129: {'tamil': 'பா', 'pronunciation': 'Pā', 'type': 'compound'},
            130: {'tamil': 'பி', 'pronunciation': 'Pi', 'type': 'compound'},
            131: {'tamil': 'பீ', 'pronunciation': 'Pī', 'type': 'compound'},
            132: {'tamil': 'பு', 'pronunciation': 'Pu', 'type': 'compound'},
            133: {'tamil': 'பூ', 'pronunciation': 'Pū', 'type': 'compound'},
            134: {'tamil': 'பெ', 'pronunciation': 'Pe', 'type': 'compound'},
            135: {'tamil': 'பே', 'pronunciation': 'Pē', 'type': 'compound'},
            136: {'tamil': 'பை', 'pronunciation': 'Pai', 'type': 'compound'},
            137: {'tamil': 'பொ', 'pronunciation': 'Po', 'type': 'compound'},
            138: {'tamil': 'போ', 'pronunciation': 'Pō', 'type': 'compound'},
            139: {'tamil': 'பௌ', 'pronunciation': 'Pau', 'type': 'compound'},
            140: {'tamil': 'ம', 'pronunciation': 'Ma', 'type': 'compound'},
            141: {'tamil': 'மா', 'pronunciation': 'Mā', 'type': 'compound'},
            142: {'tamil': 'மி', 'pronunciation': 'Mi', 'type': 'compound'},
            143: {'tamil': 'மீ', 'pronunciation': 'Mī', 'type': 'compound'},
            144: {'tamil': 'மு', 'pronunciation': 'Mu', 'type': 'compound'},
            145: {'tamil': 'மூ', 'pronunciation': 'Mū', 'type': 'compound'},
            146: {'tamil': 'மெ', 'pronunciation': 'Me', 'type': 'compound'},
            147: {'tamil': 'மே', 'pronunciation': 'Mē', 'type': 'compound'},
            148: {'tamil': 'மை', 'pronunciation': 'Mai', 'type': 'compound'},
            149: {'tamil': 'மொ', 'pronunciation': 'Mo', 'type': 'compound'},
            150: {'tamil': 'மோ', 'pronunciation': 'Mō', 'type': 'compound'},
            151: {'tamil': 'மௌ', 'pronunciation': 'Mau', 'type': 'compound'},
            152: {'tamil': 'ய', 'pronunciation': 'Ya', 'type': 'compound'},
            153: {'tamil': 'யா', 'pronunciation': 'Yā', 'type': 'compound'},
            154: {'tamil': 'யி', 'pronunciation': 'Yi', 'type': 'compound'},
            155: {'tamil': 'யீ', 'pronunciation': 'Yī', 'type': 'compound'},
            156: {'tamil': 'யு', 'pronunciation': 'Yu', 'type': 'compound'},
            157: {'tamil': 'யூ', 'pronunciation': 'Yū', 'type': 'compound'},
            158: {'tamil': 'யெ', 'pronunciation': 'Ye', 'type': 'compound'},
            159: {'tamil': 'யே', 'pronunciation': 'Yē', 'type': 'compound'},
            160: {'tamil': 'யை', 'pronunciation': 'Yai', 'type': 'compound'},
            161: {'tamil': 'யொ', 'pronunciation': 'Yo', 'type': 'compound'},
            162: {'tamil': 'யோ', 'pronunciation': 'Yō', 'type': 'compound'},
            163: {'tamil': 'யௌ', 'pronunciation': 'Yau', 'type': 'compound'},
            164: {'tamil': 'ர', 'pronunciation': 'Ra', 'type': 'compound'},
            165: {'tamil': 'ரா', 'pronunciation': 'Rā', 'type': 'compound'},
            166: {'tamil': 'ரி', 'pronunciation': 'Ri', 'type': 'compound'},
            167: {'tamil': 'ரீ', 'pronunciation': 'Rī', 'type': 'compound'},
            168: {'tamil': 'ரு', 'pronunciation': 'Ru', 'type': 'compound'},
            169: {'tamil': 'ரூ', 'pronunciation': 'Rū', 'type': 'compound'},
            170: {'tamil': 'ரெ', 'pronunciation': 'Re', 'type': 'compound'},
            171: {'tamil': 'ரே', 'pronunciation': 'Rē', 'type': 'compound'},
            172: {'tamil': 'ரை', 'pronunciation': 'Rai', 'type': 'compound'},
            173: {'tamil': 'ரொ', 'pronunciation': 'Ro', 'type': 'compound'},
            174: {'tamil': 'ரோ', 'pronunciation': 'Rō', 'type': 'compound'},
            175: {'tamil': 'ரௌ', 'pronunciation': 'Rau', 'type': 'compound'},
            176: {'tamil': 'ல', 'pronunciation': 'La', 'type': 'compound'},
            177: {'tamil': 'லா', 'pronunciation': 'Lā', 'type': 'compound'},
            178: {'tamil': 'லி', 'pronunciation': 'Li', 'type': 'compound'},
            179: {'tamil': 'லீ', 'pronunciation': 'Lī', 'type': 'compound'},
            180: {'tamil': 'லு', 'pronunciation': 'Lu', 'type': 'compound'},
            181: {'tamil': 'லூ', 'pronunciation': 'Lū', 'type': 'compound'},
            182: {'tamil': 'லெ', 'pronunciation': 'Le', 'type': 'compound'},
            183: {'tamil': 'லே', 'pronunciation': 'Lē', 'type': 'compound'},
            184: {'tamil': 'லை', 'pronunciation': 'Lai', 'type': 'compound'},
            185: {'tamil': 'லொ', 'pronunciation': 'Lo', 'type': 'compound'},
            186: {'tamil': 'லோ', 'pronunciation': 'Lō', 'type': 'compound'},
            187: {'tamil': 'லௌ', 'pronunciation': 'Lau', 'type': 'compound'},
            188: {'tamil': 'வ', 'pronunciation': 'Va', 'type': 'compound'},
            189: {'tamil': 'வா', 'pronunciation': 'Vā', 'type': 'compound'},
            190: {'tamil': 'வி', 'pronunciation': 'Vi', 'type': 'compound'},
            191: {'tamil': 'வீ', 'pronunciation': 'Vī', 'type': 'compound'},
            192: {'tamil': 'வு', 'pronunciation': 'Vu', 'type': 'compound'},
            193: {'tamil': 'வூ', 'pronunciation': 'Vū', 'type': 'compound'},
            194: {'tamil': 'வெ', 'pronunciation': 'Ve', 'type': 'compound'},
            195: {'tamil': 'வே', 'pronunciation': 'Vē', 'type': 'compound'},
            196: {'tamil': 'வை', 'pronunciation': 'Vai', 'type': 'compound'},
            197: {'tamil': 'வொ', 'pronunciation': 'Vo', 'type': 'compound'},
            198: {'tamil': 'வோ', 'pronunciation': 'Vō', 'type': 'compound'},
            199: {'tamil': 'வௌ', 'pronunciation': 'Vau', 'type': 'compound'},
            200: {'tamil': 'ழ', 'pronunciation': 'Lzha', 'type': 'compound'},
            201: {'tamil': 'ழா', 'pronunciation': 'Lzhā', 'type': 'compound'},
            202: {'tamil': 'ழி', 'pronunciation': 'Lzhi', 'type': 'compound'},
            203: {'tamil': 'ழீ', 'pronunciation': 'Lzhī', 'type': 'compound'},
            204: {'tamil': 'ழு', 'pronunciation': 'Lzhu', 'type': 'compound'},
            205: {'tamil': 'ழூ', 'pronunciation': 'Lzhū', 'type': 'compound'},
            206: {'tamil': 'ழெ', 'pronunciation': 'Lzhe', 'type': 'compound'},
            207: {'tamil': 'ழே', 'pronunciation': 'Lzhē', 'type': 'compound'},
            208: {'tamil': 'ழை', 'pronunciation': 'Lzhai', 'type': 'compound'},
            209: {'tamil': 'ழொ', 'pronunciation': 'Lzho', 'type': 'compound'},
            210: {'tamil': 'ழோ', 'pronunciation': 'Lzhō', 'type': 'compound'},
            211: {'tamil': 'ழௌ', 'pronunciation': 'Lzhau', 'type': 'compound'},
            212: {'tamil': 'ள', 'pronunciation': 'Lla', 'type': 'compound'},
            213: {'tamil': 'ளா', 'pronunciation': 'Llā', 'type': 'compound'},
            214: {'tamil': 'ளி', 'pronunciation': 'Lli', 'type': 'compound'},
            215: {'tamil': 'ளீ', 'pronunciation': 'Llī', 'type': 'compound'},
            216: {'tamil': 'ளு', 'pronunciation': 'Llu', 'type': 'compound'},
            217: {'tamil': 'ளூ', 'pronunciation': 'Llū', 'type': 'compound'},
            218: {'tamil': 'ளெ', 'pronunciation': 'Lle', 'type': 'compound'},
            219: {'tamil': 'ளே', 'pronunciation': 'Llē', 'type': 'compound'},
            220: {'tamil': 'ளை', 'pronunciation': 'Llai', 'type': 'compound'},
            221: {'tamil': 'ளொ', 'pronunciation': 'Llo', 'type': 'compound'},
            222: {'tamil': 'ளோ', 'pronunciation': 'Llō', 'type': 'compound'},
            223: {'tamil': 'ளௌ', 'pronunciation': 'Llau', 'type': 'compound'},
            224: {'tamil': 'ற', 'pronunciation': 'Ṟa', 'type': 'compound'},
            225: {'tamil': 'றா', 'pronunciation': 'Ṟā', 'type': 'compound'},
            226: {'tamil': 'றி', 'pronunciation': 'Ṟi', 'type': 'compound'},
            227: {'tamil': 'றீ', 'pronunciation': 'Ṟī', 'type': 'compound'},
            228: {'tamil': 'று', 'pronunciation': 'Ṟu', 'type': 'compound'},
            229: {'tamil': 'றூ', 'pronunciation': 'Ṟū', 'type': 'compound'},
            230: {'tamil': 'றெ', 'pronunciation': 'Ṟe', 'type': 'compound'},
            231: {'tamil': 'றே', 'pronunciation': 'Ṟē', 'type': 'compound'},
            232: {'tamil': 'றை', 'pronunciation': 'Ṟai', 'type': 'compound'},
            233: {'tamil': 'றொ', 'pronunciation': 'Ṟo', 'type': 'compound'},
            234: {'tamil': 'றோ', 'pronunciation': 'Ṟō', 'type': 'compound'},
            235: {'tamil': 'றௌ', 'pronunciation': 'Ṟau', 'type': 'compound'},
            236: {'tamil': 'ன', 'pronunciation': 'Ṉa', 'type': 'compound'},
            237: {'tamil': 'னா', 'pronunciation': 'Ṉā', 'type': 'compound'},
            238: {'tamil': 'னி', 'pronunciation': 'Ṉi', 'type': 'compound'},
            239: {'tamil': 'னீ', 'pronunciation': 'Ṉī', 'type': 'compound'},
            240: {'tamil': 'னு', 'pronunciation': 'Ṉu', 'type': 'compound'},
            241: {'tamil': 'னூ', 'pronunciation': 'Ṉū', 'type': 'compound'},
            242: {'tamil': 'னெ', 'pronunciation': 'Ṉe', 'type': 'compound'},
            243: {'tamil': 'னே', 'pronunciation': 'Ṉē', 'type': 'compound'},
            244: {'tamil': 'னை', 'pronunciation': 'Ṉai', 'type': 'compound'},
            245: {'tamil': 'னொ', 'pronunciation': 'Ṉo', 'type': 'compound'},
            246: {'tamil': 'னோ', 'pronunciation': 'Ṉō', 'type': 'compound'},
            247: {'tamil': 'னௌ', 'pronunciation': 'Ṉau', 'type': 'compound'},
        }
        
        # Create reverse mappings
        self.character_to_folder = {
            v['tamil']: k for k, v in self.folder_to_character.items()
        }
        
        # Create label mappings (0-246 for ML models)
        self.label_to_character = {
            i: self.folder_to_character[i+1] for i in range(247)
        }
        
        self.character_to_label = {
            v['tamil']: i for i, v in self.label_to_character.items()
        }
    
    def get_character_by_folder(self, folder_num: int) -> Dict:
        """Get character info by folder number (1-247)."""
        return self.folder_to_character.get(folder_num, None)
    
    def get_character_by_label(self, label: int) -> Dict:
        """Get character info by ML label (0-246)."""
        return self.label_to_character.get(label, None)
    
    def get_label_by_character(self, character: str) -> int:
        """Get ML label by Tamil character."""
        return self.character_to_label.get(character, -1)
    
    def get_folder_by_character(self, character: str) -> int:
        """Get folder number by Tamil character."""
        return self.character_to_folder.get(character, -1)
    
    def get_all_characters(self) -> List[str]:
        """Get list of all Tamil characters."""
        return [self.label_to_character[i]['tamil'] for i in range(247)]
    
    def get_character_type_counts(self) -> Dict[str, int]:
        """Get count of each character type."""
        counts = defaultdict(int)
        for char_info in self.folder_to_character.values():
            counts[char_info['type']] += 1
        return dict(counts)


class TLFS23DatasetLoader:
    """
    Main dataset loader class for TLFS23 Tamil Sign Language dataset.
    Handles loading, validation, and organization of the dataset.
    """
    
    def __init__(self, dataset_path: str, include_background: bool = False):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path: Path to the TLFS23 dataset root directory
            include_background: Whether to include the background class (default: False)
        """
        self.dataset_path = Path(dataset_path)
        self.include_background = include_background
        self.mapping = TamilCharacterMapping()
        
        # Dataset directories
        self.dataset_folders_path = self.dataset_path / "Dataset Folders"
        self.reference_images_path = self.dataset_path / "Refrence Image"  # Note: typo in original
        
        # Dataset structure
        self.class_paths = {}
        self.dataset_info = {}
        self.dataset_stats = {}
        
        # Validate paths
        self._validate_paths()
    
    def _validate_paths(self):
        """Validate that the dataset paths exist."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        if not self.dataset_folders_path.exists():
            raise FileNotFoundError(f"Dataset Folders path does not exist: {self.dataset_folders_path}")
        
        if not self.reference_images_path.exists():
            print(f"Warning: Reference Images path does not exist: {self.reference_images_path}")
    
    def load_dataset_structure(self, validate_images: bool = False) -> Dict:
        """
        Load and organize the dataset structure.
        
        Args:
            validate_images: Whether to validate that all images can be read (slower)
        
        Returns:
            Dictionary containing dataset structure and metadata
        """
        print("Loading TLFS23 Dataset Structure...")
        
        # Initialize storage
        self.class_paths = {}
        class_image_counts = {}
        
        # Scan folders 1-247
        for folder_num in tqdm(range(1, 248), desc="Scanning class folders"):
            folder_path = self.dataset_folders_path / str(folder_num)
            
            if not folder_path.exists():
                print(f"Warning: Folder {folder_num} does not exist")
                continue
            
            # Get character info
            char_info = self.mapping.get_character_by_folder(folder_num)
            if char_info is None:
                print(f"Warning: No character mapping for folder {folder_num}")
                continue
            
            # Get all image files in the folder
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(folder_path.glob(f"*{ext}")))
                image_files.extend(list(folder_path.glob(f"*{ext.upper()}")))
            
            # Store class information
            label = folder_num - 1  # ML label (0-246)
            self.class_paths[label] = {
                'folder_num': folder_num,
                'folder_path': str(folder_path),
                'tamil_char': char_info['tamil'],
                'pronunciation': char_info['pronunciation'],
                'type': char_info['type'],
                'image_paths': [str(img) for img in image_files],
                'image_count': len(image_files)
            }
            class_image_counts[label] = len(image_files)
            
            # Validate images if requested
            if validate_images and len(image_files) > 0:
                # Test read first image
                try:
                    img = cv2.imread(str(image_files[0]))
                    if img is None:
                        print(f"Warning: Cannot read image in folder {folder_num}")
                except Exception as e:
                    print(f"Error reading image in folder {folder_num}: {e}")
        
        # Calculate statistics
        total_images = sum(class_image_counts.values())
        avg_images_per_class = total_images / len(class_image_counts) if class_image_counts else 0
        
        self.dataset_stats = {
            'total_classes': len(self.class_paths),
            'total_images': total_images,
            'avg_images_per_class': avg_images_per_class,
            'min_images_per_class': min(class_image_counts.values()) if class_image_counts else 0,
            'max_images_per_class': max(class_image_counts.values()) if class_image_counts else 0,
            'class_image_counts': class_image_counts,
            'character_type_counts': self.mapping.get_character_type_counts()
        }
        
        print(f"\nDataset loaded successfully!")
        print(f"Total classes: {self.dataset_stats['total_classes']}")
        print(f"Total images: {self.dataset_stats['total_images']}")
        print(f"Average images per class: {self.dataset_stats['avg_images_per_class']:.2f}")
        print(f"Image count range: {self.dataset_stats['min_images_per_class']} - {self.dataset_stats['max_images_per_class']}")
        
        return {
            'class_paths': self.class_paths,
            'dataset_stats': self.dataset_stats
        }
    
    def get_class_info(self, label: int) -> Dict:
        """Get information about a specific class by label."""
        return self.class_paths.get(label, None)
    
    def get_all_image_paths(self) -> List[Tuple[str, int]]:
        """
        Get all image paths with their corresponding labels.
        
        Returns:
            List of tuples (image_path, label)
        """
        all_images = []
        for label, class_info in self.class_paths.items():
            for img_path in class_info['image_paths']:
                all_images.append((img_path, label))
        return all_images
    
    def get_reference_image(self, label: int) -> Optional[str]:
        """
        Get the reference image path for a given class label.
        
        Args:
            label: Class label (0-246)
        
        Returns:
            Path to reference image or None if not found
        """
        folder_num = label + 1
        
        # Get pronunciation for the label
        char_info = self.mapping.get_character_by_label(label)
        if not char_info:
            return None
        
        pronunciation = char_info['pronunciation']
        
        # Try different naming patterns and extensions
        patterns = [
            f"{folder_num}_{pronunciation}",  # e.g., 1_a.jpg
            f"{folder_num}-{pronunciation}",  # e.g., 119-Nī.jpg (some use dash)
            f"{folder_num}",                  # e.g., 1.jpg (fallback)
        ]
        
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for pattern in patterns:
            for ext in extensions:
                ref_path = self.reference_images_path / f"{pattern}{ext}"
                if ref_path.exists():
                    return str(ref_path)
        
        return None
    
    def save_dataset_info(self, output_path: str):
        """
        Save dataset information to JSON file.
        
        Args:
            output_path: Path to save the JSON file
        """
        dataset_info = {
            'dataset_path': str(self.dataset_path),
            'class_paths': self.class_paths,
            'dataset_stats': self.dataset_stats,
            'label_to_character': {
                k: v for k, v in self.mapping.label_to_character.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset info saved to: {output_path}")
    
    def create_dataframe(self) -> pd.DataFrame:
        """
        Create a pandas DataFrame with all image paths and labels.
        
        Returns:
            DataFrame with columns: image_path, label, tamil_char, pronunciation, type
        """
        data = []
        for label, class_info in self.class_paths.items():
            for img_path in class_info['image_paths']:
                data.append({
                    'image_path': img_path,
                    'label': label,
                    'tamil_char': class_info['tamil_char'],
                    'pronunciation': class_info['pronunciation'],
                    'type': class_info['type'],
                    'folder_num': class_info['folder_num']
                })
        
        df = pd.DataFrame(data)
        return df
    
    def get_dataset_summary(self) -> str:
        """
        Generate a detailed summary of the dataset.
        
        Returns:
            Formatted string with dataset summary
        """
        summary = []
        summary.append("=" * 70)
        summary.append("TLFS23 DATASET SUMMARY")
        summary.append("=" * 70)
        summary.append(f"Dataset Path: {self.dataset_path}")
        summary.append(f"Total Classes: {self.dataset_stats['total_classes']}")
        summary.append(f"Total Images: {self.dataset_stats['total_images']:,}")
        summary.append(f"Average Images per Class: {self.dataset_stats['avg_images_per_class']:.2f}")
        summary.append(f"Min Images per Class: {self.dataset_stats['min_images_per_class']}")
        summary.append(f"Max Images per Class: {self.dataset_stats['max_images_per_class']}")
        summary.append("\nCharacter Type Distribution:")
        for char_type, count in self.dataset_stats['character_type_counts'].items():
            summary.append(f"  - {char_type.capitalize()}: {count} classes")
        summary.append("=" * 70)
        
        return "\n".join(summary)


def main():
    """
    Main function to demonstrate dataset loading.
    """
    # Set dataset path (modify this to your actual dataset path)
    dataset_path = r"c:\Users\Abirami Ramanathan\Desktop\tamil-signs\TLFS23 - Tamil Language Finger Spelling Image Dataset"
    
    # Initialize loader
    print("Initializing TLFS23 Dataset Loader...")
    loader = TLFS23DatasetLoader(dataset_path)
    
    # Load dataset structure
    dataset_info = loader.load_dataset_structure(validate_images=False)
    
    # Print summary
    print("\n" + loader.get_dataset_summary())
    
    # Create DataFrame
    print("\nCreating dataset DataFrame...")
    df = loader.create_dataframe()
    print(f"DataFrame shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head(10))
    
    # Save dataset info
    output_dir = Path(dataset_path).parent / "src" / "mod1" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving dataset information...")
    loader.save_dataset_info(str(output_dir / "dataset_info.json"))
    
    # Save DataFrame
    df.to_csv(str(output_dir / "dataset_dataframe.csv"), index=False, encoding='utf-8')
    print(f"DataFrame saved to: {output_dir / 'dataset_dataframe.csv'}")
    
    # Display some examples
    print("\n" + "=" * 70)
    print("SAMPLE CHARACTER MAPPINGS")
    print("=" * 70)
    for label in [0, 13, 31, 100, 150, 200, 246]:
        class_info = loader.get_class_info(label)
        if class_info:
            print(f"Label {label}: {class_info['tamil_char']} ({class_info['pronunciation']}) - "
                  f"{class_info['type']} - {class_info['image_count']} images")
    
    print("\nModule 1 execution completed successfully!")


if __name__ == "__main__":
    main()
