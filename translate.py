import argparse
import time
import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import pdb
import dill as pickle
import argparse
from Models import get_model
from Beam import beam_search
from nltk.corpus import wordnet
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]
            
    return 0

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

def translate_sentence(sentence, model, opt, SRC, TRG):
    
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd is True:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = torch.tensor([indexed], dtype=torch.int64, device=opt.device)

    sentence = beam_search(sentence, model, SRC, TRG, opt)

    return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)

def translate(opt, model, SRC, TRG):
    if type(opt.text) == str:
        sentences = [opt.text]
    else:
        sentences = opt.text
    translated = []

    for sentence in sentences:
        translated.append(translate_sentence(sentence.lower() + '.', model, opt, SRC, TRG).capitalize())

    return translated

def clean(sentence):
    sentence = multiple_replace({'\u202f': '', '\xa0': '', '!': '', '.': '', '   ': ' ', '  ': ' '}, sentence)
    return sentence.strip()

def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score for single sentence.
    Then average BLEU scores.
    https://blog.csdn.net/weixin_44755244/article/details/102831602
    """
    score = []
    smoothie = SmoothingFunction().method1
    for ref, cand in zip(reference, candidate):
        ref, cand = clean(ref), clean(cand)
        ref = [ref.split(' ')]
        cand = cand.split(' ')
        score.append(sentence_bleu(ref, cand, smoothing_function=smoothie))
    return sum(score)/len(score)

# def calculate_bleu(reference, candidate):
#     """
#     Calculate BLEU score for all sentences
#     """
#     smoothie = SmoothingFunction().method1
#     reference = ' '.join(reference)
#     candidate = ' '.join(candidate)
#     reference = clean(reference)
#     candidate = clean(candidate)
#     reference = [reference.split(' ')]
#     candidate = candidate.split(' ')
#     return sentence_bleu(reference, candidate, smoothing_function=smoothie)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    
    opt = parser.parse_args()

    opt.device = 'cuda' if opt.no_cuda is False else 'cpu'
    opt.is_bleu = 'n'
 
    assert opt.k > 0
    assert opt.max_len > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    
    while True:
        opt.text =input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
        if opt.text=="q":
            break
        if opt.text=='$config':
            opt.is_bleu = input("Whether to calculate BLEU score (y/n):\n")
            continue
        if opt.text=='f':
            fpath =input("Enter file path to translate (path or relative path):\n")
            try:
                opt.text = open(fpath, encoding='utf-8').read().split('\n')
            except:
                print("error opening or reading text file")
                continue
            if opt.is_bleu=='y':
                ref_path = input("Enter file path of reference translation:\n")
                try:
                    reference = open(ref_path, encoding='utf-8').read().split('\n')
                except:
                    print("error opening or reading reference file")
                    continue
        else:
            if opt.is_bleu=='y':
                reference = [input("Enter reference translation:\n")]
        phrases = translate(opt, model, SRC, TRG)
        if opt.is_bleu=='y':
            bleu = calculate_bleu(reference, phrases)
        if len(phrases) > 50:
            phrases = phrases[:50]
        for phrase in phrases:
            print('> '+ phrase)
        if opt.is_bleu=='y':
            print('BLEU score: ', bleu)
        print('')

if __name__ == '__main__':
    main()
