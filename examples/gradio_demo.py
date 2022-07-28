# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install gradio
"""

import gradio as gr
import operator
import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("shibing624/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")


def ai_text(texts):
    
    text_tokens = None
    with torch.no_grad():
        text_tokens = tokenizer(texts, padding=True, return_tensors='pt')
        outputs = model(**text_tokens)

    def get_errors(corrected_text, origin_text):
        sub_details = []
        for i, ori_char in enumerate(origin_text):
            if ori_char in [' ', '“', '”', '‘', '’', '\n', '…', '—', '擤']:
                # add unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            if i >= len(corrected_text):
                break
            if ori_char != corrected_text[i]:
                if ori_char.lower() == corrected_text[i]:
                    # pass english upper char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details

    result = []
    i = 0
    for ids, text in zip(outputs.logits, texts):
        
        _text = tokenizer.decode((torch.argmax(ids, dim=-1) * text_tokens.attention_mask[i]), skip_special_tokens=True).replace(' ', '')
        corrected_text, details = get_errors(_text, text)
        result.append((corrected_text, details))
        
        print(text, ' => ', corrected_text, details)
        i += 1
    return result


if __name__ == '__main__':
    print(ai_text('少先队员因该为老人让坐'))

    examples = [
        ['真麻烦你了。希望你们好好的跳无'],
        ['少先队员因该为老人让坐'],
        ['机七学习是人工智能领遇最能体现智能的一个分知'],
        ['今天心情很好'],
        ['他法语说的很好，的语也不错'],
        ['他们的吵翻很不错，再说他们做的咖喱鸡也好吃'],
    ]

    output_text = gr.outputs.Textbox()
    gr.Interface(ai_text, "textbox", output_text,
                 title="Chinese Spelling Correction Model shibing624/macbert4csc-base-chinese",
                 description="Copy or input error Chinese text. Submit and the machine will correct text.",
                 article="Link to <a href='https://github.com/shibing624/pycorrector' style='color:blue;' target='_blank\'>Github REPO</a>",
                 examples=examples).launch()
